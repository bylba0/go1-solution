#!/usr/bin/env bash
# T4 на НАСТОЯЩЕМ Gazebo-симуляторе организаторов: run_local.py поднимает Simulator (T4, headless, lockstep) + образ
# решения (как у судьи: --network host, --entrypoint /entrypoint.sh IMG T4), затем на хосте в том же ROS_DOMAIN_ID
# запускается tools/judge_t4.py live (сценарий /cmd_vel, запись, метрики). Каждый прогон — новые контейнеры.
# Запускать в tmux (прогон ~7–10 мин wall). Один сим за раз; рядом — сторож памяти/нагрузки.
#
#   tools/run_t4_with_organizers_sim.sh --image go1-solution:t4-dev [--runs 3] [--preset tz] [--out DIR] [-- доп. аргументы judge]
#   переменные: SIM_REPO=~/go1-simulator-public  SIM_IMAGE=<digest>  DOMAIN=77  CPUS=1.0 (лимит CPU решения, docker update)
#               WATCHDOG=~/ros2_ws/tools/t4_recon/watchdog.sh  MEM_MIN_START_MB=2000
set -uo pipefail
IMAGE=""; RUNS=1; PRESET=tz; OUT=""; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$IMAGE" ]] || { echo "нужен --image <tag|digest>" >&2; exit 2; }
SIM_REPO="${SIM_REPO:-$HOME/go1-simulator-public}"
SIM_IMAGE="${SIM_IMAGE:-registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim@sha256:c79ddbece6cc7d00f114a9a9a09be0bae061ebb35ca722bcbe51bf75fce1b2e3}"
DOMAIN="${DOMAIN:-77}"
WATCHDOG="${WATCHDOG:-$HOME/ros2_ws/tools/t4_recon/watchdog.sh}"
MEM_MIN_START_MB="${MEM_MIN_START_MB:-2000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/tmp/t4_live_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sg docker -c docker"   # в оболочке без группы docker
d() { if [[ "$DOCKER" == docker ]]; then docker "$@"; else sg docker -c "docker $(printf '%q ' "$@")"; fi; }

set +u; source /opt/ros/humble/setup.bash; set -u
export ROS_DOMAIN_ID="$DOMAIN" RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI="file://$SIM_REPO/user_data/ros2_custom_config_setup/cyclonedds_local.xml"
python3 -c "import matplotlib" 2>/dev/null || echo "ВНИМАНИЕ: на хосте нет matplotlib — PNG не будет"

RL_PID=""; WD_PID=""; ST_PID=""; SIM=""; PART=""
cleanup_run() {
  [[ -n "$ST_PID" ]] && kill "$ST_PID" 2>/dev/null; ST_PID=""
  if [[ -n "$RL_PID" ]] && kill -0 "$RL_PID" 2>/dev/null; then
    kill -INT -- "-$RL_PID" 2>/dev/null             # run_local.py сам останавливает и удаляет свои контейнеры
    for _ in $(seq 1 120); do kill -0 "$RL_PID" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$RL_PID" 2>/dev/null && kill -TERM -- "-$RL_PID" 2>/dev/null
  fi
  RL_PID=""
  for c in $SIM $PART; do                            # только свои контейнеры, по имени
    if d inspect "$c" >/dev/null 2>&1; then echo "   остался $c — удаляю"; d rm -f "$c" >/dev/null 2>&1; fi
  done
  [[ -n "$WD_PID" ]] && kill "$WD_PID" 2>/dev/null; WD_PID=""
}
trap 'cleanup_run; exit 130' INT TERM
trap cleanup_run EXIT

declare -a RCS
for run in $(seq 1 "$RUNS"); do
  R="$OUT/run$run"; mkdir -p "$R"; SIM=""; PART=""
  echo "== прогон $run/$RUNS: $IMAGE, пресет $PRESET, domain $DOMAIN → $R"
  if [[ -n "$(d ps -q --filter name=go1-local)" ]]; then
    echo "уже запущены контейнеры go1-local — второй сим не поднимаю:"; d ps --filter name=go1-local; exit 3
  fi
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if (( avail < MEM_MIN_START_MB )); then echo "MemAvailable ${avail} МБ < ${MEM_MIN_START_MB} — не запускаю"; exit 3; fi
  RLCMD="python3 run_local.py --task T4 --sim-image $SIM_IMAGE --participant-image $IMAGE --headless --domain $DOMAIN --logs"
  # setsid: своя группа процессов → SIGINT доходит до run_local.py и через sg
  if [[ "$DOCKER" == docker ]]; then ( cd "$SIM_REPO" && exec setsid $RLCMD ) > "$R/run_local.log" 2>&1 &
  else ( cd "$SIM_REPO" && exec setsid sg docker -c "$RLCMD" ) > "$R/run_local.log" 2>&1 &
  fi
  RL_PID=$!
  for _ in $(seq 1 1200); do
    [[ -z "$SIM" ]] && SIM="$(grep -oE '^Simulator=[^ ]+' "$R/run_local.log" | head -1 | cut -d= -f2)"
    [[ -z "$PART" ]] && PART="$(grep -oE '^Participant=[^ ]+' "$R/run_local.log" | head -1 | cut -d= -f2)"
    if [[ -n "$SIM" && -z "$WD_PID" ]] && d inspect "$SIM" >/dev/null 2>&1; then
      "$WATCHDOG" "$SIM" "$R/watchdog.log" & WD_PID=$!
    fi
    grep -qE 'Pair started|Traceback|RuntimeError' "$R/run_local.log" && break
    kill -0 "$RL_PID" 2>/dev/null || break
    sleep 0.5
  done
  if ! grep -q 'Pair started' "$R/run_local.log"; then
    echo "run_local не поднял пару:"; tail -20 "$R/run_local.log"; cleanup_run; RCS+=("$run:нет пары"); continue
  fi
  echo "   Simulator=$SIM Participant=$PART"
  if [[ -n "${CPUS:-}" ]]; then d update --cpus "$CPUS" "$PART" >/dev/null && echo "   лимит CPU решения: $CPUS"; fi
  ( while d inspect -f '{{.State.Running}}' "$SIM" 2>/dev/null | grep -q true; do
      d stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' "$PART" "$SIM" 2>/dev/null; sleep 2; done ) > "$R/stats.txt" &
  ST_PID=$!

  python3 "$HERE/judge_t4.py" live --preset "$PRESET" --out "$R" "${EXTRA[@]}" 2>&1 | tee "$R/judge.log"
  RC=${PIPESTATUS[0]}
  RCS+=("$run:rc=$RC")
  for t in /odom/ground_truth /cmd_vel /odom; do   # снимок графа для протокола (judge_t4 сам опрашивает граф весь прогон)
    echo "### ros2 topic info -v $t"; timeout 30 ros2 topic info -v "$t" 2>&1 | grep -E 'Node name|Endpoint type|Reliability|Count' 
  done > "$R/topic_info.txt"
  part_state="$(d inspect -f '{{.State.Status}} exit={{.State.ExitCode}}' "$PART" 2>/dev/null || echo 'нет контейнера')"
  echo "   контейнер решения в конце: $part_state"
  echo "$part_state" > "$R/participant_state.txt"
  echo "Нагрузка (docker stats, раз в ~2 с):"
  for c in "$PART" "$SIM"; do
    grep "^$c " "$R/stats.txt" | awk -v n="$c" '{gsub("%","",$2); v=$2+0; s+=v; if(v>m)m=v; k++; mem=$3} END{if(k) printf "  %s: CPU средн. %.1f%%, макс %.1f%%, RAM %s\n", n, s/k, m, mem}'
  done
  grep -q ALERT "$R/watchdog.log" 2>/dev/null && echo "   СТОРОЖ: $(grep ALERT "$R/watchdog.log")"
  cleanup_run
  left="$(d ps -aq --filter "name=$SIM" --filter "name=$PART" | wc -l)"
  echo "   контейнеров прогона после уборки: $left"
done
echo "== итог: ${RCS[*]}  (rc 0 — все критерии ТЗ PASS, 1 — есть FAIL, 2 — стенд не дождался сима)"
echo "   результаты: $OUT/run*/result.json, summary.txt, trajectory.png"
