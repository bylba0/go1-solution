#!/usr/bin/env bash
# T3 на НАСТОЯЩЕМ MuJoCo-симуляторе организаторов: на каждый эпизод — НОВАЯ пара контейнеров (ТЗ_T3 5.3):
# run_local.py --task T3 --headless --participant-image IMG (как у судьи: --network host, /entrypoint.sh T3),
# затем tools/judge_t3.py в контейнере из образа сима (там есть unitree_go) в том же ROS_DOMAIN_ID, --network host.
# Перед уборкой снимается лог контейнера решения (диагностика Safety Block) и docker stats. Рядом — сторож.
# Запускать в tmux sim (эпизод ≈ 2 мин wall). Один сим за раз.
#
#   tools/run_t3_with_organizers_sim.sh --image go1-solution:t3-int [--episodes 10] [--first 1] [--out DIR] [-- аргументы judge_t3]
#   переменные: SIM_REPO=~/go1-simulator-public  SIM_IMAGE=<digest>  DOMAIN=78  CPUS=<лимит CPU решения>
#               SPEED=0.3 (preferred_speed_mps; по умолчанию берётся из ping образа)
#               WATCHDOG=~/ros2_ws/tools/t4_recon/watchdog.sh  MEM_MIN_START_MB=1500
set -uo pipefail
IMAGE=""; EPISODES=10; FIRST=1; OUT=""; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --first) FIRST="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$IMAGE" ]] || { echo "нужен --image <tag|digest>" >&2; exit 2; }
SIM_REPO="${SIM_REPO:-$HOME/go1-simulator-public}"
SIM_IMAGE="${SIM_IMAGE:-registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim@sha256:c79ddbece6cc7d00f114a9a9a09be0bae061ebb35ca722bcbe51bf75fce1b2e3}"
DOMAIN="${DOMAIN:-78}"
WATCHDOG="${WATCHDOG:-$HOME/ros2_ws/tools/t4_recon/watchdog.sh}"
MEM_MIN_START_MB="${MEM_MIN_START_MB:-1500}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/tmp/t3_live_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

PING="$(docker run --rm --network none --entrypoint /entrypoint.sh "$IMAGE" ping 2>/dev/null)"
echo "ping: $PING" | tee "$OUT/ping.json.txt"
if [[ -z "${SPEED:-}" ]]; then
  SPEED="$(python3 -c 'import json,sys; v=json.loads(sys.argv[1]).get("preferred_speed_mps"); print(v if v is not None else "")' "$PING")"
fi
[[ -n "$SPEED" ]] || { echo "в ping нет preferred_speed_mps — T3 не заявлено" >&2; exit 2; }
docker image inspect -f '{{.Id}}' "$IMAGE" > "$OUT/image_id.txt"
echo "образ $IMAGE id $(cat "$OUT/image_id.txt"), скорость $SPEED м/с"

RL_PID=""; WD_PID=""; ST_PID=""; SIM=""; PART=""; JUDGE=""
cleanup_run() {
  [[ -n "$ST_PID" ]] && kill "$ST_PID" 2>/dev/null; ST_PID=""
  [[ -n "$JUDGE" ]] && docker rm -f "$JUDGE" >/dev/null 2>&1; JUDGE=""
  if [[ -n "$RL_PID" ]] && kill -0 "$RL_PID" 2>/dev/null; then
    kill -INT -- "-$RL_PID" 2>/dev/null             # run_local.py сам останавливает и удаляет свои контейнеры
    for _ in $(seq 1 120); do kill -0 "$RL_PID" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$RL_PID" 2>/dev/null && kill -TERM -- "-$RL_PID" 2>/dev/null
  fi
  RL_PID=""
  for c in $SIM $PART; do                            # только свои контейнеры, по имени
    if docker inspect "$c" >/dev/null 2>&1; then echo "   остался $c — удаляю"; docker rm -f "$c" >/dev/null 2>&1; fi
  done
  [[ -n "$WD_PID" ]] && kill "$WD_PID" 2>/dev/null; WD_PID=""
}
trap 'cleanup_run; exit 130' INT TERM
trap cleanup_run EXIT

declare -a RCS
LAST=$((FIRST + EPISODES - 1))
for ep in $(seq "$FIRST" "$LAST"); do
  R="$OUT/ep$(printf %02d "$ep")"; mkdir -p "$R"; SIM=""; PART=""
  echo "== эпизод $ep ($FIRST..$LAST): $IMAGE, domain $DOMAIN → $R"
  if [[ -n "$(docker ps -q --filter name=go1-local)" ]]; then
    echo "уже запущены контейнеры go1-local — второй сим не поднимаю:"; docker ps --filter name=go1-local; exit 3
  fi
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if (( avail < MEM_MIN_START_MB )); then echo "MemAvailable ${avail} МБ < ${MEM_MIN_START_MB} — не запускаю"; exit 3; fi
  ( cd "$SIM_REPO" && exec setsid python3 run_local.py --task T3 --sim-image "$SIM_IMAGE" --participant-image "$IMAGE" \
      --headless --domain "$DOMAIN" ) > "$R/run_local.log" 2>&1 &
  RL_PID=$!
  for _ in $(seq 1 1200); do
    [[ -z "$SIM" ]] && SIM="$(grep -oE '^Simulator=[^ ]+' "$R/run_local.log" | head -1 | cut -d= -f2)"
    [[ -z "$PART" ]] && PART="$(grep -oE '^Participant=[^ ]+' "$R/run_local.log" | head -1 | cut -d= -f2)"
    if [[ -n "$SIM" && -z "$WD_PID" ]] && docker inspect "$SIM" >/dev/null 2>&1; then
      "$WATCHDOG" "$SIM" "$R/watchdog.log" & WD_PID=$!
    fi
    grep -qE 'Pair started|Traceback|RuntimeError' "$R/run_local.log" && break
    kill -0 "$RL_PID" 2>/dev/null || break
    sleep 0.5
  done
  if ! grep -q 'Pair started' "$R/run_local.log"; then
    echo "run_local не поднял пару:"; tail -20 "$R/run_local.log"; cleanup_run; RCS+=("$ep:нет пары"); continue
  fi
  echo "   Simulator=$SIM Participant=$PART"
  if [[ -n "${CPUS:-}" ]]; then docker update --cpus "$CPUS" "$PART" >/dev/null && echo "   лимит CPU решения: $CPUS"; fi
  ( while docker inspect -f '{{.State.Running}}' "$SIM" 2>/dev/null | grep -q true; do
      docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' "$PART" "$SIM" 2>/dev/null; sleep 2; done ) > "$R/stats.txt" &
  ST_PID=$!

  JUDGE="t3judge-$$-$ep"
  docker run --rm --name "$JUDGE" --network host --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e ROS_DOMAIN_ID="$DOMAIN" -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e ROS_LOCALHOST_ONLY=0 \
    -e CYCLONEDDS_URI=file:///cfg/cyclonedds_local.xml -e PYTHONUNBUFFERED=1 \
    -v "$SIM_REPO/user_data/ros2_custom_config_setup:/cfg:ro" -v "$HERE:/stand:ro" -v "$R:/out" \
    --entrypoint bash "$SIM_IMAGE" -c 'source /opt/ros/humble/setup.bash && source /home/ros/go1_ws/install/setup.bash && exec python3 /stand/judge_t3.py --out /out "$@"' \
    judge --speed "$SPEED" "${EXTRA[@]}" 2>&1 | tee "$R/judge.log"
  RC=${PIPESTATUS[0]}; JUDGE=""
  RCS+=("$ep:rc=$RC")
  part_state="$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}}' "$PART" 2>/dev/null || echo 'нет контейнера')"
  echo "   контейнер решения в конце: $part_state"; echo "$part_state" > "$R/participant_state.txt"
  docker logs -t "$PART" > "$R/participant.log" 2>&1     # до уборки: диагностика Safety Block
  docker logs -t "$SIM" > "$R/simulator.log" 2>&1
  grep -iE 'stat|принят|отклон|клип|clip|reject' "$R/participant.log" | tail -3 | sed 's/^/   блок: /'
  for c in "$PART" "$SIM"; do
    grep "^$c " "$R/stats.txt" | awk -v n="$c" '{gsub("%","",$2); v=$2+0; s+=v; if(v>m)m=v; k++; mem=$3} END{if(k) printf "   %s: CPU средн. %.1f%%, макс %.1f%%, RAM %s\n", n, s/k, m, mem}'
  done | tee "$R/resources.txt"
  grep -q ALERT "$R/watchdog.log" 2>/dev/null && echo "   СТОРОЖ: $(grep ALERT "$R/watchdog.log")"
  cleanup_run
  left="$(docker ps -aq --filter "name=$SIM" --filter "name=$PART" | wc -l)"
  echo "   контейнеров эпизода после уборки: $left"
done
echo "== итог: ${RCS[*]}  (rc 0 — успех эпизода, 1 — неуспех, 2 — стенд не дождался)"
python3 - "$OUT" <<'EOF'
import json, glob, os, sys
rs = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], 'ep*/result.json'))):
    r = json.load(open(p)); rs.append(r); lc = r.get('lowcmd', {})
    print(f"{os.path.basename(os.path.dirname(p))}: успех={r.get('success')} t3м={r.get('t_3m_s')} путь={r.get('path_90s_m', 0):.2f} "
          f"падение={r.get('fall')} r/p={r.get('roll_max_deg', 0):.1f}/{r.get('pitch_max_deg', 0):.1f} zmin={r.get('z_min_m', 0):.3f} "
          f"min1s={lc.get('min_hz_1s')} gap={lc.get('max_gap_ms_sim')} NaN={lc.get('nan_inf_msgs_fields')}")
n = sum(1 for r in rs if r.get('success'))
print(f"успешных {n}/{len(rs)}; баллы за движение: {(5 if n >= 1 else 0) + (10 if n >= 7 else 0)} (≥1 → 5, ≥7 → +10)")
EOF
