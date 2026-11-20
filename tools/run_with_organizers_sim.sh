#!/usr/bin/env bash
# T2 на НАСТОЯЩЕМ MuJoCo-симуляторе организаторов: run_local.py поднимает Simulator + наш образ (как у организаторов),
# затем в том же ROS_DOMAIN_ID запускается tools/judge_sim.py --external-sim (время и состояние — от сима).
# Использование: tools/run_with_organizers_sim.sh [IMAGE] ; SIM_REPO=~/go1-simulator-public CMD_RATE=100
set -uo pipefail
IMAGE="${1:-go1-solution:dev}"
SIM_REPO="${SIM_REPO:-$HOME/go1-simulator-public}"
SIM_IMAGE="${SIM_IMAGE:-registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0}"
CMD_RATE="${CMD_RATE:-100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$(mktemp)"
JUDGE="go1-t2-judge-ext-$$"
XML="$(cat "$SIM_REPO/user_data/ros2_custom_config_setup/cyclonedds_local.xml")"

cleanup() {
  docker rm -f "$JUDGE" >/dev/null 2>&1
  if [[ -n "${RL_PID:-}" ]] && kill -0 "$RL_PID" 2>/dev/null; then
    kill -INT "$RL_PID" 2>/dev/null                    # run_local.py сам останавливает и удаляет свои контейнеры
    for _ in $(seq 1 30); do kill -0 "$RL_PID" 2>/dev/null || break; sleep 0.5; done
  fi
  [[ -n "${STATS_PID:-}" ]] && kill "$STATS_PID" 2>/dev/null
}
trap cleanup EXIT

echo "== T2 с симулятором организаторов: $SIM_IMAGE + $IMAGE"
( cd "$SIM_REPO" && exec python3 run_local.py --task T2 --sim-image "$SIM_IMAGE" --participant-image "$IMAGE" --headless --logs ) \
  > "$LOG" 2>&1 &
RL_PID=$!
for _ in $(seq 1 240); do
  grep -qE 'Pair started|Traceback|timeout|Error' "$LOG" && break
  sleep 0.5
done
if ! grep -q 'Pair started' "$LOG"; then echo "run_local не поднял пару:"; tail -20 "$LOG"; exit 1; fi
DOMAIN="$(grep -oE 'domain=[0-9]+' "$LOG" | head -1 | cut -d= -f2)"
PART="$(grep -oE 'Participant=[^ ]+' "$LOG" | head -1 | cut -d= -f2)"
SIM="$(grep -oE 'Simulator=[^ ]+' "$LOG" | head -1 | cut -d= -f2)"
echo "   domain=$DOMAIN  Simulator=$SIM  Participant=$PART"
sleep 4

( while docker inspect -f '{{.State.Running}}' "$PART" 2>/dev/null | grep -q true; do
    docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' "$PART" "$SIM" 2>/dev/null; sleep 1; done ) > "$LOG.stats" &
STATS_PID=$!

docker run --rm --name "$JUDGE" --platform linux/amd64 --network host \
  --env "ROS_DOMAIN_ID=$DOMAIN" --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp --env "CYCLONEDDS_URI=$XML" \
  --mount "type=bind,src=$HERE,dst=/tools,readonly" --entrypoint bash "$IMAGE" \
  -c "source /opt/ros/humble/setup.bash && source /opt/go1_ws/install/setup.bash && \
      python3 /tools/judge_sim.py --external-sim --cmd-rate $CMD_RATE"
RC=$?
kill "$STATS_PID" 2>/dev/null; STATS_PID=

echo
echo "Нагрузка (docker stats, раз в ~1 с):"
for c in "$PART" "$SIM"; do
  grep "^$c " "$LOG.stats" | awk -v n="$c" '{gsub("%","",$2); v=$2+0; s+=v; if(v>m)m=v; k++; mem=$3} END{if(k) printf "  %s: CPU средн. %.1f%%, макс %.1f%%, RAM %s\n", n, s/k, m, mem}'
done
[[ -n "${PART_LOG:-}" ]] && cp "$LOG" "$PART_LOG"   # полный лог run_local (Simulator + решение) — диагностика
echo "Лог Safety Block (из run_local):"
grep -E 'go1_safety_block|entrypoint' "$LOG" | grep -E 'stats:|Стартовая|запуск|Safety Block:' | tail -4 | cut -c1-220 | sed 's/^/  /'
grep -E 'LowCmd не приходит|держу позу' "$LOG" | head -2 | sed 's/^/  сим: /'
exit $RC
