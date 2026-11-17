#!/usr/bin/env bash
# Живая демонстрация T2 с окном MuJoCo: симулятор организаторов (run_local.py --gui) + наш образ + judge_sim.
#   tools/live_demo.sh [IMAGE]          одна команда — всё запускается; закрытие окна/остановка — Enter или Ctrl+C
# Переменные: SIM_REPO (~/go1-simulator-public), SIM_IMAGE, WAIT (10 с до начала команд), CYCLES (5), CMD_RATE (100)
# Требуется графический сеанс: Linux с X11 или WSL2 с WSLg (DISPLAY=:0, /tmp/.X11-unix/X0).
# Этапы (что видно в окне) печатаются в консоль по ходу демонстрации.
set -uo pipefail
IMAGE="${1:-go1-solution:dev}"
SIM_REPO="${SIM_REPO:-$HOME/go1-simulator-public}"
SIM_IMAGE="${SIM_IMAGE:-registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0}"
WAIT="${WAIT:-10}"
CYCLES="${CYCLES:-5}"
CMD_RATE="${CMD_RATE:-100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$(mktemp)"
JUDGE="go1-t2-demo-judge-$$"

die() { echo "ОШИБКА: $*" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || die "DISPLAY не задан — нужен графический сеанс (Linux X11 или WSLg)"
[[ -S "/tmp/.X11-unix/X${DISPLAY#*:}" || -S /tmp/.X11-unix/X0 ]] || die "нет X11-сокета в /tmp/.X11-unix"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "нет образа $IMAGE (docker build -t $IMAGE .)"
docker image inspect "$SIM_IMAGE" >/dev/null 2>&1 || die "нет образа симулятора $SIM_IMAGE (docker pull)"
[[ -f "$SIM_REPO/run_local.py" ]] || die "нет $SIM_REPO/run_local.py (git clone go1-simulator-public)"
if docker ps --format '{{.Names}}' | grep -qE '^go1-'; then
  echo "Уже запущены контейнеры go1-*:"; docker ps --format '  {{.Names}}' | grep -E 'go1-'
  die "остановите их (Ctrl+C в окне run_local.py или docker stop <имя>) и запустите демонстрацию снова"
fi
XML="$(cat "$SIM_REPO/user_data/ros2_custom_config_setup/cyclonedds_local.xml")"

cleanup() {
  docker rm -f "$JUDGE" >/dev/null 2>&1
  if [[ -n "${RL_PID:-}" ]] && kill -0 "$RL_PID" 2>/dev/null; then
    echo "Останавливаю симулятор и контейнер решения..."
    kill -INT "$RL_PID" 2>/dev/null                    # run_local.py сам останавливает и удаляет свои контейнеры
    for _ in $(seq 1 30); do kill -0 "$RL_PID" 2>/dev/null || break; sleep 0.5; done
  fi
  rm -f "$LOG"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "== 1/4 Запуск симулятора организаторов с окном MuJoCo + решения $IMAGE"
( cd "$SIM_REPO" && exec python3 run_local.py --task T2 --gui --sim-image "$SIM_IMAGE" --participant-image "$IMAGE" --logs ) \
  > "$LOG" 2>&1 &
RL_PID=$!
for _ in $(seq 1 240); do
  grep -qE 'Pair started|Traceback|timeout|Error' "$LOG" && break
  kill -0 "$RL_PID" 2>/dev/null || break
  sleep 0.5
done
grep -q 'Pair started' "$LOG" || { echo "run_local.py не поднял пару:"; tail -20 "$LOG"; exit 1; }
DOMAIN="$(grep -oE 'domain=[0-9]+' "$LOG" | head -1 | cut -d= -f2)"
echo "   ROS_DOMAIN_ID=$DOMAIN"

echo "== 2/4 Жду окно MuJoCo"
if command -v xwininfo >/dev/null; then
  for _ in $(seq 1 60); do xwininfo -root -tree 2>/dev/null | grep -q '"MuJoCo' && break; sleep 0.5; done
  xwininfo -root -tree 2>/dev/null | grep -q '"MuJoCo' && echo "   окно открыто" || echo "   окно не найдено (смотрите на экран)"
else
  sleep 5
fi
grep -iE 'error|exception|glfw' "$LOG" | grep -viE 'alsa' | head -3 | sed 's/^/   лог: /'

echo "== 3/4 Робот стоит: Safety Block держит стартовую позу. Команды начнутся через $WAIT с"
for ((s = WAIT; s > 0; s--)); do printf '\r   %2d с ' "$s"; sleep 1; done; echo

echo "== 4/4 judge_sim: команды в /lowcmd_bare (фазы ниже), оценка выхода /lowcmd по ТЗ_T2"
docker run --rm --name "$JUDGE" --platform linux/amd64 --network host \
  --env "ROS_DOMAIN_ID=$DOMAIN" --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp --env "CYCLONEDDS_URI=$XML" \
  --mount "type=bind,src=$HERE,dst=/tools,readonly" --entrypoint bash "$IMAGE" \
  -c "source /opt/ros/humble/setup.bash && source /opt/go1_ws/install/setup.bash && \
      python3 /tools/judge_sim.py --external-sim --demo --announce --cycles $CYCLES --cmd-rate $CMD_RATE" 2>/dev/null
RC=$?

echo
echo "Статистика Safety Block:"
grep -E 'go1_safety_block' "$LOG" | grep -E 'stats:' | tail -1 | cut -c1-230 | sed 's/^/  /'
if [[ -t 0 ]]; then
  read -r -p "Демонстрация окончена. Enter — закрыть окно и остановить контейнеры... " _
else
  sleep 5
fi
exit $RC
