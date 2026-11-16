#!/usr/bin/env bash
# /entrypoint.sh — контракт запуска Hackathon Go1 (ТЗ_T1, разделы 3.2–3.3).
#   ping      — без ROS: печатает в stdout ровно один JSON из манифеста и выходит с кодом 0 (≤ 10 с).
#   T2..T5    — запуск ROS 2-узлов задания (use_sim_time:=true) до остановки контейнера (TERM → SIGINT для launch).
#               Задание, которого нет в supported_tasks манифеста, запускает заглушку (пустой узел).
#   иначе     — сообщение в stderr, код 2.
# Всё диагностическое — только в stderr. ROS_DOMAIN_ID, RMW_IMPLEMENTATION, CYCLONEDDS_URI, ROS_LOCALHOST_ONLY
# задаёт запускатель организаторов — здесь они НЕ перезаписываются (только значение по умолчанию для RMW, если пусто).
set -o pipefail

MANIFEST="${GO1_MANIFEST:-/opt/go1/solution_manifest.json}"
WS_SETUP="${GO1_WS_SETUP:-/opt/go1_ws/install/setup.bash}"

log() { printf '[entrypoint] %s\n' "$*" >&2; }

# Печатает компактный JSON манифеста после проверки типов. Только стандартная библиотека Python.
print_ping() {
  python3 -S - "$MANIFEST" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    allowed = {'T2', 'T3', 'T4', 'T5'}
    out = {
        'protocol_version': m['protocol_version'],
        'supported_tasks': m['supported_tasks'],
        'self_trained_rl': m['self_trained_rl'],
        'preferred_speed_mps': m['preferred_speed_mps'],
    }
    assert out['protocol_version'] == '1.0', 'protocol_version must be "1.0"'
    assert isinstance(out['supported_tasks'], list), 'supported_tasks must be a list'
    assert all(isinstance(t, str) and t in allowed for t in out['supported_tasks']), 'supported_tasks: only T2..T5'
    assert len(set(out['supported_tasks'])) == len(out['supported_tasks']), 'supported_tasks: duplicates'
    assert isinstance(out['self_trained_rl'], bool), 'self_trained_rl must be bool'
    s = out['preferred_speed_mps']
    assert s is None or (isinstance(s, (int, float)) and not isinstance(s, bool)), 'preferred_speed_mps: number|null'
except Exception as e:  # noqa: BLE001
    print(f'[entrypoint] bad manifest {path}: {e}', file=sys.stderr)
    sys.exit(1)
sys.stdout.write(json.dumps(out, separators=(',', ':')) + '\n')
PY
}

task_supported() {
  python3 -S -c 'import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1]))["supported_tasks"] else 1)' \
    "$MANIFEST" "$1" 2>/dev/null
}

source_ros() {
  # setup.bash не совместим с `set -u`; окружение запускателя сохраняется (setup.bash его не трогает).
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  # shellcheck disable=SC1090
  source "$WS_SETUP"
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
  log "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>} RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION" \
      "CYCLONEDDS_URI=$([ -n "${CYCLONEDDS_URI:-}" ] && echo set || echo empty) user=$(id -u)"
}

# ros2 launch (Humble) в роли PID 1 не реагирует на SIGTERM от `docker stop` → контейнер добивается SIGKILL.
# Поэтому PID 1 остаётся этот bash: TERM/INT переводим в SIGINT для launch, он штатно гасит узлы.
run_launch() {
  # set -m: без управления заданиями bash запускает фоновый процесс с ИГНОРИРУЕМЫМ SIGINT (SigIgn) — launch бы его не услышал
  set -m
  ros2 launch "$@" &
  local child=$!
  trap 'log "сигнал остановки → SIGINT для ros2 launch"; kill -INT "$child" 2>/dev/null' TERM INT
  local rc=0
  while kill -0 "$child" 2>/dev/null; do
    wait "$child"; rc=$?
  done
  exit "$rc"
}

case "${1:-}" in
  ping)
    print_ping
    exit $?
    ;;
  T2|T3|T4|T5)
    TASK="$1"
    source_ros
    if task_supported "$TASK"; then
      log "запуск $TASK"
      run_launch go1_bringup "$(echo "$TASK" | tr 'T' 't').launch.py" use_sim_time:=true
    else
      log "$TASK не заявлено в supported_tasks — запускаю заглушку (пустой узел)"
      run_launch go1_bringup idle.launch.py task:="$TASK" use_sim_time:=true
    fi
    ;;
  *)
    log "неизвестный аргумент: '${1:-}'. Использование: /entrypoint.sh ping|T2|T3|T4|T5"
    exit 2
    ;;
esac
