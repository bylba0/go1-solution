#!/usr/bin/env bash
# Проверка T1 (ТЗ_T1, 3.1–3.4) так, как запускают организаторы (participant_runtime.py):
#   docker create --platform linux/amd64 --pull never --entrypoint /entrypoint.sh [--network ...] IMAGE <ping|T2>
# Проверяет:
#   1. образ linux/amd64, USER не root, /entrypoint.sh исполняемый;
#   2. ping с --network none: ≤ 10 с от начала запуска (create+start+wait), код 0, stdout = ровно один JSON-объект
#      с полями нужных типов, без лишнего текста; диагностика только в stderr;
#   3. неизвестный аргумент → ненулевой код;
#   4. T2 с --network none и окружением как у запускателя (ROS_DOMAIN_ID, RMW, inline CYCLONEDDS_URI)
#      стартует и живёт ≥ ALIVE_SEC секунд.
# Использование: tests/check_t1.sh [IMAGE] ; RUNS=3 ALIVE_SEC=12
set -uo pipefail

IMAGE="${1:-go1-solution:dev}"
RUNS="${RUNS:-3}"
ALIVE_SEC="${ALIVE_SEC:-12}"
PING_LIMIT="${PING_LIMIT:-10}"
NAME="go1-t1-check-$$"
TMP="$(mktemp -d)"
trap 'docker rm -f "$NAME" >/dev/null 2>&1; rm -rf "$TMP"' EXIT

fails=0
ok()   { echo "  [OK]   $*"; }
fail() { echo "  [FAIL] $*"; fails=$((fails + 1)); }
now()  { date +%s.%N; }

# CycloneDDS inline-XML как в run_local.py организаторов (loopback, multicast SPDP)
CYCLONE_XML='<?xml version="1.0" encoding="UTF-8"?><CycloneDDS xmlns="https://cdds.io/config"><Domain id="any"><General><Interfaces><NetworkInterface name="lo" multicast="true" /></Interfaces><AllowMulticast>spdp</AllowMulticast><EnableMulticastLoopback>true</EnableMulticastLoopback></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery><Internal><SocketReceiveBufferSize max="10MB" /></Internal></Domain></CycloneDDS>'

echo "== check_t1: $IMAGE"

# 1. Образ
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  [FAIL] образ $IMAGE не найден локально"; exit 1
fi
plat="$(docker image inspect "$IMAGE" --format '{{.Os}}/{{.Architecture}}')"
[[ "$plat" == "linux/amd64" ]] && ok "платформа $plat" || fail "платформа $plat (нужно linux/amd64)"
cfg_user="$(docker image inspect "$IMAGE" --format '{{.Config.User}}')"
uid="$(docker run --rm --platform linux/amd64 --network none --entrypoint id "$IMAGE" -u 2>/dev/null)"
if [[ -n "$uid" && "$uid" != "0" && -n "$cfg_user" && "$cfg_user" != "root" && "$cfg_user" != "0" ]]; then
  ok "пользователь не root (USER=$cfg_user, uid=$uid)"
else
  fail "пользователь root или не задан (USER='$cfg_user', uid='$uid')"
fi
if docker run --rm --platform linux/amd64 --network none --entrypoint test "$IMAGE" -x /entrypoint.sh; then
  ok "/entrypoint.sh исполняемый"
else
  fail "/entrypoint.sh отсутствует или не исполняемый"
fi

# 2. ping (как у организаторов: create → start → wait), RUNS раз
for ((i = 1; i <= RUNS; i++)); do
  docker rm -f "$NAME" >/dev/null 2>&1
  t0="$(now)"
  docker create --name "$NAME" --platform linux/amd64 --pull never --network none \
    --entrypoint /entrypoint.sh "$IMAGE" ping >/dev/null || { fail "ping #$i: docker create"; continue; }
  docker start "$NAME" >/dev/null
  rc="$(timeout $((PING_LIMIT + 20)) docker wait "$NAME")"
  t1="$(now)"
  docker logs "$NAME" >"$TMP/out" 2>"$TMP/err"
  dt="$(echo "$t1 - $t0" | bc)"; dts="$(printf '%.2f' "$dt")"
  verdict="$(python3 - "$TMP/out" <<'PY'
import json, sys
raw = open(sys.argv[1], encoding='utf-8').read()
try:
    obj = json.loads(raw)          # json.loads падает на любом лишнем тексте, кроме пробельных символов
except Exception as e:
    print(f'stdout не один JSON: {e}; stdout={raw[:200]!r}'); sys.exit(0)
errs = []
if not isinstance(obj, dict): errs.append('не объект')
else:
    need = {'protocol_version', 'supported_tasks', 'self_trained_rl', 'preferred_speed_mps'}
    if set(obj) != need: errs.append(f'поля {sorted(obj)} != {sorted(need)}')
    if obj.get('protocol_version') != '1.0': errs.append('protocol_version != "1.0"')
    st = obj.get('supported_tasks')
    if not isinstance(st, list) or not all(isinstance(t, str) and t in ('T2', 'T3', 'T4', 'T5') for t in st):
        errs.append('supported_tasks не массив строк из T2..T5')
    if not isinstance(obj.get('self_trained_rl'), bool): errs.append('self_trained_rl не bool')
    s = obj.get('preferred_speed_mps')
    if not (s is None or (isinstance(s, (int, float)) and not isinstance(s, bool))): errs.append('preferred_speed_mps не number|null')
print('OK ' + json.dumps(obj, separators=(',', ':')) if not errs else '; '.join(errs))
PY
)"
  if [[ "$rc" == "0" && "$verdict" == OK* ]] && (( $(echo "$dt <= $PING_LIMIT" | bc) )); then
    ok "ping #$i: ${dts} с, код 0, ${verdict#OK }"
  else
    fail "ping #$i: ${dts} с, код '$rc', $verdict"
  fi
  [[ -s "$TMP/err" ]] && echo "         stderr: $(head -c 200 "$TMP/err" | tr '\n' ' ')"
  docker rm -f "$NAME" >/dev/null 2>&1
done

# 3. Неизвестный аргумент
docker run --rm --platform linux/amd64 --network none --entrypoint /entrypoint.sh "$IMAGE" T9 >"$TMP/out" 2>"$TMP/err"
rc=$?
if [[ "$rc" != "0" && ! -s "$TMP/out" ]]; then ok "неизвестный аргумент: код $rc, stdout пуст"; else fail "неизвестный аргумент: код $rc, stdout='$(head -c 100 "$TMP/out")'"; fi

# 4. T2 без сети, окружение как у запускателя
docker rm -f "$NAME" >/dev/null 2>&1
docker create --name "$NAME" --platform linux/amd64 --pull never --network none \
  --env ROS_DOMAIN_ID=151 --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp --env "CYCLONEDDS_URI=$CYCLONE_XML" \
  --env ROS_LOCALHOST_ONLY=0 --env GO1_DDS_LOCAL=0 --env GUI=false \
  --entrypoint /entrypoint.sh "$IMAGE" T2 >/dev/null
docker start "$NAME" >/dev/null
sleep "$ALIVE_SEC"
running="$(docker inspect "$NAME" --format '{{.State.Running}}')"
logs="$(docker logs "$NAME" 2>&1 | tail -5 | tr '\n' ' ')"
if [[ "$running" == "true" ]]; then
  ok "T2 без сети живёт ≥ ${ALIVE_SEC} с"
else
  fail "T2 завершился раньше ${ALIVE_SEC} с (код $(docker inspect "$NAME" --format '{{.State.ExitCode}}')): $logs"
fi
echo "         лог T2: ${logs:0:300}"
t0="$(now)"; docker stop --time 5 "$NAME" >/dev/null; t1="$(now)"
dstop="$(printf '%.2f' "$(echo "$t1 - $t0" | bc)")"; code="$(docker inspect "$NAME" --format '{{.State.ExitCode}}')"
if (( $(echo "$dstop < 5" | bc) )); then ok "docker stop: ${dstop} с, код выхода $code (штатная остановка, без SIGKILL)"
else fail "docker stop: ${dstop} с — контейнер не завершился по SIGTERM (добит SIGKILL)"; fi

echo
if (( fails == 0 )); then echo "ИТОГ T1: OK"; else echo "ИТОГ T1: FAIL ($fails)"; exit 1; fi
