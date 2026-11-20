#!/usr/bin/env bash
# Прогон T2 в Docker: контейнер решения (`/entrypoint.sh T2`, как у организаторов: --network host, их inline-XML CycloneDDS)
# + соседний контейнер с tools/judge_sim.py. Печатает таблицу баллов и нагрузку CPU контейнера решения.
# Использование: tools/run_judge_docker.sh [IMAGE] ; CMD_RATE=100 FORCE_T2=1 CPUS=0.5
#   CPUS — ограничение CPU контейнера решения (docker --cpus), имитирует слабое железо; пусто — без ограничения.
#   FORCE_T2=1 (по умолчанию) — подменяет манифест на {"supported_tasks":["T2"]} через GO1_MANIFEST, чтобы проверить T2
#   до того, как он заявлен в solution_manifest.json; FORCE_T2=0 — запуск как есть.
set -uo pipefail
IMAGE="${1:-go1-solution:dev}"
CMD_RATE="${CMD_RATE:-100}"
FORCE_T2="${FORCE_T2:-1}"
CPUS="${CPUS:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAIN=$((100 + RANDOM % 130))
PART="go1-t2-participant-$$"
JUDGE="go1-t2-judge-$$"
OUT="$(mktemp -d)"
chmod 777 "$OUT"
XML='<?xml version="1.0" encoding="UTF-8"?><CycloneDDS xmlns="https://cdds.io/config"><Domain id="any"><General><Interfaces><NetworkInterface name="lo" multicast="true" /></Interfaces><AllowMulticast>spdp</AllowMulticast><EnableMulticastLoopback>true</EnableMulticastLoopback></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery><Internal><SocketReceiveBufferSize max="10MB" /></Internal></Domain></CycloneDDS>'
ENV=(--env "ROS_DOMAIN_ID=$DOMAIN" --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp --env "CYCLONEDDS_URI=$XML"
     --env ROS_LOCALHOST_ONLY=0 --env GO1_DDS_LOCAL=0 --env GUI=false)
cleanup() { docker rm -f "$PART" "$JUDGE" >/dev/null 2>&1; [[ -n "${STATS_PID:-}" ]] && kill "$STATS_PID" 2>/dev/null; }
trap cleanup EXIT

EXTRA=()
[[ -n "$CPUS" ]] && EXTRA+=(--cpus "$CPUS")
if [[ "$FORCE_T2" == "1" ]]; then
  echo '{"protocol_version":"1.0","supported_tasks":["T2"],"self_trained_rl":false,"preferred_speed_mps":null}' > "$OUT/manifest.json"
  chmod 644 "$OUT/manifest.json"
  EXTRA+=(--mount "type=bind,src=$OUT/manifest.json,dst=/tmp/manifest.json,readonly" --env GO1_MANIFEST=/tmp/manifest.json)
fi

echo "== T2 judge_sim: образ $IMAGE, ROS_DOMAIN_ID=$DOMAIN, вход $CMD_RATE Гц, CPU решения: ${CPUS:-без ограничения}"
docker create --name "$PART" --platform linux/amd64 --pull never --network host "${ENV[@]}" "${EXTRA[@]}" \
  --entrypoint /entrypoint.sh "$IMAGE" T2 >/dev/null
docker start "$PART" >/dev/null
sleep 4

( while docker inspect -f '{{.State.Running}}' "$PART" 2>/dev/null | grep -q true; do
    docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}}' "$PART" 2>/dev/null; sleep 1; done ) > "$OUT/stats.txt" &
STATS_PID=$!

docker run --rm --name "$JUDGE" --platform linux/amd64 --network host "${ENV[@]}" \
  --mount "type=bind,src=$HERE,dst=/tools,readonly" --mount "type=bind,src=$OUT,dst=/out" \
  --entrypoint bash "$IMAGE" -c "source /opt/ros/humble/setup.bash && source /opt/go1_ws/install/setup.bash && \
    python3 /tools/judge_sim.py --cmd-rate $CMD_RATE --json /out/report.json"
RC=$?

kill "$STATS_PID" 2>/dev/null; STATS_PID=
echo
echo "Нагрузка контейнера решения (docker stats, раз в ~1 с):"
awk '{gsub("%","",$1); v=$1+0; s+=v; if(v>m)m=v; n++} END{if(n) printf "  CPU среднее %.1f%%, макс %.1f%% (100%% = 1 ядро), замеров %d\n", s/n, m, n}' "$OUT/stats.txt"
tail -1 "$OUT/stats.txt" | awk '{print "  RAM", $2, $3, $4}'
echo "Статистика Safety Block из лога:"
docker logs "$PART" 2>&1 | grep -E 'stats:|Стартовая поза|отклонена' | tail -4 | sed 's/^/  /'
[[ -n "${PART_LOG:-}" ]] && docker logs "$PART" > "$PART_LOG" 2>&1   # полный лог контейнера решения (диагностика)
[[ -f "$OUT/report.json" ]] && cp "$OUT/report.json" "${REPORT:-/tmp/t2_report.json}" 2>/dev/null
exit $RC
