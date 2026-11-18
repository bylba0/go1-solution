#!/usr/bin/env bash
# Публикация решения в Harbor организаторов и подготовка строк для личного кабинета (Инструкция по сдаче решения).
#   tools/publish.sh [--dry-run] <команда>     # имя команды (проекта) в Harbor, латиницей, как выдали организаторы
# Адрес образа по умолчанию: registry.cookcloud.ru/<команда>/<команда>  (так у нашей команды: kafedra_problem/kafedra_problem;
#   формат подтверждён организатором при сдаче T1, 18.11.2026). Шаблон из инструкции «go1-hackathon-remote-stage-<команда>/…»
#   для нас НЕВЕРЕН. Другой адрес — переменной: HARBOR_REPO=registry.cookcloud.ru/<проект>/<репозиторий> tools/publish.sh <команда>
# --dry-run: только шаг 1 (проверки + итоговый адрес), без сборки и push; отсутствие входа в Harbor — предупреждение, не ошибка.
# Что делает (по шагам, при любой ошибке — остановка):
#   1. проверки: чистая рабочая папка; текущий коммит уже есть в публичном репозитории (remote public);
#      docker login в реестр уже выполнен (скрипт НЕ логинится и паролей не хранит; логин/логаут — только владелец);
#   2. сборка образа из ТЕКУЩЕГО коммита (git archive → docker build, linux/amd64), тег <адрес>:<короткий хэш коммита>;
#   3. docker push, получение digest;
#   4. удаление локального образа и повторная загрузка ПО DIGEST; сверка ID образа и метки коммита;
#   5. tests/check_t1.sh на скачанном образе;
#   6. печать строк для личного кабинета (+ копия в ~/go1-submissions.log).
set -euo pipefail

DEFAULT_REGISTRY="registry.cookcloud.ru"
PUBLIC_REMOTE="${PUBLIC_REMOTE:-public}"
die() { echo "ОСТАНОВКА: $*" >&2; exit 1; }
step() { echo; echo "== $*"; }

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
TEAM="${1:-}"
[[ -n "$TEAM" ]] || die "укажите имя команды в Harbor: tools/publish.sh [--dry-run] <команда>"
[[ "$TEAM" =~ ^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$ ]] || die "имя команды — латиница в нижнем регистре, цифры, . _ - (получено: '$TEAM')"

# Адрес репозитория образа (без тега и digest)
REPO="${HARBOR_REPO:-$DEFAULT_REGISTRY/$TEAM/$TEAM}"
[[ "$REPO" =~ ^[a-z0-9.-]+(:[0-9]+)?(/[a-z0-9]([a-z0-9._-]*[a-z0-9])?)+$ ]] \
  || die "HARBOR_REPO — адрес без тега, вида реестр/проект/репозиторий в нижнем регистре (получено: '$REPO')"
[[ "$REPO" == */*/* ]] || die "адрес должен содержать реестр, проект и репозиторий: '$REPO'"
REGISTRY="${REPO%%/*}"

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
REPO_ROOT="$PWD"

# ---------------------------------------------------------------- 1. проверки
step "1/6 Проверки$([[ $DRY_RUN == 1 ]] && echo ' (dry-run: без сборки и push)')"
echo "   адрес образа: $REPO$([[ -n "${HARBOR_REPO:-}" ]] && echo '  (из HARBOR_REPO)' || echo '  (по умолчанию: реестр/<команда>/<команда>)')"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  git status --short | head -20
  die "рабочая папка не чистая — закоммитьте или уберите изменения (образ должен соответствовать коммиту)"
fi
COMMIT="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short HEAD)"
IMAGE="$REPO:$SHORT"
echo "   коммит: $COMMIT ($(git log -1 --format=%s | cut -c1-70))"
echo "   тег сборки: $IMAGE"

git remote get-url "$PUBLIC_REMOTE" >/dev/null 2>&1 || die "нет remote '$PUBLIC_REMOTE' (публичный репозиторий)"
git fetch -q "$PUBLIC_REMOTE" || die "не удалось получить $PUBLIC_REMOTE"
PUBLIC_BRANCH="$(git remote show "$PUBLIC_REMOTE" 2>/dev/null | sed -n 's/.*HEAD branch: //p')"
PUBLIC_BRANCH="${PUBLIC_BRANCH:-main}"
if ! git merge-base --is-ancestor "$COMMIT" "$PUBLIC_REMOTE/$PUBLIC_BRANCH"; then
  die "коммит $SHORT ещё не опубликован в $PUBLIC_REMOTE/$PUBLIC_BRANCH. Сначала обновите публичный репозиторий
          (по решению капитана: git push $PUBLIC_REMOTE main), затем запустите publish.sh снова."
fi
PUBLIC_URL="$(git remote get-url "$PUBLIC_REMOTE" | sed -E 's#^git@github.com:#https://github.com/#; s#\.git$##')"
echo "   коммит есть в $PUBLIC_REMOTE/$PUBLIC_BRANCH ($PUBLIC_URL)"

# docker login: только проверка наличия входа в $REGISTRY (без ввода и хранения паролей).
# Читается конфиг Docker ($DOCKER_CONFIG или ~/.docker/config.json) — только ключи реестров, без вывода содержимого.
if python3 - "$REGISTRY" <<'PY'
import json, os, subprocess, sys
reg = sys.argv[1]
path = os.path.join(os.environ.get('DOCKER_CONFIG', os.path.expanduser('~/.docker')), 'config.json')
try:
    cfg = json.load(open(path))
except Exception:
    sys.exit(1)
keys = {reg, 'https://' + reg, 'https://' + reg + '/', 'http://' + reg}
helper = (cfg.get('credHelpers') or {}).get(reg) or cfg.get('credsStore')
if helper:
    try:
        out = subprocess.run(['docker-credential-' + helper, 'list'], capture_output=True, text=True, timeout=10).stdout
        sys.exit(0 if any(k in json.loads(out or '{}') for k in keys) else 1)
    except Exception:
        sys.exit(1)
auths = cfg.get('auths') or {}
sys.exit(0 if any(k in auths and (auths[k] or {}).get('auth') for k in keys) else 1)
PY
then
  echo "   вход в $REGISTRY выполнен"
elif [[ $DRY_RUN == 1 ]]; then
  echo "   ВНИМАНИЕ: входа в $REGISTRY нет — перед настоящей публикацией владелец выполняет: docker login $REGISTRY"
else
  die "нет входа в $REGISTRY. Выполните сами: docker login $REGISTRY — и запустите снова"
fi

if [[ $DRY_RUN == 1 ]]; then
  echo
  echo "dry-run: проверки пройдены. Будет собран и отправлен $IMAGE; строка для кабинета:"
  echo "  docker pull --platform linux/amd64 $REPO@sha256:<digest после push>"
  exit 0
fi

# ---------------------------------------------------------------- 2. сборка из коммита
step "2/6 Сборка $IMAGE из коммита $SHORT"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
git archive --format=tar "$COMMIT" | tar -x -C "$CTX"
# --provenance/--sbom=false: без манифестов-аттестаций (unknown/unknown) — в реестре один манифест linux/amd64
docker build --platform linux/amd64 --provenance=false --sbom=false \
  --label "org.opencontainers.image.revision=$COMMIT" \
  --label "org.opencontainers.image.source=$PUBLIC_URL" \
  -t "$IMAGE" "$CTX"
PLAT="$(docker image inspect "$IMAGE" --format '{{.Os}}/{{.Architecture}}')"
[[ "$PLAT" == "linux/amd64" ]] || die "платформа образа $PLAT, нужна linux/amd64"
BUILT_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"

# ---------------------------------------------------------------- 3. push и digest
step "3/6 Отправка в Harbor"
docker push "$IMAGE"
DIGEST="$(docker image inspect "$IMAGE" --format '{{range .RepoDigests}}{{println .}}{{end}}' \
          | grep -F "$REPO@" | head -1 | sed 's/.*@//')"
[[ "$DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die "не удалось получить digest после push (RepoDigests: $(docker image inspect "$IMAGE" --format '{{.RepoDigests}}'))"
FINAL="$REPO@$DIGEST"
echo "   digest: $DIGEST"

# ---------------------------------------------------------------- 4. удалить и скачать по digest
step "4/6 Удаляю локальный образ и скачиваю заново по digest"
docker rmi "$IMAGE" >/dev/null
docker rmi "$FINAL" >/dev/null 2>&1 || true
if docker image inspect "$FINAL" >/dev/null 2>&1; then die "локальный образ не удалился — проверка загрузки была бы нечестной"; fi
docker pull --platform linux/amd64 "$FINAL"
PULLED_ID="$(docker image inspect "$FINAL" --format '{{.Id}}')"
PULLED_REV="$(docker image inspect "$FINAL" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
[[ "$PULLED_ID" == "$BUILT_ID" ]] || die "ID скачанного образа ($PULLED_ID) не совпадает с собранным ($BUILT_ID)"
[[ "$PULLED_REV" == "$COMMIT" ]] || die "метка коммита в образе ($PULLED_REV) не совпадает с $COMMIT"
echo "   скачан по digest; ID и метка коммита совпадают"

# ---------------------------------------------------------------- 5. T1 на скачанном образе
step "5/6 tests/check_t1.sh на скачанном образе"
"$REPO_ROOT/tests/check_t1.sh" "$FINAL" || die "check_t1.sh не прошёл на опубликованном образе — НЕ сдавайте его"

# ---------------------------------------------------------------- 6. строки для личного кабинета
COMMIT_URL="$PUBLIC_URL/commit/$COMMIT"
PULL_CMD="docker pull --platform linux/amd64 $FINAL"
TASKS="$(docker run --rm --network none --platform linux/amd64 "$FINAL" ping 2>/dev/null)"
step "6/6 Готово. Строки для личного кабинета:"
echo
echo "  GitHub (конкретный коммит):  $COMMIT_URL"
echo "  Команда загрузки образа:     $PULL_CMD"
echo
echo "  ping: $TASKS"
{
  echo "$(date '+%Y-%m-%d %H:%M:%S') команда=$TEAM коммит=$COMMIT"
  echo "  $COMMIT_URL"
  echo "  $PULL_CMD"
  echo "  ping: $TASKS"
} >> "$HOME/go1-submissions.log"
echo "  (копия — в ~/go1-submissions.log). Перед отправкой откройте ссылку на коммит в браузере."
