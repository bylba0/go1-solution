# go1-solution

Решение команды для заочного этапа **Hackathon Go1** (Unitree Go1 в симуляции организаторов).
Образ решения — контейнер *Participant*: `linux/amd64`, Ubuntu 22.04, ROS 2 Humble, `rmw_cyclonedds_cpp`,
только CPU, пользователь не root, работает без интернета.

| Задание | Статус | Где |
|---|---|---|
| T1 — образ и `/entrypoint.sh ping` | готово | `Dockerfile`, `entrypoint.sh`, `solution_manifest.json`, `tests/check_t1.sh` |
| T2 — Safety Block | готово (judge_sim 12/12, в т.ч. с симулятором организаторов) | `src/go1_safety`, `tools/judge_sim.py`, `docs/T2_DESIGN.md` |
| T3, T4, T5 | не начато | — |

Заявленные задания перечислены в одном месте — `solution_manifest.json` (поле `supported_tasks`);
`/entrypoint.sh ping` печатает его без запуска ROS.

Новым участникам команды — [docs/TEAM_ONBOARDING.md](docs/TEAM_ONBOARDING.md).

## Репозитории

| Репозиторий | Доступ | Что в нём |
|---|---|---|
| [bylba0/go1-solution](https://github.com/bylba0/go1-solution) | публичный | только версии **для сдачи**: каждый коммит соответствует образу, отправленному организаторам по SHA-256 digest |
| `bylba0/go1-solution-dev` | приватный | рабочий репозиторий команды: весь текущий прогресс, пушится после каждого рабочего шага |
| `bylba0/go1-solution-personal` | приватный | личная копия капитана: только содержимое go1-solution-dev, без доступа команды; обновляется только по команде капитана |

В локальной копии капитана: remote `dev` → go1-solution-dev (по умолчанию для работы), remote `public` → go1-solution,
remote `personal` → go1-solution-personal.
В `public` пушим только по решению капитана, когда готова новая версия для сдачи (`git push public main`).
В `personal` — только по команде капитана: `git push personal --all` и теги, которые есть в dev (личные `backup/*` — не пушим).

## Сборка

```bash
docker build --platform linux/amd64 -t go1-solution:dev .
```

- База — `registry.cookcloud.ru/base_images/ros2-base-humble-amd64:v1.0.1` (та же, что у симулятора организаторов;
  скачивается без логина). Запасной вариант: `--build-arg BASE_IMAGE=ros:humble-ros-base`.
- Сообщения `unitree_go` берутся из [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2), коммит
  `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88` (как у симулятора), только `cyclonedds_ws/src/unitree/unitree_go`.
- Python-зависимости — `config/standalone-python.lock` из репозитория симулятора (`pip --require-hashes`,
  MuJoCo 3.3.7); numpy и zipp — из apt.
- Модель Go1 и перечень пар коллизий — `models/unitree_go1/` (источник и SHA-256 — `SOURCE.md`, `SHA256SUMS`;
  сборка падает при несовпадении хэшей).
- Всё скачивается на этапе сборки; во время работы контейнеру интернет не нужен.

## Проверка T1

```bash
tests/check_t1.sh go1-solution:dev
```

Повторяет запуск организаторов (`docker create --platform linux/amd64 --pull never --entrypoint /entrypoint.sh ...`)
и проверяет: платформу, пользователя не root, `ping` без сети (≤ 10 с, код 0, ровно один JSON нужных типов, 3 запуска),
код ошибки на неизвестный аргумент, запуск `T2` без сети с окружением как у запускателя (≥ 12 с).

## Проверка T2

```bash
# unit-тесты (в образе, та же MuJoCo 3.3.7, что у Judge)
docker run --rm --network none --mount type=bind,src=$PWD/src/go1_safety,dst=/src,readonly --entrypoint bash \
  go1-solution:dev -c "source /opt/ros/humble/setup.bash && source /opt/go1_ws/install/setup.bash && cd /tmp && python3 -m pytest /src/test -q"
# имитатор Judge: /entrypoint.sh T2 + judge_sim.py в соседнем контейнере, таблица баллов ТЗ_T2
tools/run_judge_docker.sh go1-solution:dev                 # CMD_RATE=500 — частый вход; CPUS=1.0 — лимит CPU решения
# то же на настоящем MuJoCo-симуляторе организаторов (run_local.py + judge_sim --external-sim)
tools/run_with_organizers_sim.sh go1-solution:dev
# живая демонстрация с окном MuJoCo (нужен X11/WSLg): фазы печатаются в консоль, в конце — таблица баллов
tools/live_demo.sh go1-solution:dev
```
`judge_sim` — не официальный Judge: сценарии (безопасная траектория, выход за пределы, скачок 30°, поза с самоколлизией,
NaN/Inf, обрыв входа) и пороги взяты из текста ТЗ_T2; коллизии проверяются независимо (исходный go1.xml, шаг 0.1°, > 1 мм).

## Запуск вместе с симулятором организаторов

```bash
git clone https://github.com/Urika000/go1-simulator-public.git && cd go1-simulator-public
export SIM_IMAGE=registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0
python3 run_local.py --task T2 --sim-image "$SIM_IMAGE" --participant-image go1-solution:dev --headless --logs
```

## Контракт `/entrypoint.sh`

| Аргумент | Поведение |
|---|---|
| `ping` | без ROS; stdout — один JSON `{"protocol_version":"1.0","supported_tasks":[...],"self_trained_rl":false,"preferred_speed_mps":null}`, код 0 |
| `T2`..`T5` | `ros2 launch --noninteractive go1_bringup tN.launch.py use_sim_time:=true` (остановка: TERM → SIGINT, узлам 1.5 с до SIGTERM и ещё 0.5 с до SIGKILL); незаявленное задание — узел-заглушка |
| `check [clock_wait_s=30] [topic_wait_s=5]` | диагностика связи с симулятором (не для испытаний): один JSON в stdout, код 0 — `/clock` пришёл, 1 — нет, 2 — неверные аргументы |
| другое | сообщение в stderr, код 2 |

Переменные `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, `CYCLONEDDS_URI`, `ROS_LOCALHOST_ONLY` задаёт запускатель
организаторов; образ их не перезаписывает.

**Режим `check`** — быстро понять, видит ли контейнер симулятор: ждёт `/clock` до `clock_wait_s` секунд, затем до
`topic_wait_s` — по сообщению из `/lowstate`, `/imu/data`, `/scan` (тех, у кого есть издатели), и печатает JSON:
RMW, домен, пришёл ли `/clock` и через сколько секунд, sim time, какие топики получены, список чужих топиков с числом
издателей/подписчиков. Пример (при запущенном симуляторе, в его домене):
```bash
docker run --rm --network host -e ROS_DOMAIN_ID=<домен сима> -e CYCLONEDDS_URI="$(cat <cyclonedds_local.xml>)" \
  go1-solution:dev check 20 5
```

При сборке образ проверяет манифест (`tools/validate_manifest.py`): ровно 4 поля протокола, их типы, задания из T2..T5
без повторов, `preferred_speed_mps` в [0.1, 0.6] только вместе с T3 и launch-файл `t<N>.launch.py` для каждого
заявленного задания — иначе сборка падает.

## Сдача

Порядок по «Инструкции по подготовке и сдаче решения» организаторов. Сдаёт капитан.

1. **Проверить версию** в рабочем репозитории: `docker build ...`, `tests/check_t1.sh go1-solution:dev` — «ИТОГ T1: OK»;
   для каждого задания из `supported_tasks` — его проверки (для T2: unit-тесты, `tools/run_judge_docker.sh`,
   `tools/run_with_organizers_sim.sh` — 12/12).
2. **Опубликовать исходники** этой версии в публичный репозиторий (решение капитана): `git push public main`.
   Публичную историю после сдачи не переписывать — каждая новая версия = новый коммит.
3. **Войти в Harbor** (капитан вводит учётку команды сам, в репозитории и скриптах пароли не хранятся):
   `docker login registry.cookcloud.ru`
4. **Опубликовать образ** одной командой из чистой рабочей папки на нужном коммите:
   ```bash
   tools/publish.sh <имя_команды_в_Harbor>
   ```
   Скрипт откажется, если рабочая папка не чистая, коммит ещё не в `public` или нет входа в Harbor. Иначе он соберёт
   образ из этого коммита (`git archive` → `docker build`, linux/amd64, без аттестаций), отправит
   `registry.cookcloud.ru/go1-hackathon-remote-stage-<команда>/<команда>:<короткий хэш>`, получит digest, удалит
   локальный образ, скачает его заново по digest, сверит ID и метку коммита и прогонит `tests/check_t1.sh`.
5. **Заполнить личный кабинет** строками, которые печатает скрипт (копия — в `~/go1-submissions.log`):
   - GitHub (конкретный коммит): `https://github.com/bylba0/go1-solution/commit/<полный SHA>`
   - Команда загрузки образа: `docker pull --platform linux/amd64 registry.cookcloud.ru/go1-hackathon-remote-stage-<команда>/<команда>@sha256:<digest>`
6. Открыть ссылку на коммит в браузере и убедиться, что она открывается.

## Лицензии

Код — MIT (`LICENSE`). Модель Unitree Go1 (MuJoCo Menagerie) — BSD-3-Clause, `models/unitree_go1/LICENSE`.
