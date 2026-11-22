# go1-solution

Решение команды для заочного этапа **Hackathon Go1** (Unitree Go1 в симуляции организаторов).
Образ решения — контейнер *Participant*: `linux/amd64`, Ubuntu 22.04, ROS 2 Humble, `rmw_cyclonedds_cpp`,
только CPU, пользователь не root, работает без интернета.

| Задание | Статус | Где |
|---|---|---|
| T1 — образ и `/entrypoint.sh ping` | готово | `Dockerfile`, `entrypoint.sh`, `solution_manifest.json`, `tests/check_t1.sh` |
| T2 — Safety Block | сдан 20.11.2026 (judge_sim 12/12, в т.ч. с симулятором организаторов) | `src/go1_safety`, `tools/judge_sim.py`, `docs/T2_DESIGN.md` |
| T4 — одометрия (`/odom` + TF `odom → trunk`) | готово, заявлено в манифесте (стенд: 5 живых + 4 bag, худшее RPE 11,1 %, 20/20); сдача — по решению владельца | `src/go1_odometry`, `tools/judge_t4.py`, `docs/T4_DESIGN.md`, `docs/T4_REPORT.md` |
| T3, T5 | не начато | — |

Заявленные задания перечислены в одном месте — `solution_manifest.json` (поле `supported_tasks`, сейчас `["T2", "T4"]`);
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

## Проверка T4

Узел `go1_odometry` (`/entrypoint.sh T4` → `go1_bringup/launch/t4.launch.py`) публикует `/odom` (`odom → trunk`) и
TF `odom → trunk` с частотой 50 Гц sim. Истину и `/cmd_vel` решение не использует. Два режима оценщика (параметр
`estimator.mode`):

| Режим | Ориентация | Положение | Подписки |
|---|---|---|---|
| `legs` (запасной) | кватернион `/imu/data` (мировой курс) | кинематика опорных ног по `/joint_states` | `/imu/data`, `/joint_states`, `/clock` (+ `/foot_force`, если включён) |
| `lidar` | кватернион `/imu/data` (мировой курс) | сопоставление сканов `/scan` + прогноз по ногам (ядро `lidar_core.py`, источник и лицензия — в его заголовке) | `/imu/data`, `/joint_states`, `/clock`, `/scan` (RELIABLE) |

Переключение: `estimator.mode: lidar` в `src/go1_odometry/config/odometry.yaml` или без правки файла —
`ros2 launch go1_bringup t4.launch.py estimator_mode:=lidar`. По умолчанию — `lidar` (принят 22.11: худший RPE 3,4 %, bag B 1,1 %); `legs` — запасной.
Устройство, параметры и риски — [docs/T4_DESIGN.md](docs/T4_DESIGN.md),
результаты прогонов — [docs/T4_REPORT.md](docs/T4_REPORT.md).

```bash
# unit-тесты расчёта без ROS (на хосте, нужен numpy)
cd src/go1_odometry && python3 -m pytest -q test/test_core.py test/test_lidar_core.py test/test_lidar_mode.py test/test_lidar_scenarios.py && cd -
# все тесты пакета — в образе
docker run --rm --network none --mount type=bind,src=$PWD/src/go1_odometry,dst=/src,readonly --entrypoint bash \
  go1-solution:dev -c "source /opt/ros/humble/setup.bash && source /opt/go1_ws/install/setup.bash && cd /tmp && python3 -m pytest /src/test -q"
# живой прогон на Gazebo-симуляторе организаторов (headless, lockstep) + стенд judge_t4.py live; запускать в tmux,
# один сим за раз, ~7–10 мин wall; пресеты: tz (2 м, влево), right (1,25 м, вправо), tz_open (1,5 м, без коррекции курса)
CPUS=1.0 tools/run_t4_with_organizers_sim.sh --image go1-solution:dev --preset tz --out /tmp/t4_live_tz
# офлайн по записанному bag (истина и /cmd_vel — из bag, оценку даёт образ решения)
tools/judge_t4.py bag <bag> --image go1-solution:dev --cpus 1.0 --rate 1.5 --out /tmp/t4_bag
# пересчёт метрик по сохранённой записи (в т.ч. с совмещением только положения)
tools/judge_t4.py reanalyze /tmp/t4_bag/record.npz --align position
```
`judge_t4.py` — не официальный Judge: истину (`/odom/ground_truth` публичного сима) он использует только сам — для
коррекции курса, RPE (A — после совмещения начальной позы, B — в системе начала участка), дрейфа за 10 с и частоты по
`header.stamp`. В образ стенд не входит.

## Запуск вместе с симулятором организаторов

```bash
git clone https://github.com/Urika000/go1-simulator-public.git && cd go1-simulator-public
export SIM_IMAGE=registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0
python3 run_local.py --task T2 --sim-image "$SIM_IMAGE" --participant-image go1-solution:dev --headless --logs
python3 run_local.py --task T4 --sim-image "$SIM_IMAGE" --participant-image go1-solution:dev --headless --logs   # Gazebo, lockstep
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

Порядок по «Инструкции по подготовке и сдаче решения» организаторов. Сдаёт капитан. У каждого задания свой этап сдачи
(свой срок, своя кнопка в личном кабинете, 3 попытки) — лишних публикаций не делать.

**Адрес образа в Harbor:** `registry.cookcloud.ru/<команда>/<команда>` — у нашей команды
`registry.cookcloud.ru/kafedra_problem/kafedra_problem` (приватный проект). Шаблон из инструкции
`go1-hackathon-remote-stage-<команда>/<команда>` для нас неверен; фактический формат подтвердил организатор при сдаче T1.

1. **Проверить версию** в рабочем репозитории: `docker build ...`, `tests/check_t1.sh go1-solution:dev` — «ИТОГ T1: OK»;
   для каждого задания из `supported_tasks` — его проверки (для T2: unit-тесты, `tools/run_judge_docker.sh`,
   `tools/run_with_organizers_sim.sh` — 12/12).
2. **Опубликовать исходники** этой версии в публичный репозиторий (решение капитана): `git push public main`.
   Публичную историю после сдачи не переписывать — каждая новая версия = новый коммит.
3. **Проверить без публикации:** `tools/publish.sh --dry-run <команда>` — чистая папка, коммит в `public`, вход в
   реестр (здесь — только предупреждение) и итоговый адрес; ничего не собирает и не отправляет.
4. **Войти в Harbor** (только капитан/владелец, учётку вводит сам; в репозитории и скриптах пароли не хранятся):
   `docker login registry.cookcloud.ru`
5. **Опубликовать образ** одной командой из чистой рабочей папки на нужном коммите:
   ```bash
   tools/publish.sh kafedra_problem
   # другой адрес, если организаторы выдадут иной проект/репозиторий:
   HARBOR_REPO=registry.cookcloud.ru/<проект>/<репозиторий> tools/publish.sh <команда>
   ```
   Скрипт откажется, если рабочая папка не чистая, коммит ещё не в `public` или нет входа в реестр. Иначе он соберёт
   образ из этого коммита (`git archive` → `docker build`, linux/amd64, без аттестаций), отправит
   `<адрес>:<короткий хэш>`, получит digest, удалит локальный образ, скачает его заново по digest, сверит ID и метку
   коммита и прогонит `tests/check_t1.sh`.
6. **Заполнить личный кабинет** строками, которые печатает скрипт (копия — в `~/go1-submissions.log`):
   - GitHub (конкретный коммит): `https://github.com/bylba0/go1-solution/commit/<полный SHA>`
   - Команда загрузки образа: `docker pull --platform linux/amd64 registry.cookcloud.ru/<команда>/<команда>@sha256:<digest>`
7. Открыть ссылку на коммит в браузере и убедиться, что она открывается. После сдачи — `docker logout registry.cookcloud.ru`.

**Сдано:** T1 — 18.11.2026, коммит `7f95a1aa721b6be45f4381f53d46d8c4f644a5fe`, образ
`registry.cookcloud.ru/kafedra_problem/kafedra_problem@sha256:e22295bd8940f57284a9833da55248fc35fb4540578f7dd184690df234ac1821`.

## Лицензии

Код — MIT (`LICENSE`). Модель Unitree Go1 (MuJoCo Menagerie) — BSD-3-Clause, `models/unitree_go1/LICENSE`.
