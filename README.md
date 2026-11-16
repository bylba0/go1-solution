# go1-solution

Решение команды для заочного этапа **Hackathon Go1** (Unitree Go1 в симуляции организаторов).
Образ решения — контейнер *Participant*: `linux/amd64`, Ubuntu 22.04, ROS 2 Humble, `rmw_cyclonedds_cpp`,
только CPU, пользователь не root, работает без интернета.

| Задание | Статус | Где |
|---|---|---|
| T1 — образ и `/entrypoint.sh ping` | в работе | `Dockerfile`, `entrypoint.sh`, `solution_manifest.json`, `tests/check_t1.sh` |
| T2 — Safety Block | в работе | `src/go1_safety` (план) |
| T3, T4, T5 | не начато | — |

Заявленные задания перечислены в одном месте — `solution_manifest.json` (поле `supported_tasks`);
`/entrypoint.sh ping` печатает его без запуска ROS.

## Репозитории

| Репозиторий | Доступ | Что в нём |
|---|---|---|
| [bylba0/go1-solution](https://github.com/bylba0/go1-solution) | публичный | только версии **для сдачи**: каждый коммит соответствует образу, отправленному организаторам по SHA-256 digest |
| `bylba0/go1-solution-dev` | приватный | рабочий репозиторий команды: весь текущий прогресс, пушится после каждого рабочего шага |

В локальной копии: remote `dev` → go1-solution-dev (по умолчанию для работы), remote `public` → go1-solution.
В `public` пушим только по решению капитана, когда готова новая версия для сдачи (`git push public main`).

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
| `T2`..`T5` | `ros2 launch go1_bringup tN.launch.py use_sim_time:=true`; незаявленное задание — узел-заглушка |
| другое | сообщение в stderr, код 2 |

Переменные `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, `CYCLONEDDS_URI`, `ROS_LOCALHOST_ONLY` задаёт запускатель
организаторов; образ их не перезаписывает.

## Лицензии

Код — MIT (`LICENSE`). Модель Unitree Go1 (MuJoCo Menagerie) — BSD-3-Clause, `models/unitree_go1/LICENSE`.
