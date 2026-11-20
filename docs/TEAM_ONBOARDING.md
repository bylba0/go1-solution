# Онбординг участника команды (Hackathon Go1)

Коротко: работаем в приватном **bylba0/go1-solution-dev**, собираем Docker-образ, проверяем задание на симуляторе
организаторов, присылаем pull request. Публичный bylba0/go1-solution и сдачу ведёт только капитан.
Справочник по ТЗ (топики, пороги, баллы) — у капитана в `ros2_ws/docs/TZ_SUMMARY.md`; первоисточник — PDF в
`go1-simulator-public/docs/parts_of_tz/`.

## 1. Доступ
1. Пришлите капитану свой GitHub-логин. Капитан добавляет вас:
   `gh api -X PUT repos/bylba0/go1-solution-dev/collaborators/<логин> -f permission=push`
2. Примите приглашение (письмо или https://github.com/bylba0/go1-solution-dev/invitations).
3. Почта в коммитах — только noreply: `git config user.email <id>+<логин>@users.noreply.github.com`
   (GitHub → Settings → Emails → «Keep my email addresses private»). Никаких паролей/токенов в репозитории.

## 2. Что поставить
- **Ubuntu 22.04 x86_64** или **Windows 11 + WSL2 (Ubuntu 22.04)**. Для T4/T5 нужен мощный CPU (≥ 8 ядер, ≥ 16 ГБ RAM),
  для обучения T3 — NVIDIA GPU. На ноутбуке с Iris Xe: T2/T3 в реальном времени, T4 в 4 раза медленнее, T5 в 14 раз.
- **Docker Engine + compose plugin** по официальной инструкции Docker для Ubuntu (Docker Desktop не нужен):
  https://docs.docker.com/engine/install/ubuntu/ , затем `sudo usermod -aG docker $USER` и перелогиниться.
  Проверка: `docker run --rm hello-world`, `docker compose version`.
- WSL2 с `networkingMode=mirrored` и VPN на весь трафик: если `dockerd` падает с «all predefined address pools have been
  fully subnetted», задайте свободные подсети в `/etc/docker/daemon.json`
  (`{"bip": "172.30.250.1/24", "default-address-pools": [{"base": "10.201.0.0/16", "size": 24}]}`) и `sudo systemctl restart docker`.
- git, Python 3.10+ (есть в Ubuntu 22.04). ROS 2 на хосте не обязателен — всё внутри образов.

## 3. Симулятор, сборка, запуск своего задания
```bash
git clone https://github.com/Urika000/go1-simulator-public.git ~/go1-simulator-public
docker pull registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0      # публичный, без логина
git clone https://github.com/bylba0/go1-solution-dev.git ~/go1-solution && cd ~/go1-solution
docker build --platform linux/amd64 -t go1-solution:dev .
tests/check_t1.sh go1-solution:dev                                              # должен быть «ИТОГ T1: OK»
```
Запуск задания TN вместе с симулятором (только headless; GUI в WSL не работает):
```bash
cd ~/go1-simulator-public
python3 run_local.py --task T4 --sim-image registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0 \
  --participant-image go1-solution:dev --headless --logs
```
Пока задание не заявлено в `supported_tasks`, `/entrypoint.sh TN` запускает пустую заглушку. Чтобы проверить своё задание,
не трогая манифест в git, подмените его монтированием (файл с `"supported_tasks": ["T2", "T4"]` у себя локально):
`--mount type=bind,src=$HOME/dev_manifest.json,dst=/opt/go1/solution_manifest.json,readonly` к команде `run_local.py`.
Симулятор внутри: `docker exec -it <Simulator=...> bash`, затем `source /opt/ros/humble/setup.bash;
source /home/ros/go1_ws/install/setup.bash; source /etc/go1_sim_env.sh; ros2 topic list --no-daemon`.

## 4. Куда класть код
| Что | Где |
|---|---|
| Пакет задания | `src/go1_<задание>/` (ament_python по умолчанию), напр. `src/go1_odom`, `src/go1_aruco`, `src/go1_nav`, `src/go1_rl` |
| Launch задания | `src/go1_bringup/launch/t<N>.launch.py` (его запускает `/entrypoint.sh TN`) + `exec_depend` на свой пакет в `src/go1_bringup/package.xml` |
| Параметры | YAML в пакете (`config/`), не хардкодить |
| Тесты | `src/go1_<задание>/test/` (pytest, без ROS где возможно) |
Правила для всех узлов: `use_sim_time:=true`; **не** перезаписывать `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, `CYCLONEDDS_URI`
(их задаёт запускатель организаторов); базовый фрейм робота — `trunk`; без интернета во время работы (всё — в образе);
долгие процессы — в tmux, не в переднем плане.

## 5. Ветки и pull request
1. Ветка от `main`: `git switch -c t4-odom` (имя: `t<N>-<кратко>`).
2. Маленькие коммиты `<кто>: <что сделано>`, push в свою ветку: `git push -u origin t4-odom`.
3. Pull request в `main` репозитория go1-solution-dev. В описании: что сделано, как проверено (команды и вывод),
   нагрузка CPU/RAM.
4. **Перед слиянием обязательно:** `docker build` + `tests/check_t1.sh go1-solution:dev` → «ИТОГ T1: OK», unit-тесты пакета
   зелёные, задание проверено на симуляторе организаторов (`run_local.py`). Сломанный T1 = недопуск всей команды.
5. Сливает капитан. В публичный bylba0/go1-solution пушит только капитан, когда готова версия для сдачи.

## 6. Что меняет только капитан
- **`Dockerfile`** (базовый образ, apt-пакеты, pip-зависимости): нужен пакет — напишите в PR, какой и зачем;
  все зависимости ставятся на этапе сборки, версии закреплены.
- **`solution_manifest.json` / `supported_tasks`**: задание добавляется **только когда работает** — проверено по критериям
  ТЗ на симуляторе организаторов. Заявленное, но сломанное задание хуже незаявленного.
- `entrypoint.sh`, `tests/check_t1.sh`, `models/unitree_go1/` (эталонная модель с SHA-256).

## 7. T3 (RL-политика) — что учесть
- **Хранение:** веса — `models/t3_policy/` (≤ 50 МБ на файл; больше — обсудить с капитаном), конфиги обучения —
  `config/t3/`, журналы и результаты экспериментов (графики, метрики, таблицы прогонов) — `docs/t3_training/`.
  ТЗ требует для доп. баллов подтверждение обучения: веса + конфиг обучения/дообучения + журналы. Сырые большие логи —
  во внешнем хранилище со ссылкой из `docs/t3_training/README.md`.
- Политика публикует в `/lowcmd_bare`, **через Safety Block** (`src/go1_safety`, обязателен по ТЗ) идёт в `/lowcmd`.
- **Safety Block ограничивает шаг сустава 4.5° за сообщение** (200 Гц → ≈ 900°/с), клипует углы с запасом 0.01 рад,
  kp 0..1000, kd 0..100, |dq| ≤ 20 рад/с, |tau| ≤ 100 (диапазоны организаторов; вне — обрезается). Цель, путь к которой проходит через самоколлизию, отклоняется
  (выход держит прошлую цель), NaN/Inf — отклоняются. Обучайте и проверяйте политику **с этими ограничениями**
  (в симе обучения повторите клип и лимит шага), иначе на испытании поведение будет отличаться.
- Проверка: `run_local.py --task T3 ...` + свой скрипт, публикующий `/cmd_vel` (`linear.x = preferred_speed_mps`).
  Падение по ТЗ: высота < 0.16 м или |roll|/|pitch| > 50° дольше 0.5 с; успех — ≥ 3 м за 90 с sim.
