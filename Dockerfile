# Hackathon Go1 — образ решения (Participant). Требования ТЗ_T1: linux/amd64, Ubuntu 22.04, ROS 2 Humble,
# rmw_cyclonedds_cpp, только CPU, не root, без интернета на испытаниях (всё скачивается здесь, на этапе сборки).
#
# Сборка:  docker build --platform linux/amd64 -t go1-solution:dev .
# Запасная база, если образ организаторов недоступен:  --build-arg BASE_IMAGE=ros:humble-ros-base
ARG BASE_IMAGE=registry.cookcloud.ru/base_images/ros2-base-humble-amd64:v1.0.1
# unitree_ros2: тот же коммит, что у симулятора организаторов (Dockerfile сима, UNITREE_ROS2_REF)
ARG UNITREE_ROS2_REF=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88

# ---------------------------------------------------------------- этап 1: сборка
FROM --platform=linux/amd64 ${BASE_IMAGE} AS build
ARG UNITREE_ROS2_REF
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential cmake python3-pip python3-setuptools \
        python3-colcon-common-extensions \
        ros-humble-rosidl-default-generators ros-humble-rosidl-generator-dds-idl ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Сообщения unitree_go: только cyclonedds_ws/src/unitree/unitree_go с закреплённого коммита
RUN git init -q /tmp/unitree_ros2 && cd /tmp/unitree_ros2 \
    && git remote add origin https://github.com/unitreerobotics/unitree_ros2.git \
    && git fetch -q --depth 1 origin "${UNITREE_ROS2_REF}" && git checkout -q FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${UNITREE_ROS2_REF}" \
    && test -f cyclonedds_ws/src/unitree/unitree_go/msg/LowCmd.msg \
    && mkdir -p /opt/go1_ws/src && cp -r cyclonedds_ws/src/unitree/unitree_go /opt/go1_ws/src/unitree_go \
    && rm -rf /tmp/unitree_ros2

COPY src/ /opt/go1_ws/src/
RUN source /opt/ros/humble/setup.bash && cd /opt/go1_ws \
    && colcon build --merge-install --install-base /opt/go1_ws/install \
         --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && rm -rf build log

# Python-колёса по lock-файлу симулятора (те же версии и sha256: mujoco==3.3.7 и др.); numpy и zipp — из apt
COPY config/standalone-python.lock /opt/wheels/requirements.lock
RUN python3 -m pip download --only-binary=:all: --no-deps --require-hashes \
        -r /opt/wheels/requirements.lock -d /opt/wheels \
    && python3 -m pip install --no-index --find-links=/opt/wheels --no-deps --require-hashes \
        --prefix=/opt/pyprefix -r /opt/wheels/requirements.lock

# ---------------------------------------------------------------- этап 2: итоговый образ
FROM --platform=linux/amd64 ${BASE_IMAGE} AS runtime
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rmw-cyclonedds-cpp python3-numpy python3-zipp \
    && rm -rf /var/lib/apt/lists/*

# Колёса установлены на этапе сборки (с проверкой хэшей) в отдельный префикс; pip в итоговом образе не нужен.
# pip из Debian с --prefix кладёт в <prefix>/local/lib/python3.10/dist-packages → копируем в /usr/local (есть в sys.path). Проверяем версию и что numpy остался 1.x (ROS Humble).
COPY --from=build /opt/pyprefix/local/ /usr/local/
RUN python3 -c "import mujoco, numpy; assert mujoco.__version__ == '3.3.7', mujoco.__version__; assert int(numpy.__version__.split('.')[0]) < 2"

COPY --from=build /opt/go1_ws/install /opt/go1_ws/install

# Модель Go1 и перечень пар коллизий (источник: models/unitree_go1/SOURCE.md) + проверка SHA-256
COPY models/unitree_go1 /opt/go1/models/unitree_go1
RUN cd /opt/go1/models/unitree_go1 && sha256sum -c --quiet SHA256SUMS \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_path('go1.xml'); assert m.nu == 12, m.nu"

COPY solution_manifest.json /opt/go1/solution_manifest.json
COPY --chmod=0755 entrypoint.sh /entrypoint.sh

# Проверки на этапе сборки: типы unitree_go, манифест, ping без ROS
RUN source /opt/ros/humble/setup.bash && source /opt/go1_ws/install/setup.bash \
    && python3 -c "from unitree_go.msg import LowCmd, LowState; assert len(LowCmd().motor_cmd) == 20" \
    && /entrypoint.sh ping | python3 -c "import json, sys; json.load(sys.stdin)"

# Не root: пользователь ros (UID 1000) есть в базе организаторов; в ros:humble-ros-base — создаём
RUN id -u ros >/dev/null 2>&1 || useradd -m -u 1000 -s /bin/bash ros
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    GO1_MODEL_DIR=/opt/go1/models/unitree_go1 \
    PYTHONUNBUFFERED=1
USER ros
WORKDIR /home/ros
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ping"]
