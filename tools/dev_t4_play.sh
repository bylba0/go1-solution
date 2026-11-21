#!/usr/bin/env bash
# Разработка T4: проигрывает ВХОДЫ из bag (/clock, /imu/data, /joint_states) в живой узел go1_odometry (хост,
# install/ этой рабочей копии), пишет /odom и /tf в bag и считает метрики tools/dev_t4_offline_eval.py odom.
# /clock берётся из записи (sim time), а не `--clock` (метки bag — wall time записи при RTF ≈ 0,18).
# Запуск (долго: длительность bag / RATE; запускать в tmux):
#   tools/dev_t4_play.sh ~/ros2_ws/data/t4_bags/t4_A_square_left [RATE=4] [ROS_DOMAIN_ID=41] [OUT_DIR=/tmp/t4_play]
# Окружение DDS — как для связи с контейнерами: RMW_IMPLEMENTATION=rmw_cyclonedds_cpp, CYCLONEDDS_URI (если задан).
BAG=${1:?bag}; RATE=${2:-4}; export ROS_DOMAIN_ID=${3:-41}; OUT=${4:-/tmp/t4_play}
# set -m: иначе фоновые процессы bash без управления заданиями игнорируют SIGINT и узел не останавливается
set -m
HERE=$(cd "$(dirname "$0")/.." && pwd)
NAME=$(basename "${BAG%/}")
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$HERE/install/setup.bash"
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
mkdir -p "$OUT"; rm -rf "${OUT:?}/rec_$NAME"
ros2 launch go1_bringup t4.launch.py > "$OUT/node_$NAME.log" 2>&1 &
NODE=$!
ros2 bag record -o "$OUT/rec_$NAME" /odom /tf > "$OUT/rec_$NAME.log" 2>&1 &
REC=$!
sleep 4
ros2 bag play "$BAG" --rate "$RATE" --topics /clock /imu/data /joint_states > "$OUT/play_$NAME.log" 2>&1
sleep 2
kill -INT "$REC" "$NODE"; wait
python3 "$HERE/tools/dev_t4_offline_eval.py" odom "$BAG" "$OUT/rec_$NAME" 2>&1 | grep -v '^\[INFO\]'
