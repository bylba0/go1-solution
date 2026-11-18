#!/usr/bin/env python3
"""Диагностика связи с симулятором: `/entrypoint.sh check [clock_wait_s=30] [topic_wait_s=5]`.

Ждёт /clock до clock_wait_s секунд, затем до topic_wait_s секунд — по сообщению из тех /lowstate, /imu/data, /scan,
у которых есть издатели. Ожидание — по реальному времени. Печатает в stdout ОДИН JSON:
  rmw, domain_id, clock (bool), clock_after_s, sim_time, received {топик: bool}, topics {имя: types/publishers/subscribers}.
Код выхода: 0 — /clock пришёл, 1 — нет; 2 — неверные аргументы (argparse, сообщение в stderr).
"""
import argparse
import json
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, LaserScan
from unitree_go.msg import LowState

NODE = 'go1_link_check'
TOPICS = {'/lowstate': LowState, '/imu/data': Imu, '/scan': LaserScan}
DISCOVERY_S = 1.0          # пауза на обнаружение издателей перед решением, каких топиков ждать


def main():
    ap = argparse.ArgumentParser(prog='/entrypoint.sh check')
    ap.add_argument('clock_wait_s', nargs='?', type=float, default=30.0)
    ap.add_argument('topic_wait_s', nargs='?', type=float, default=5.0)
    args = ap.parse_args()  # ошибка аргументов: сообщение в stderr, код 2

    rclpy.init()
    node = rclpy.create_node(NODE)
    last = {}
    # BEST_EFFORT-подписка совместима с издателем любой надёжности
    for topic, msg_type in {'/clock': Clock, **TOPICS}.items():
        node.create_subscription(msg_type, topic, lambda msg, t=topic: last.__setitem__(t, msg), qos_profile_sensor_data)

    def spin_until(done, seconds):
        end = time.monotonic() + seconds
        while not done() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)

    def foreign(endpoints):
        return sum(e.node_name != NODE for e in endpoints)

    t0 = time.monotonic()
    spin_until(lambda: '/clock' in last, args.clock_wait_s)
    clock_after = time.monotonic() - t0
    t1 = time.monotonic()
    spin_until(lambda: time.monotonic() - t1 > DISCOVERY_S
               and all(t in last for t in TOPICS if node.count_publishers(t)), args.topic_wait_s)

    topics = {}
    for name, types in sorted(node.get_topic_names_and_types()):
        pubs = foreign(node.get_publishers_info_by_topic(name))
        subs = foreign(node.get_subscriptions_info_by_topic(name))
        if pubs or subs:                                  # топики только этого узла не показываем
            topics[name] = {'types': types, 'publishers': pubs, 'subscribers': subs}
    clock = last.get('/clock')
    print(json.dumps({
        'rmw': rclpy.get_rmw_implementation_identifier(),
        'domain_id': node.context.get_domain_id(),
        'clock': clock is not None,
        'clock_after_s': round(clock_after, 2) if clock else None,
        'sim_time': clock.clock.sec + clock.clock.nanosec * 1e-9 if clock else None,
        'received': {t: t in last for t in TOPICS},
        'topics': topics,
    }, indent=1, ensure_ascii=False))
    node.destroy_node()
    rclpy.try_shutdown()
    return 0 if clock else 1


if __name__ == '__main__':
    raise SystemExit(main())
