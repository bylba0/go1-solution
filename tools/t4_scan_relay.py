#!/usr/bin/env python3
"""Стенд T4: ретранслятор /scan → /scan_relay с искусственной пропажей сканов (проверка страховки режима lidar).

Только для стенда, в образ решения не входит. Решение на время теста слушает /scan_relay (тестовый образ-наследник).
Подписка /scan (RELIABLE) → публикация /scan_relay (RELIABLE), сообщение без изменений.
Пропажа: молчит --gap с sim, начиная
  --at-straight N  с начала N-й прямой сценария (N-й фронт vx > 0,05 в /cmd_vel от стенда-судьи), или
  --at-sim T       с момента sim T с (по /clock).
Лог (stdout, flush): начало/конец молчания по sim, сколько сообщений не передано, итог. Выход — по Ctrl-C или если
/clock не идёт --stall с wall после начала (сим остановлен).

  python3 tools/t4_scan_relay.py --at-straight 2 --gap 15
"""
import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan


class Relay(Node):
    def __init__(self, a):
        super().__init__('t4_scan_relay')
        self.a = a
        q = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10,
                       durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(LaserScan, a.dst, q)
        self.create_subscription(LaserScan, a.src, self.on_scan, q)
        self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.now = None
        self.clk_wall = time.monotonic()
        self.moving = False
        self.straights = 0
        self.t_off = a.at_sim          # начало молчания, sim с
        self.t_on = None
        self.state = 'wait'            # wait → gap → done
        self.n_in = self.n_out = self.n_drop = 0
        self.first_out = self.last_before = self.first_after = None

    def log(self, s):
        print(f'[relay] {s}', flush=True)

    def on_clock(self, m):
        t = m.clock.sec + m.clock.nanosec * 1e-9
        if t != self.now:
            self.clk_wall = time.monotonic()
        self.now = t

    def on_cmd(self, m):
        mv = m.linear.x > 0.05
        if mv and not self.moving:
            self.straights += 1
            self.log(f'sim {self.now}: начало прямой {self.straights}')
            if self.a.at_straight and self.straights == self.a.at_straight and self.t_off is None:
                self.t_off = self.now
        self.moving = mv

    def on_scan(self, m):
        self.n_in += 1
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        t = self.now if self.now is not None else st
        if self.state == 'wait' and self.t_off is not None and t >= self.t_off:
            self.state, self.t_on = 'gap', self.t_off + self.a.gap
            self.log(f'МОЛЧУ: sim {t:.3f} (stamp скана {st:.3f}), до sim {self.t_on:.3f}; последний переданный stamp '
                     f'{self.last_before}')
        if self.state == 'gap':
            if t < self.t_on:
                self.n_drop += 1
                return
            self.state = 'done'
            self.first_after = st
            self.log(f'ВОЗОБНОВИЛ: sim {t:.3f} (stamp {st:.3f}); не передано {self.n_drop} сообщ.; '
                     f'фактическая пауза по stamp {st - (self.last_before or st):.3f} с')
        self.pub.publish(m)
        self.n_out += 1
        if self.first_out is None:
            self.first_out = st
            self.log(f'первый скан передан: stamp {st:.3f}')
        if self.state == 'wait':
            self.last_before = st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='/scan')
    ap.add_argument('--dst', default='/scan_relay')
    ap.add_argument('--gap', type=float, default=15.0, help='длительность молчания, с sim')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--at-straight', type=int)
    g.add_argument('--at-sim', type=float)
    ap.add_argument('--stall', type=float, default=45.0, help='/clock не идёт столько wall с → выход')
    a = ap.parse_args()
    rclpy.init()
    n = Relay(a)
    n.log(f'{a.src} → {a.dst}, молчание {a.gap} с sim, '
          + (f'с начала прямой {a.at_straight}' if a.at_straight else f'с sim {a.at_sim}'))
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.1)
            if n.now is not None and time.monotonic() - n.clk_wall > a.stall:
                n.log(f'/clock не идёт {a.stall:.0f} с wall — выход')
                break
    except KeyboardInterrupt:
        pass
    n.log(f'итог: принято {n.n_in}, передано {n.n_out}, не передано {n.n_drop}, состояние {n.state}, '
          f'молчание sim {n.t_off} → {n.t_on}')
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
