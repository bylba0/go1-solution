#!/usr/bin/env python3
"""Заглушки одометрии T4 для самопроверки стенда judge_t4.py. ЧАСТЬ СТЕНДА: истину использовать можно, в образ не входит.

  t4_stub_odom.py --mode zero|truth|rot10|drift [--rate 50] [--noise 0.005]

  zero   — поза всегда (0,0,0), курс 0                       → RPE ≈ 100 % (A и B), дрейф 0
  truth  — истина + шум σ (м) и 0,1° по курсу                 → RPE ≈ 0, дрейф ≈ 0
  rot10  — вся траектория (положения и ориентация) повёрнута на 10° вокруг старта
                                                              → A ≈ 2·sin 5° ≈ 17 %, B ≈ 0
  drift  — истина + уход 0,02 м/с по x и 0,5 °/с по курсу      → дрейф 0,2 м и 5° за 10 с (FAIL)
Публикует /odom и /tf (odom→trunk) с одинаковым stamp = текущее /clock, с частотой --rate по sim-времени;
поза истины (20 Гц) экстраполируется линейно на текущий момент.
"""
import argparse
import math
import random

import rclpy
import rclpy.executors
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def qz(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


class Stub(Node):
    def __init__(self, a):
        super().__init__('t4_stub_odom')
        self.a = a
        self.gt = []            # [(t, (x,y,z), q)]
        self.last = -1.0
        self.t0 = None
        self.start = None
        be = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, '/odom/ground_truth', self.on_gt, be)
        self.create_subscription(Clock, '/clock', self.on_clock, be)
        self.po = self.create_publisher(Odometry, '/odom', 50)
        self.pt = self.create_publisher(TFMessage, '/tf', 100)

    def on_gt(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.gt = (self.gt + [(t, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))])[-2:]
        if self.start is None:
            self.start = (p.x, p.y, p.z)

    def on_clock(self, m):
        t = m.clock.sec + m.clock.nanosec * 1e-9
        if not self.gt or t - self.last < 1.0 / self.a.rate - 1e-9:
            return
        self.last = t
        if self.t0 is None:
            self.t0 = t
        (t1, p1, q1) = self.gt[-1]
        p = list(p1)
        if len(self.gt) == 2:
            (t0, p0, _) = self.gt[0]
            if t1 > t0:
                k = min(max(t - t1, 0.0), 0.1) / (t1 - t0)
                p = [p1[i] + (p1[i] - p0[i]) * k for i in range(3)]
        q = q1
        mode = self.a.mode
        if mode == 'zero':
            p, q = [0.0, 0.0, 0.0], (0.0, 0.0, 0.0, 1.0)
        elif mode == 'truth':
            p = [v + random.gauss(0, self.a.noise) for v in p]
            q = qmul(qz(math.radians(random.gauss(0, 0.1))), q)
        elif mode == 'rot10':
            th = math.radians(10.0)
            c, s = math.cos(th), math.sin(th)
            dx, dy = p[0] - self.start[0], p[1] - self.start[1]
            p = [self.start[0] + c * dx - s * dy, self.start[1] + s * dx + c * dy, p[2]]
            q = qmul(qz(th), q)
        elif mode == 'drift':
            dt = t - self.t0
            p = [p[0] + 0.02 * dt, p[1], p[2]]
            q = qmul(qz(math.radians(0.5 * dt)), q)
        n = math.sqrt(sum(v * v for v in q))
        q = tuple(v / n for v in q)
        o = Odometry()
        o.header.stamp = m.clock
        o.header.frame_id, o.child_frame_id = 'odom', 'trunk'
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = p
        (o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z,
         o.pose.pose.orientation.w) = q
        tf = TransformStamped()
        tf.header = o.header
        tf.child_frame_id = 'trunk'
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = p
        tf.transform.rotation = o.pose.pose.orientation
        self.po.publish(o)
        self.pt.publish(TFMessage(transforms=[tf]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['zero', 'truth', 'rot10', 'drift'])
    ap.add_argument('--rate', type=float, default=50.0)
    ap.add_argument('--noise', type=float, default=0.005)
    a = ap.parse_args()
    rclpy.init()
    n = Stub(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
