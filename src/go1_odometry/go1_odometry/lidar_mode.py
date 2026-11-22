"""T4, режим lidar: обёртка ядра lidar_core.Odometry для узла (без ROS).

Что добавляет к ядру (само ядро не правится):
  * проверку входов: NaN/Inf или нулевой кватернион, суставы не той формы, NaN-штамп, NaN-углы скана — отбрасываются
    (счётчик rejected); дальности NaN/Inf ядро отсеивает само (np.isfinite);
  * раскладку суставов по ИМЕНАМ (ядро ждёт FL, FR, RL, RR × hip, thigh, calf — LIDAR_JOINT_NAMES);
  * пакетную подачу событий по штампам (feed) и замер времени обработки сканов;
  * выход pose(t) → (xy, кватернион IMU) с проверкой на конечность.
Опора: on_contact не вызывается никогда (/go1_contacts не используем) — ядро берёт опору по высоте стоп
(стопа не выше самой низкой на CONTACT_Z = 2 см). Курс — мировой yaw IMU (ориентация ядра = кватернион IMU).
"""
import math
import time
from collections import deque

import numpy as np

from . import lidar_core

LIDAR_JOINT_NAMES = lidar_core.JOINTS          # FL, FR, RL, RR × hip, thigh, calf
_ORDER = {'imu': 0, 'js': 2, 'scan': 3}        # при равных штампах: IMU раньше суставов и скана (как в источнике)


def joint_index(names, order=LIDAR_JOINT_NAMES):
    """Индексы суставов order в списке имён names; None, если какого-то нет."""
    idx = {n: i for i, n in enumerate(names)}
    return [idx[n] for n in order] if all(n in idx for n in order) else None


def _finite(*vals):
    return all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in vals)


class LidarEstimator:
    def __init__(self, leg_scale=1.0, stats_len=2000):
        self.core = lidar_core.Odometry(use_scan=True, use_legs=True, leg_scale=float(leg_scale))
        self.rejected = 0
        self.scan_ms = deque(maxlen=stats_len)  # время обработки сканов (сопоставление), мс wall
        self.n_scans_in = 0

    # ------------------------------------------------------------ входы
    def set_imu(self, t, quat):
        q = np.asarray(quat, dtype=float)
        n = float(np.linalg.norm(q)) if q.shape == (4,) and _finite(q) else 0.0
        if n < 0.5 or not math.isfinite(t):
            self.rejected += 1
            return False
        self._timed(self.core.on_imu, t, q / n)   # на IMU ядро разбирает и ждавшие её сканы
        return True

    def set_joints(self, t, q):
        """q: 12 углов в порядке LIDAR_JOINT_NAMES."""
        q = np.asarray(q, dtype=float)
        if q.shape != (12,) or not _finite(q) or not math.isfinite(t):
            self.rejected += 1
            return False
        self.core.on_joints(t, q)
        return True

    def set_scan(self, t, ranges, angle_min, angle_increment):
        r = np.asarray(ranges, dtype=float)
        if r.ndim != 1 or not len(r) or not _finite(t, angle_min, angle_increment) or angle_increment == 0:
            self.rejected += 1
            return False
        self.n_scans_in += 1
        self._timed(self.core.on_scan, t, r, float(angle_min), float(angle_increment))
        return True

    def _timed(self, fn, *args):
        c = self.core
        before = c.n_match + c.n_skip + c.n_key
        t0 = time.perf_counter()
        fn(*args)
        if c.n_match + c.n_skip + c.n_key != before:
            self.scan_ms.append(1000 * (time.perf_counter() - t0))

    def feed(self, events):
        """events: ('imu', t, quat) | ('js', t, q12) | ('scan', t, ranges, angle_min, angle_increment) — по штампам."""
        handlers = {'imu': self.set_imu, 'js': self.set_joints, 'scan': self.set_scan}
        for e in sorted(events, key=lambda e: (e[1], _ORDER[e[0]])):
            handlers[e[0]](*e[1:])

    # ------------------------------------------------------------ выход
    def pose(self, t):
        """(xy (2,), кватернион (x, y, z, w) нормированный, мировой курс) в момент t или None."""
        p = self.core.pose(t)
        if p is None or self.core.q_imu is None:
            return None
        q = np.asarray(self.core.q_imu, dtype=float)
        q = q / np.linalg.norm(q)
        xy = np.array(p[:2])
        if not _finite(xy, q):
            return None
        return xy, q
