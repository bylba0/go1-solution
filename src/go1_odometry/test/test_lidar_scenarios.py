"""Режим lidar через обёртку LidarEstimator (как в узле): ноги + IMU + сканы комнаты, сбои входов.

Сценарии: пауза сканов на ходу (выход по ногам, после паузы — снова по сканам), лучи NaN/Inf в части скана, скан
целиком без отражений, пустой /scan, скачок штампа назад (сброс). Опора — без контактов (по высоте стоп), как в узле.
Синтетика (комната, обратная кинематика ног) — из test_lidar_core.py.
"""
import math

import numpy as np

from go1_odometry.lidar_mode import LIDAR_JOINT_NAMES, LidarEstimator
from test_lidar_core import ANG, LEG_TABLE, leg_ik, quat, rot3, scan

DTH = ANG[1] - ANG[0]


def walk(T=3.0, v=(0.12, 0.04), w=0.15, scan_ok=lambda t: True, spoil=None, est=None, t0=0.0,
         probe=()):
    """Рысь (опора FL+RR / FR+RL по 0,25 с), корпус едет со скоростью v и поворачивается с w; IMU 200 Гц, суставы
    250 Гц (порядок ядра, по именам), сканы 20 Гц, если scan_ok(t); поза запрашивается 200 Гц, как в узле. spoil(ranges, rng) портит скан.
    Возвращает оценщик, {t: (оценка xy, истина xy)} в моменты probe и в конце."""
    nom = {g: np.array([sx * 0.19, sy * 0.13, -0.3]) for g, (sx, sy) in LEG_TABLE.items()}
    est = est or LidarEstimator()
    rng = np.random.default_rng(3)
    feet_w, phase, out = {}, None, {}
    probe = {round(p * 1000) for p in probe} | {round(T * 1000)}
    for k in range(round(T * 1000) + 1):
        t = k / 1000
        ts = t0 + t
        c, yaw = np.r_[np.array(v) * t, 0.0], w * t
        R = rot3(0, 0, yaw)
        ph = int(t / 0.25) % 2
        stance = (('FL', 'RR'), ('FR', 'RL'))[ph]
        if ph != phase:
            phase = ph
            for g in stance:
                feet_w[g] = c + R @ (nom[g] + np.r_[v, 0.0] * 0.125)
        if k % 5 == 0:
            est.set_imu(ts, quat(0, 0, yaw))
        if k % 50 == 0 and scan_ok(t):
            r = scan(c[:2], yaw) + 0.01 * rng.standard_normal(720)
            if spoil is not None:
                r = spoil(r, rng)
            est.set_scan(ts, r, ANG[0], DTH)
        if k % 4 == 0:
            q = {}
            for g, (sx, sy) in LEG_TABLE.items():
                pb = R.T @ (feet_w[g] - c) if g in stance else nom[g] + np.r_[0.0, 0.0, 0.06]
                q[g] = leg_ik(pb, sx, sy)
            est.set_joints(ts, np.array([q[j[:2]][('hip', 'thigh', 'calf').index(j[3:-6])] for j in LIDAR_JOINT_NAMES]))
        if k % 5 == 0 or k in probe:                     # как таймер узла: поза на каждом тике 200 Гц
            p = est.pose(ts)
            if k in probe:
                out[t] = (None if p is None else p[0], c[:2].copy())
    return est, out


def _err(res):
    est, true = res
    return float(np.hypot(*(est - true)))


def test_scans_ok_baseline():
    """Опорный случай: всё приходит — ошибка конца < 2 см на 0,4 м пути."""
    est, out = walk()
    assert _err(out[3.0]) < 0.02
    assert est.core.n_match > 40 and est.rejected == 0


def test_scan_pause_output_follows_legs_then_recovers():
    """Сканов нет 1,5 с на ходу: выход не стоит, а идёт по ногам (ошибка < 2 см); после паузы сканы снова в работе."""
    est, out = walk(T=4.0, scan_ok=lambda t: not 1.0 <= t < 2.5, probe=(1.0, 2.45))
    moved = np.hypot(*(out[2.45][0] - out[1.0][0]))
    assert moved > 0.15                                  # истина за паузу — 0,19 м
    assert _err(out[2.45]) < 0.02
    n = est.core.n_match
    assert est.core.n_legs_out > 0                        # часть выхода — по ногам
    assert _err(out[4.0]) < 0.02 and n >= 45            # 81 скан минус 30 в паузе


def test_scan_rays_nan_inf_part():
    """30 % лучей NaN/Inf (нет отражения) — отсеиваются ядром, точность прежняя, входы не отбрасываются целиком."""
    def spoil(r, rng):
        r = r.copy()
        bad = rng.random(720) < 0.3
        r[bad] = np.where(rng.random(bad.sum()) < 0.5, np.nan, np.inf)
        return r
    est, out = walk(spoil=spoil)
    assert est.rejected == 0
    assert _err(out[3.0]) < 0.02


def test_scan_without_returns_follows_legs():
    """Все лучи Inf (открытое пространство): сопоставлять не с чем — выход по ногам, конечен, ошибка < 2 см."""
    est, out = walk(spoil=lambda r, rng: np.full_like(r, np.inf))
    assert est.core.n_match == 0
    assert _err(out[3.0]) < 0.02


def test_empty_scan_rejected():
    """Пустой /scan (ranges = []) отбрасывается обёрткой и не ломает расчёт."""
    est, out = walk(T=2.0, probe=(1.0,))
    before = est.rejected
    assert not est.set_scan(2.0001, [], -math.pi, DTH)
    assert est.rejected == before + 1
    est2, out2 = walk(T=2.0, spoil=lambda r, rng: r[:0])
    assert est2.rejected == 41 and est2.core.n_match == 0
    assert _err(out2[2.0]) < 0.02


def test_stamp_backwards_resets():
    """Штамп назад (перезапуск сима): ядро сбрасывается — поза снова от нуля, а не продолжает старую."""
    est, out = walk(T=2.0, t0=10.0)
    assert np.hypot(*out[2.0][0]) > 0.2 and est.core.n_match > 20
    est, out = walk(T=1.0, t0=0.0, est=est)              # те же входы со штампами от 0
    assert _err(out[1.0]) < 0.02
