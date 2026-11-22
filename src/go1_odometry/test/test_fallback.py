"""Страховка режима lidar (fallback.LidarFallback) без ROS, синтетика: переключение по таймауту, отсутствие скачков,
плавный возврат, большая поправка, повторные пропадания, штамп назад, зависший рабочий поток, NaN на входе."""
import math

import numpy as np
import pytest

from go1_odometry.fallback import LidarFallback

DT = 0.02            # тик публикации 50 Гц
V = np.array([0.2, 0.0])   # скорость робота, м/с
JUMP = 0.010         # м за тик: 0,2 м/с · 0,02 с = 4 мм + запас на ввод поправки (≤ 0,25 м / 1 с → 5 мм) и шум


def _run(fb, t0, t1, scans, lidar_err=lambda t: np.zeros(2), legs_scale=1.0, legs_rot=0.0, out=None, src=None):
    """Робот едет с V; lidar = истина + lidar_err(t), legs = истина, повёрнутая и масштабированная (дрейф legs).
    scans(t) — приходят ли сканы (20 Гц). Возвращает (t, выход, истина) по тикам."""
    out = [] if out is None else out
    c, s = math.cos(legs_rot), math.sin(legs_rot)
    R = legs_scale * np.array([[c, -s], [s, c]])
    t = t0
    while t < t1 - 1e-9:
        t = round(t + DT, 6)
        truth = V * t
        if scans(t) and round(t / 0.05, 6) == int(round(t / 0.05, 6)):
            fb._t_scan_sim = t
        xy = fb.update(t, truth + lidar_err(t), R @ truth, getattr(fb, '_t_scan_sim', None))
        out.append((t, xy.copy(), truth))
        if src is not None:
            src.append(fb.source)
    return out


def _jumps(rows):
    d = [np.linalg.norm((b[1] - a[1]) - (b[2] - a[2])) for a, b in zip(rows, rows[1:])]
    return max(d)


def test_switch_after_timeout_and_back():
    fb = LidarFallback(scan_timeout=1.0, resume_delay=0.5, blend_tau=1.0)
    src = []
    rows = _run(fb, 0.0, 10.0, lambda t: True, src=src)
    assert set(src) == {'lidar'} and fb.switches == 0
    rows = _run(fb, 10.0, 20.0, lambda t: t < 12.0 or t >= 15.0, out=rows, src=src)
    times = [r[0] for r in rows]
    first_legs = times[src.index('legs')]
    # сканы в синтетике — на тиках, кратных 0,1 с: последний в 11,9; таймаут 1,0 с → переход в первый тик после 12,9
    assert 12.9 < first_legs <= 12.92 + 1e-6, first_legs
    back = times[len(src) - 1 - src[::-1].index('legs')] + DT
    assert abs(back - 15.5) < 1e-6, back                      # сканы с 15,0 + resume_delay 0,5 с
    assert fb.switches == 2 and fb.source == 'lidar'


def test_no_jump_on_switches_with_drifting_legs_and_correction():
    """legs дрейфует (масштаб 0,9, поворот 5°), lidar сдвинут на 0,2 м после паузы: выход непрерывен."""
    fb = LidarFallback()
    rows = _run(fb, 0.0, 30.0, lambda t: not (10.0 <= t < 20.0), legs_scale=0.9, legs_rot=math.radians(5),
                lidar_err=lambda t: np.array([0.0, 0.2]) if t > 20 else np.zeros(2))
    assert fb.switches == 2
    assert _jumps(rows) < JUMP
    # поправка (выход − lidar) в возврате < max_correction — затухает: через 5 τ выход = lidar
    t, xy, truth = rows[-1]
    assert np.linalg.norm(xy - (truth + [0.0, 0.2])) < 0.25 * math.exp(-4.0)


def test_large_correction_is_kept_not_blended():
    """Расхождение на возврате больше max_correction — выход не тянется к позе ядра: наша поза + приращения lidar."""
    fb = LidarFallback(max_correction=0.25)
    rows = _run(fb, 0.0, 30.0, lambda t: not (10.0 <= t < 20.0),
                lidar_err=lambda t: np.array([0.5, 0.0]) if t > 20 else np.zeros(2))
    assert fb.keep_off and _jumps(rows) < JUMP
    t, xy, truth = rows[-1]
    assert np.linalg.norm(xy - truth) < 1e-9                 # legs точен, приращения lidar точны


def test_repeated_dropouts():
    fb = LidarFallback()
    gaps = [(5.0, 7.0), (9.0, 9.8), (12.0, 15.0), (15.6, 18.0)]   # вторая короче таймаута; четвёртая — сразу после
    rows = _run(fb, 0.0, 25.0, lambda t: not any(a <= t < b for a, b in gaps), legs_scale=0.95)
    # 9,0–9,8: пауза 0,8 с + шаг сканов < 1 с — без переключения; 15,6: возврат (15,5) уже был — снова на legs
    assert fb.switches == 6, fb.switches
    assert _jumps(rows) < JUMP
    assert np.linalg.norm(rows[-1][1] - rows[-1][2]) < 0.05


def test_no_scans_at_all_follows_legs_from_start():
    fb = LidarFallback()
    src = []
    rows = _run(fb, 0.0, 20.0, lambda t: False, legs_scale=0.9, src=src)
    assert src[0] == 'lidar' and src[-1] == 'legs' and fb.switches == 1
    # выход = legs (масштаб 0,9) от позы в момент переключения (1,0 с)
    t, xy, truth = rows[-1]
    assert np.allclose(xy, V * 1.02 + 0.9 * V * (t - 1.02), atol=0.005)
    assert fb.no_scan_s(t, None) == pytest.approx(t - DT)


def test_stamp_backwards_resets():
    fb = LidarFallback()
    _run(fb, 0.0, 5.0, lambda t: t < 2.0)
    assert fb.source == 'legs'
    fb._t_scan_sim = None
    xy = fb.update(0.5, np.array([1.0, 2.0]), np.array([0.0, 0.0]), None)
    assert fb.resets == 1 and fb.source == 'lidar'
    assert np.allclose(xy, [1.0, 2.0])                       # после сброса — снова lidar без поправки


def test_stale_worker_pose_switches_to_legs():
    """Сканы приходят, но поза рабочего потока старше таймаута (поток завис) — legs."""
    fb = LidarFallback()
    for k in range(200):
        t = k * DT
        fb.update(t, V * min(t, 1.0), V * t, t, lidar_t=min(t, 1.0))
    assert fb.source == 'legs'
    xy = fb.update(4.02, V * 1.0, V * 4.02, 4.02, lidar_t=1.0)
    # переход в 2,02 (возраст позы > 1 с) от выхода 0,2 м (застывшая поза), дальше — путь legs 0,4 м
    assert np.allclose(xy, [0.6, 0.0], atol=1e-6)


def test_nan_inputs_ignored():
    fb = LidarFallback()
    assert fb.update(0.02, None, None, None) is None          # выхода ещё нет
    xy = fb.update(0.04, np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.04)
    assert np.allclose(xy, [1.0, 1.0])
    assert np.allclose(fb.update(0.06, np.array([math.nan, 1.0]), np.array([0.0, math.inf]), 0.06), [1.0, 1.0])
    assert np.allclose(fb.update(math.nan, np.array([5.0, 5.0]), np.array([0.0, 0.0]), 0.06), [1.0, 1.0])
    with pytest.raises(ValueError):
        LidarFallback(scan_timeout=0.0)
