#!/usr/bin/env python3
"""Стенд-судья T4 (одометрия): сценарий /cmd_vel, запись, метрики по ТЗ 6.2–6.5. Только для разработки, в образ не входит.

Истина — /odom/ground_truth (публичный сим, GO1_GAZEBO_GT_MODE=public). Используется ТОЛЬКО здесь, в стенде:
для коррекции курса (как может делать Judge, ТЗ 6.3), аварийной остановки и подсчёта RPE/дрейфа.

Режимы:
  live      сим и решение уже запущены в текущем ROS_DOMAIN_ID (см. tools/run_t4_with_organizers_sim.sh):
            ждём /go1_ready, ведём робота по пресету, пишем /odom, TF, истину, считаем метрики.
              judge_t4.py live --preset tz --out DIR
  bag       офлайн: проигрываем bag разведки (истина и /cmd_vel — из bag), оценку даёт заглушка или образ решения.
              judge_t4.py bag ~/ros2_ws/data/t4_bags/t4_A_square_left --stub truth --out DIR
              judge_t4.py bag <bag> --image go1-solution:t4-dev --out DIR
  reanalyze пересчёт метрик по сохранённой записи (DIR/record.npz).

Время критериев — sim (/clock и header.stamp); wall и RTF — отдельно, для справки.
Выход: DIR/result.json (PASS/FAIL по каждому критерию ТЗ и отдельно — пороги приёмки этапа 3), DIR/summary.txt,
DIR/trajectory.png, DIR/record.npz. Код возврата: 0 — все критерии ТЗ PASS, 1 — есть FAIL, 2 — ошибка стенда.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- пороги (одно место; ссылки на ТЗ_T4)
TZ = dict(
    rate_hz=20.0,          # 6.2: /odom и TF не менее 20 Гц sim
    max_gap_s=0.10,        # TODO(уточнить): в ТЗ нет; строже «20 Гц в среднем» — разрыв не больше двух периодов
    rpe_5=0.50,            # 6.5: средний RPE ≤ 50 % → 5 б.
    rpe_11=0.20,           # 6.5: средний RPE ≤ 20 % → ещё 6 б.
    drift_pos_m=0.10,      # 6.4: дрейф за 10 с ≤ 0,10 м
    drift_ang_deg=5.0,     # 6.4: ≤ 5° (считаем полным углом поворота — строже, чем по курсу)
)
STAGE3 = dict(rate_hz=40.0, rpe=0.35, drift_pos_m=0.03, drift_ang_deg=1.5)  # приёмка этапа 3 (тимлид)
QUAT_TOL = 1e-6            # |q| = 1 ± 1e-6
CONSIST_POS_M = 1e-3       # /odom и TF при одном stamp — одна поза
CONSIST_ANG_DEG = 0.1
STAMP_VS_CLOCK_S = 0.5     # /odom: stamp не дальше 0,5 с от /clock в момент приёма (ловит wall-время и отставание)
STAMP_VS_CLOCK_TF_S = 5.0  # TF: мягче — /tf несёт ~1 кГц sim суставов, очередь подписчика стенда отстаёт на секунды
                           # (артефакт стенда, замерено при проигрывании bag); wall-время в stamp ловит и так
# Частота и разрывы — по header.stamp (sim). По времени приёма (/clock стенда) — только справочно: в Python-стенде
# под нагрузкой callbacks /tf идут пачками, и «частота по приёму» занижается не по вине решения.
SEG_LEN_M = 1.0            # 6.4: участки ≈ 1 м фактического пути
SEG_MIN_TAIL_M = 0.5       # хвост короче — не оцениваем (TODO(уточнить) у Judge)
PATH_SMOOTH_S = 0.5        # длина пути — по горизонтали, по истине, сглаженной скользящим средним 0,5 с: раскачка корпуса
                           # на рыси (и подпрыгивание по z) раздувает «путь» по сырым 20 Гц в 1,3–1,7 раза → участки
                           # короче 1 м реального хода и ошибка занижается. Сглаживание — строже. TODO(уточнить) у Judge

V_LIN, W_ANG = 0.2, 0.3    # 6.3: 0,2 м/с и 0,3 рад/с
STAND_NODES = {'t4_judge', 't4_stub_odom', 'rosbag2_player'}

# ---------------------------------------------------------------- комната odom_validation (docs/T4_RECON.md, п. f)
WALLS = dict(xmin=-3.40, xmax=8.40, ymin=-3.09, ymax=8.10)   # внутренние грани стен; на y −3,1 / 8,11 — детали на стенах
BOXES = [  # имя, центр x, y, размер x, y
    ('shelf_west_a', -2.55, -0.3, 0.55, 1.3), ('shelf_west_b', -2.55, 4.35, 0.65, 1.0),
    ('cabinet_east', 7.5, 1.4, 0.65, 1.25), ('crate_east', 7.48, 5.25, 0.85, 0.85),
    ('desk_north', 2.5, 7.4, 1.65, 0.8), ('crate_south', 1.0, -2.6, 0.9, 0.9)]
CYLS = [('pillar_nw', -2.55, 7.25, 0.33), ('pillar_se', 7.45, -2.5, 0.37), ('pillar_sw', -2.5, -2.45, 0.3)]
ESTOP_CLEAR_M = 0.6        # корпус ближе — аварийная остановка

PRESETS = {
    # «как в ТЗ»: квадрат влево, Judge замыкает курс по истине (ТЗ 6.3 допускает); сторона не задана — 2 м
    'tz': dict(side=2.0, turn='left', correct=True, still0=5.0, transition=5.0, drift=10.0),
    # без коррекции (как разведка A): CHAMP уходит по курсу на 12–47°, поворачивает на 75–86°
    'tz_open': dict(side=1.5, turn='left', correct=False, still0=5.0, transition=5.0, drift=10.0),
    # правые повороты — в −y, до стены ~3,1 м и ящик crate_south
    'right': dict(side=1.25, turn='right', correct=True, still0=5.0, transition=5.0, drift=10.0),
    # короткий для быстрой проверки
    'short': dict(side=1.0, turn='left', correct=True, still0=5.0, transition=5.0, drift=10.0),
}


# ---------------------------------------------------------------- геометрия
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_to_mat(q):
    """q: (..., 4) в порядке x, y, z, w → (..., 3, 3)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def yaw_of(q):
    q = np.asarray(q, float)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot_angle(Ra, Rb):
    """Полный угол поворота между ориентациями, рад."""
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(math.acos(max(-1.0, min(1.0, c))))


def interp_pose(ts, P, Q, t, max_gap=0.5):
    """Поза оценки в момент t (линейно по положению, nlerp по кватерниону). None — нет данных рядом."""
    if len(ts) == 0 or t < ts[0] - 1e-9 or t > ts[-1] + 1e-9:
        return None
    i = int(np.searchsorted(ts, t))
    if i < len(ts) and abs(ts[i] - t) < 1e-9:
        return P[i], Q[i]
    i0, i1 = max(i - 1, 0), min(i, len(ts) - 1)
    if ts[i1] - ts[i0] > max_gap:
        return None
    a = 0.0 if ts[i1] == ts[i0] else (t - ts[i0]) / (ts[i1] - ts[i0])
    q0, q1 = Q[i0], Q[i1] if np.dot(Q[i0], Q[i1]) >= 0 else -Q[i1]
    q = (1 - a) * q0 + a * q1
    return (1 - a) * P[i0] + a * P[i1], q / np.linalg.norm(q)


def clearance(x, y):
    """Расстояние от центра корпуса до ближайшей стены/препятствия и имя."""
    best = (min(x - WALLS['xmin'], WALLS['xmax'] - x, y - WALLS['ymin'], WALLS['ymax'] - y), 'стена')
    for name, cx, cy, sx, sy in BOXES:
        dx, dy = max(abs(x - cx) - sx / 2, 0), max(abs(y - cy) - sy / 2, 0)
        best = min(best, (math.hypot(dx, dy), name))
    for name, cx, cy, r in CYLS:
        best = min(best, (math.hypot(x - cx, y - cy) - r, name))
    return best


# ---------------------------------------------------------------- метрики
def window_rate(t, t0, t1, step=0.05):
    """Минимум сообщений в скользящем окне 1 с внутри [t0, t1] и максимальный разрыв (с учётом краёв)."""
    t = np.sort(t[(t >= t0) & (t <= t1)])
    if t1 - t0 < 1.0:
        return None, None
    starts = np.arange(t0, t1 - 1.0 + 1e-9, step)
    if len(t) == 0:
        return 0, t1 - t0
    cnt = np.searchsorted(t, starts + 1.0, 'left') - np.searchsorted(t, starts, 'left')
    gaps = np.diff(np.concatenate([[t0], t, [t1]]))
    return int(cnt.min()), float(gaps.max())


def stream_checks(name, a, t0, t1, clock_tol=STAMP_VS_CLOCK_S):
    """a: dict массивов одного потока (/odom или TF odom→trunk)."""
    r = {'name': name, 'count_total': int(len(a['stamp']))}
    sel = (a['stamp'] >= t0) & (a['stamp'] <= t1)
    r['count_eval'] = int(sel.sum())
    r['eval_window_s'] = round(t1 - t0, 3)
    mn_s, gap_s = window_rate(a['stamp'], t0, t1)
    mn_r, gap_r = window_rate(a['recv'], t0, t1)
    r['min_msgs_per_1s_stamp'], r['max_gap_s_stamp'] = mn_s, gap_s
    r['min_msgs_per_1s_recv'], r['max_gap_s_recv'] = mn_r, gap_r
    r['min_rate_hz'] = mn_s if mn_s is not None else 0
    r['max_gap_s'] = gap_s if r['count_eval'] and gap_s is not None else float(t1 - t0)
    r['mean_rate_hz'] = round(r['count_eval'] / (t1 - t0), 2) if t1 > t0 else 0
    r['nonfinite'] = int((~a['finite']).sum())
    qn = np.linalg.norm(a['q'], axis=1) if len(a['q']) else np.zeros(0)
    bad_q = ~np.isfinite(qn) | (np.abs(qn - 1) > QUAT_TOL)
    r['bad_quat'] = int(bad_q.sum())
    r['max_quat_norm_err'] = float(np.nanmax(np.abs(qn - 1))) if len(qn) else None
    r['bad_frames'] = int((~a['frames_ok']).sum())
    d = np.diff(a['stamp'])
    r['stamp_backwards'] = int((d < 0).sum())
    r['stamp_repeats'] = int((d == 0).sum())
    off = np.abs(a['stamp'] - a['recv'])
    r['stamp_vs_clock_max_s'] = float(off.max()) if len(off) else None
    r['stamp_vs_clock_tol_s'] = clock_tol
    r['stamp_vs_clock_bad'] = int((off > clock_tol).sum())
    return r


def consistency(od, tf):
    """Одна поза при одном stamp: для каждого /odom ищем TF с тем же stamp (±1 мкс)."""
    r = {'odom_msgs': int(len(od['stamp'])), 'matched_same_stamp': 0, 'max_pos_diff_m': None, 'max_ang_diff_deg': None}
    if not len(od['stamp']) or not len(tf['stamp']):
        return r
    order = np.argsort(tf['stamp'], kind='stable')
    ts, P, Q = tf['stamp'][order], tf['p'][order], tf['q'][order]
    idx = np.clip(np.searchsorted(ts, od['stamp']), 0, len(ts) - 1)
    idx2 = np.clip(idx - 1, 0, len(ts) - 1)
    best = np.where(np.abs(ts[idx2] - od['stamp']) < np.abs(ts[idx] - od['stamp']), idx2, idx)
    same = np.abs(ts[best] - od['stamp']) <= 1e-6
    r['matched_same_stamp'] = int(same.sum())
    if same.any():
        dp = np.linalg.norm(P[best[same]] - od['p'][same], axis=1)
        Ro, Rt = quat_to_mat(od['q'][same]), quat_to_mat(Q[best[same]])
        c = (np.einsum('nij,nij->n', Ro, Rt) - 1) / 2
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        r['max_pos_diff_m'] = float(np.nanmax(dp))
        r['max_ang_diff_deg'] = float(np.nanmax(ang))
    return r


def rpe(gt, od, t0, t1):
    """RPE по участкам ≈1 м фактического пути (по истине) на [t0, t1].
    A — разности положений в мировой системе; начальная поза оценки совмещается с истиной только переносом
        (строгое толкование: поворот совмещения не делаем → оценка должна начинаться с курсом мира).
    A_se3 — то же с полным совмещением начальной позы (перенос+поворот), справочно.
    B — перемещение в системе начала участка своей траектории (оценка — в своей, истина — в своей).
    Ошибка участка = |Δp_оценки − Δp_истины| / длина пути участка; итог — среднее по участкам."""
    sel = (gt['stamp'] >= t0) & (gt['stamp'] <= t1)
    tg, Pg, Qg = gt['stamp'][sel], gt['p'][sel], gt['q'][sel]
    out = {'segments': [], 'A_mean': None, 'B_mean': None, 'A_se3_mean': None, 'worst': None, 'missing': 0}
    if len(tg) < 2:
        out['error'] = 'нет истины в интервале'
        return out
    order = np.argsort(od['stamp'], kind='stable')
    ts, Pe, Qe = od['stamp'][order], od['p'][order], od['q'][order]
    good = np.all(np.isfinite(Pe), 1) & np.all(np.isfinite(Qe), 1) & (np.linalg.norm(Qe, axis=1) > 0.5)
    ts, Pe, Qe = ts[good], Pe[good], Qe[good]
    n = max(1, int(round(PATH_SMOOTH_S / max(np.median(np.diff(tg)), 1e-3))))
    k_ = np.ones(n) / n
    pad = np.pad(Pg[:, :2], ((n // 2, n - 1 - n // 2), (0, 0)), mode='edge')
    Ps = np.stack([np.convolve(pad[:, i], k_, 'valid') for i in range(2)], 1)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Ps, axis=0), axis=1))])
    out['path_len_raw3d_m'] = float(np.linalg.norm(np.diff(Pg, axis=0), axis=1).sum())
    out['path_len_m'] = float(cum[-1])
    cuts = [0]
    k = 1
    while True:
        j = int(np.searchsorted(cum, k * SEG_LEN_M))
        if j >= len(cum):
            break
        cuts.append(j)
        k += 1
    if cum[-1] - cum[cuts[-1]] >= SEG_MIN_TAIL_M and cuts[-1] != len(cum) - 1:
        cuts.append(len(cum) - 1)
    e0 = interp_pose(ts, Pe, Qe, tg[0])
    R_align = quat_to_mat(Qg[0]) @ quat_to_mat(e0[1]).T if e0 is not None else np.eye(3)
    out['align_t'] = float(tg[0])
    out['align_yaw_offset_deg'] = (float(np.degrees(wrap(yaw_of(Qg[0]) - yaw_of(e0[1])))) if e0 is not None else None)
    for a, b in zip(cuts[:-1], cuts[1:]):
        L = float(cum[b] - cum[a])
        dg = Pg[b] - Pg[a]
        seg = {'t0': float(tg[a]), 't1': float(tg[b]), 'len_m': round(L, 4),
               'gt_disp_m': round(float(np.linalg.norm(dg)), 4)}
        ea, eb = interp_pose(ts, Pe, Qe, tg[a]), interp_pose(ts, Pe, Qe, tg[b])
        if ea is None or eb is None:
            # нет оценки на краю участка — считаем как отсутствие перемещения (≈100 %) и помечаем
            seg.update(A=float(np.linalg.norm(dg) / L), B=float(np.linalg.norm(dg) / L),
                       A_se3=float(np.linalg.norm(dg) / L), missing=True)
            out['missing'] += 1
        else:
            de = eb[0] - ea[0]
            seg['A'] = float(np.linalg.norm(de - dg) / L)
            seg['A_se3'] = float(np.linalg.norm(R_align @ de - dg) / L)
            dl_e = quat_to_mat(ea[1]).T @ de
            dl_g = quat_to_mat(Qg[a]).T @ dg
            seg['B'] = float(np.linalg.norm(dl_e - dl_g) / L)
        out['segments'].append(seg)
    if out['segments']:
        for key in ('A', 'B', 'A_se3'):
            out[key + '_mean'] = float(np.mean([s[key] for s in out['segments']]))
        out['worst'] = max(out['A_mean'], out['B_mean'])
    out['n_segments'] = len(out['segments'])
    return out


def drift(od, gt, t0, t1):
    r = {'t0': t0, 't1': t1}
    order = np.argsort(od['stamp'], kind='stable')
    ts, P, Q = od['stamp'][order], od['p'][order], od['q'][order]
    a, b = interp_pose(ts, P, Q, t0), interp_pose(ts, P, Q, t1)
    if a is None or b is None:
        r['error'] = 'нет оценки на краях неподвижного интервала'
        r['pos_m'] = r['ang_deg'] = r['yaw_deg'] = None
    else:
        r['pos_m'] = float(np.linalg.norm(b[0] - a[0]))
        r['ang_deg'] = math.degrees(rot_angle(quat_to_mat(a[1]), quat_to_mat(b[1])))
        r['yaw_deg'] = float(abs(np.degrees(wrap(yaw_of(b[1]) - yaw_of(a[1])))))
    ga, gb = interp_pose(gt['stamp'], gt['p'], gt['q'], t0), interp_pose(gt['stamp'], gt['p'], gt['q'], t1)
    if ga is not None and gb is not None:   # справочно: сдвинулся ли сам робот
        r['gt_pos_m'] = float(np.linalg.norm(gb[0] - ga[0]))
        r['gt_ang_deg'] = math.degrees(rot_angle(quat_to_mat(ga[1]), quat_to_mat(gb[1])))
    return r


def crit(name, value, op, thr, points=None):
    ok = value is not None and not (isinstance(value, float) and math.isnan(value)) and (
        value <= thr if op == '<=' else value >= thr if op == '>=' else value == thr)
    d = {'criterion': name, 'value': value, 'need': f'{op} {thr}', 'result': 'PASS' if ok else 'FAIL'}
    if points is not None:
        d['points'] = points if ok else 0
    return d


def evaluate(rec, ph, meta):
    od, tf, gt = rec['odom'], rec['tf'], rec['gt']
    t0, t1 = ph['eval0'], ph.get('eval1', ph['drift1'])
    res = {'meta': meta, 'phases': ph, 'thresholds_tz': TZ, 'thresholds_stage3': STAGE3}
    so, st = stream_checks('/odom', od, t0, t1), stream_checks('TF odom->trunk', tf, t0, t1, STAMP_VS_CLOCK_TF_S)
    res['odom'], res['tf'] = so, st
    res['tf_other_parents_for_trunk'] = rec.get('tf_other', {})
    res['consistency'] = consistency(od, tf)
    res['rpe'] = rpe(gt, od, ph['eval0'], min(ph['drift0'], t1))
    res['drift'] = drift(od, gt, ph['drift0'], ph['drift1'])
    if ph.get('rpe_na'):
        res['rpe'] = {'segments': [], 'A_mean': None, 'B_mean': None, 'A_se3_mean': None, 'worst': None,
                      'missing': 0, 'n_segments': 0, 'na': 'нет движения'}
    if ph.get('drift_na'):
        res['drift'].update(pos_m=None, ang_deg=None, yaw_deg=None, na='интервал дрейфа не помещается в запись')
    res['forbidden'] = rec.get('forbidden', {})
    c = res['consistency']
    worst_rate = min(so['min_rate_hz'] or 0, st['min_rate_hz'] or 0)
    worst_gap = max(so['max_gap_s'], st['max_gap_s'])
    nonfinite = so['nonfinite'] + st['nonfinite']
    badq = so['bad_quat'] + st['bad_quat']
    publish_checks = [
        crit('/odom: мин. сообщений в окне 1 с sim', so['min_rate_hz'], '>=', TZ['rate_hz']),
        crit('TF odom→trunk: мин. в окне 1 с sim', st['min_rate_hz'], '>=', TZ['rate_hz']),
        crit('макс. разрыв /odom и TF, с sim', round(worst_gap, 4), '<=', TZ['max_gap_s']),
        crit('frame_id=odom, child_frame_id=trunk (нарушений)', so['bad_frames'] + st['bad_frames'], '==', 0),
        crit('NaN/Inf (сообщений)', nonfinite, '==', 0),
        crit('|q|=1±1e-6 (нарушений)', badq, '==', 0),
        crit('stamp назад (случаев)', so['stamp_backwards'] + st['stamp_backwards'], '==', 0),
        crit(f'stamp дальше {STAMP_VS_CLOCK_S}/{STAMP_VS_CLOCK_TF_S} с (odom/TF) от /clock (сообщений)',
             so['stamp_vs_clock_bad'] + st['stamp_vs_clock_bad'], '==', 0),
        crit('/odom без TF с тем же stamp (сообщений)', c['odom_msgs'] - c['matched_same_stamp'] if c['odom_msgs'] else
             None, '==', 0),
        crit('расхождение /odom и TF, м', c['max_pos_diff_m'], '<=', CONSIST_POS_M),
        crit('расхождение /odom и TF, °', c['max_ang_diff_deg'], '<=', CONSIST_ANG_DEG),
    ]
    fb = res['forbidden']
    if fb:
        publish_checks.append(crit('чужие подписчики /odom/ground_truth', len(fb.get('gt_subscribers_foreign', [])),
                                   '==', 0))
        publish_checks.append(crit('чужие публикаторы /cmd_vel', len(fb.get('cmd_vel_publishers_foreign', [])), '==', 0))
    pub_ok = all(x['result'] == 'PASS' for x in publish_checks)
    r = res['rpe']
    d = res['drift']
    worst = r['worst']
    tz_list = [{'criterion': 'Корректная публикация /odom и TF ≥ 20 Гц (6.5)', 'result': 'PASS' if pub_ok else 'FAIL',
                'points': 3 if pub_ok else 0, 'checks': publish_checks},
               crit('RPE средний, худший из A/B (≤ 50 %)', worst, '<=', TZ['rpe_5'], 5),
               crit('RPE средний, худший из A/B (≤ 20 %)', worst, '<=', TZ['rpe_11'], 6)]
    dr_ok = (d['pos_m'] is not None and d['pos_m'] <= TZ['drift_pos_m'] and d['ang_deg'] <= TZ['drift_ang_deg'])
    tz_list.append({'criterion': 'Дрейф за 10 с: ≤ 0,10 м и ≤ 5° (полный угол)', 'result': 'PASS' if dr_ok else 'FAIL',
                    'value': {'pos_m': d['pos_m'], 'ang_deg': d['ang_deg'], 'yaw_deg': d['yaw_deg']},
                    'points': 6 if dr_ok else 0})
    if meta.get('aborted'):   # сценарий не доведён до конца — результат не засчитываем
        tz_list.append({'criterion': 'Сценарий выполнен до конца (стенд)', 'result': 'FAIL', 'points': 0,
                        'value': meta['aborted']})
    def na(lst):   # критерии, не применимые к записи (нет движения / дрейф не помещается) → N/A, в вердикт не идут
        for x in lst:
            c_ = x['criterion']
            if (ph.get('rpe_na') and 'RPE' in c_) or (ph.get('drift_na') and ('рейф' in c_)):
                x['result'], x['points'] = 'N/A', 0
    na(tz_list)
    res['criteria_tz'] = tz_list
    res['points_tz'] = sum(x.get('points', 0) for x in tz_list)
    res['verdict_tz'] = 'PASS' if all(x['result'] in ('PASS', 'N/A') for x in tz_list) else 'FAIL'
    s3 = [crit('частота /odom и TF, мин. в окне 1 с', worst_rate, '>=', STAGE3['rate_hz']),
          crit('NaN/Inf', nonfinite, '==', 0),
          crit('публикация по ТЗ (все проверки)', pub_ok, '==', True),
          crit('RPE худший из A/B', worst, '<=', STAGE3['rpe']),
          crit('дрейф положения, м', d['pos_m'], '<=', STAGE3['drift_pos_m']),
          crit('дрейф ориентации (полный угол), °', d['ang_deg'], '<=', STAGE3['drift_ang_deg'])]
    na(s3)
    res['criteria_stage3'] = s3
    res['verdict_stage3'] = 'PASS' if all(x['result'] in ('PASS', 'N/A') for x in s3) else 'FAIL'
    return res


def fmt(v, p=3):
    return 'нет' if v is None else (f'{v:.{p}f}' if isinstance(v, float) else str(v))


def summary_text(res):
    m, r, d, so, st, c = res['meta'], res['rpe'], res['drift'], res['odom'], res['tf'], res['consistency']
    L = [f"T4 стенд: режим {m.get('mode')} {m.get('source', '')}  sim {fmt(m.get('sim_s'), 1)} с, wall "
         f"{fmt(m.get('wall_s'), 1)} с, RTF {fmt(m.get('rtf'))}"]
    if m.get('aborted'):
        L.append(f"  ПРОГОН ПРЕРВАН: {m['aborted']}")
    if m.get('estop'):
        L.append(f"  АВАРИЙНАЯ ОСТАНОВКА: {m['estop']}")
    for s in (so, st):
        L.append(f"  {s['name']}: {s['count_eval']} сообщ. за {s['eval_window_s']} с, средн. {s['mean_rate_hz']} Гц, "
                 f"мин. в окне 1 с {s['min_rate_hz']}, разрыв {fmt(s['max_gap_s'])} с, NaN/Inf {s['nonfinite']}, "
                 f"|q| {s['bad_quat']}, frame {s['bad_frames']}, stamp назад {s['stamp_backwards']}")
    L.append(f"  согласованность: {c['matched_same_stamp']}/{c['odom_msgs']} с тем же stamp, "
             f"Δp {fmt(c['max_pos_diff_m'], 6)} м, Δθ {fmt(c['max_ang_diff_deg'], 4)}°")
    L.append(f"  RPE ({r.get('n_segments', 0)} участков, путь {fmt(r.get('path_len_m'), 2)} м, нет оценки: "
             f"{r['missing']}): A {fmt(r['A_mean'] and r['A_mean'] * 100, 1)} %, B {fmt(r['B_mean'] and r['B_mean'] * 100, 1)}"
             f" %, худший {fmt(r['worst'] and r['worst'] * 100, 1)} %  (A с поворотом совмещения "
             f"{fmt(r['A_se3_mean'] and r['A_se3_mean'] * 100, 1)} %, курс оценки на старте отличается на "
             f"{fmt(r.get('align_yaw_offset_deg'), 2)}°)")
    L.append(f"  дрейф [{d['t0']:.1f}; {d['t1']:.1f}] с: {fmt(d['pos_m'], 4)} м, {fmt(d['ang_deg'], 3)}° полный, "
             f"курс {fmt(d['yaw_deg'], 3)}°  (истина: {fmt(d.get('gt_pos_m'), 4)} м, {fmt(d.get('gt_ang_deg'), 3)}°)")
    fb = res.get('forbidden') or {}
    if fb:
        L.append(f"  подписчики истины: {fb.get('gt_subscribers')}; чужие: {fb.get('gt_subscribers_foreign')}")
        L.append(f"  публикаторы /cmd_vel: {fb.get('cmd_vel_publishers')}; чужие: {fb.get('cmd_vel_publishers_foreign')}")
        L.append(f"  публикаторы /odom: {fb.get('odom_publishers')}")
    L.append('  Критерии ТЗ:')
    for x in res['criteria_tz']:
        L.append(f"    {x['result']}  {x['criterion']}  → {x.get('points', 0)} б.  ({fmt(x.get('value'))})")
        for y in x.get('checks', []):
            if y['result'] != 'PASS':
                L.append(f"        FAIL {y['criterion']}: {fmt(y['value'])} (нужно {y['need']})")
    L.append(f"  ИТОГО по ТЗ: {res['points_tz']}/20 — {res['verdict_tz']}")
    L.append('  Пороги приёмки этапа 3: ' + res['verdict_stage3'] + ' — ' + '; '.join(
        f"{x['criterion']} {fmt(x['value'])} {x['need']} {x['result']}" for x in res['criteria_stage3']))
    return '\n'.join(L)


def plot(rec, res, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'matplotlib нет — PNG не построен'
    gt, od, ph, r = rec['gt'], rec['odom'], res['phases'], res['rpe']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [3, 2]})
    sel = (gt['stamp'] >= ph['eval0']) & (gt['stamp'] <= ph['drift1'])
    ax.plot(gt['p'][sel, 0], gt['p'][sel, 1], 'k-', lw=2, label='истина /odom/ground_truth')
    osel = (od['stamp'] >= ph['eval0']) & (od['stamp'] <= ph['drift1'])
    if osel.any() and sel.any():
        o = np.argsort(od['stamp'])
        e0 = interp_pose(od['stamp'][o], od['p'][o], od['q'][o], gt['stamp'][sel][0])
        if e0 is not None:
            g0 = gt['p'][sel][0]
            pa = od['p'][osel] - e0[0] + g0
            Ra = quat_to_mat(gt['q'][sel][0]) @ quat_to_mat(e0[1]).T
            pb = (Ra @ (od['p'][osel] - e0[0]).T).T + g0
            ax.plot(pa[:, 0], pa[:, 1], 'C0-', lw=1.2, label='оценка /odom (совмещение переносом, A)')
            ax.plot(pb[:, 0], pb[:, 1], 'C1--', lw=1, label='оценка (перенос+поворот)')
    for s in r.get('segments', []):
        g = interp_pose(gt['stamp'], gt['p'], gt['q'], s['t0'])
        if g is not None:
            ax.plot(g[0][0], g[0][1], 'r|', ms=10)
    ax.plot(0, 0, 'ko', ms=5)
    for name, cx, cy, sx, sy in BOXES:
        ax.add_patch(plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, color='0.8'))
    for name, cx, cy, rr in CYLS:
        ax.add_patch(plt.Circle((cx, cy), rr, color='0.8'))
    ax.add_patch(plt.Rectangle((WALLS['xmin'], WALLS['ymin']), WALLS['xmax'] - WALLS['xmin'],
                               WALLS['ymax'] - WALLS['ymin'], fill=False, color='0.5'))
    ax.set_aspect('equal')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_title(f"{res['meta'].get('source', '')}  RPE худший {fmt(r['worst'] and r['worst'] * 100, 1)} %", fontsize=9)
    segs = r.get('segments', [])
    if segs:
        i = np.arange(len(segs))
        ax2.bar(i - 0.2, [s['A'] * 100 for s in segs], 0.4, label='A')
        ax2.bar(i + 0.2, [s['B'] * 100 for s in segs], 0.4, label='B')
        ax2.axhline(TZ['rpe_11'] * 100, color='g', ls='--', lw=1, label='20 %')
        ax2.axhline(STAGE3['rpe'] * 100, color='orange', ls='--', lw=1, label='35 %')
        ax2.axhline(TZ['rpe_5'] * 100, color='r', ls='--', lw=1, label='50 %')
        ax2.set_xlabel('участок ≈1 м')
        ax2.set_ylabel('ошибка, %')
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return None


# ---------------------------------------------------------------- запись (ROS)
def make_node(args, drive_preset=None):
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Bool
    from tf2_msgs.msg import TFMessage

    be = QoSProfile(depth=500, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    rel = QoSProfile(depth=5000, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

    class Judge(Node):
        def __init__(self):
            super().__init__('t4_judge')
            self.now = None
            self.ready = False
            self.od, self.tf, self.gt, self.cmd = [], [], [], []
            self.tf_other = {}
            self.graph = {'gt_sub': set(), 'cmd_pub': set(), 'odom_pub': set(), 'odom_qos': set()}
            self.create_subscription(Clock, '/clock', self.on_clock, be)
            self.create_subscription(Odometry, '/odom', self.on_odom, rel)
            self.create_subscription(TFMessage, '/tf', self.on_tf, rel)
            self.create_subscription(Odometry, '/odom/ground_truth', self.on_gt, be)
            self.create_subscription(Bool, '/go1_ready', lambda m: setattr(self, 'ready', bool(m.data)),
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                durability=DurabilityPolicy.TRANSIENT_LOCAL))
            if drive_preset is None:
                self.create_subscription(Twist, '/cmd_vel', self.on_cmd, be)
                self.pub = None
            else:
                self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        def on_clock(self, m):
            self.now = m.clock.sec + m.clock.nanosec * 1e-9

        def on_odom(self, m):
            if self.now is None:
                return
            p, q, tw = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist
            vals = [p.x, p.y, p.z, q.x, q.y, q.z, q.w, tw.linear.x, tw.linear.y, tw.linear.z,
                    tw.angular.x, tw.angular.y, tw.angular.z] + list(m.pose.covariance) + list(m.twist.covariance)
            self.od.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, self.now,
                            m.header.frame_id == 'odom' and m.child_frame_id == 'trunk',
                            bool(np.all(np.isfinite(vals))), p.x, p.y, p.z, q.x, q.y, q.z, q.w))

        def on_tf(self, m):
            if self.now is None:
                return
            for t in m.transforms:
                child = t.child_frame_id.lstrip('/')
                parent = t.header.frame_id.lstrip('/')
                if child != 'trunk' and parent != 'odom':
                    continue
                if parent != 'odom' or child != 'trunk':
                    key = f'{t.header.frame_id}->{t.child_frame_id}'
                    self.tf_other[key] = self.tf_other.get(key, 0) + 1
                    continue
                a, r = t.transform.translation, t.transform.rotation
                vals = [a.x, a.y, a.z, r.x, r.y, r.z, r.w]
                self.tf.append((t.header.stamp.sec + t.header.stamp.nanosec * 1e-9, self.now,
                                t.header.frame_id == 'odom' and t.child_frame_id == 'trunk',
                                bool(np.all(np.isfinite(vals))), *vals))

        def on_gt(self, m):
            if self.now is None:
                return
            p, q = m.pose.pose.position, m.pose.pose.orientation
            self.gt.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, self.now, True, True,
                            p.x, p.y, p.z, q.x, q.y, q.z, q.w))

        def on_cmd(self, m):
            if self.now is not None:
                self.cmd.append((self.now, m.linear.x, m.angular.z))

        def sample_graph(self):
            fq = lambda e: (e.node_namespace.rstrip('/') + '/' + e.node_name).replace('//', '/')
            for e in self.get_subscriptions_info_by_topic('/odom/ground_truth'):
                self.graph['gt_sub'].add(fq(e))
            for e in self.get_publishers_info_by_topic('/cmd_vel'):
                self.graph['cmd_pub'].add(fq(e))
            for e in self.get_publishers_info_by_topic('/odom'):
                self.graph['odom_pub'].add(fq(e))
                self.graph['odom_qos'].add(f'{fq(e)}: {e.qos_profile.reliability.name}/{e.qos_profile.durability.name}')

        def send(self, vx, wz):
            m = Twist()
            m.linear.x, m.angular.z = float(vx), float(wz)
            self.pub.publish(m)

    rclpy.init()
    return rclpy, Judge()


def to_arrays(rows):
    a = np.array(rows, dtype=float) if rows else np.zeros((0, 11))
    return {'stamp': a[:, 0], 'recv': a[:, 1], 'frames_ok': a[:, 2].astype(bool), 'finite': a[:, 3].astype(bool),
            'p': a[:, 4:7], 'q': a[:, 7:11]}


def collect(node):
    rec = {'odom': to_arrays(node.od), 'tf': to_arrays(node.tf), 'gt': to_arrays(node.gt), 'tf_other': node.tf_other}
    rec['cmd'] = np.array(node.cmd, float) if node.cmd else np.zeros((0, 3))
    g = node.graph
    foreign = lambda names: sorted(n for n in names if n.split('/')[-1] not in STAND_NODES
                                   and not n.split('/')[-1].startswith('_ros2cli'))
    rec['forbidden'] = {'gt_subscribers': sorted(g['gt_sub']), 'gt_subscribers_foreign': foreign(g['gt_sub']),
                        'cmd_vel_publishers': sorted(g['cmd_pub']), 'cmd_vel_publishers_foreign': foreign(g['cmd_pub']),
                        'odom_publishers': sorted(g['odom_pub']), 'odom_publisher_qos': sorted(g['odom_qos'])}
    return rec


def save_rec(rec, ph, meta, path):
    flat = {}
    for k in ('odom', 'tf', 'gt'):
        for kk, v in rec[k].items():
            flat[f'{k}__{kk}'] = v
    flat['cmd'] = rec['cmd']
    flat['extra'] = np.array(json.dumps({'phases': ph, 'meta': meta, 'tf_other': rec['tf_other'],
                                         'forbidden': rec['forbidden']}, ensure_ascii=False))
    np.savez_compressed(path, **flat)


def load_rec(path):
    z = np.load(path, allow_pickle=False)
    rec = {k: {} for k in ('odom', 'tf', 'gt')}
    for key in z.files:
        if '__' in key:
            a, b = key.split('__')
            rec[a][b] = z[key]
    rec['cmd'] = z['cmd']
    extra = json.loads(str(z['extra']))
    rec['tf_other'], rec['forbidden'] = extra['tf_other'], extra['forbidden']
    return rec, extra['phases'], extra['meta']


def spin_for(rclpy, node, cond, wall_timeout, graph_every=3.0):
    t_end, t_graph = time.monotonic() + wall_timeout, 0.0
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.01)
        if time.monotonic() - t_graph > graph_every:
            node.sample_graph()
            t_graph = time.monotonic()
        if cond():
            return True
    return False


# ---------------------------------------------------------------- живой режим: сценарий
def build_steps(p):
    sgn = 1.0 if p['turn'] == 'left' else -1.0
    steps = [('still0', 'still', p['still0'], 0.0)]
    for k in range(4):
        steps.append((f'straight{k + 1}', 'straight', p['side'] / V_LIN, sgn * k * math.pi / 2))
        steps.append((f'turn{k + 1}', 'turn', (math.pi / 2) / W_ANG, sgn * (k + 1) * math.pi / 2))
    steps += [('transition', 'still', p['transition'], 0.0), ('drift', 'still', p['drift'], 0.0),
              ('tail', 'still', 1.0, 0.0)]
    return steps, sgn


def run_live(args, p):
    rclpy, node = make_node(args, drive_preset=p)
    w0 = time.monotonic()
    print(f"[live] пресет {args.preset}: {p}", flush=True)
    if not spin_for(rclpy, node, lambda: node.now is not None and node.ready and node.gt, args.ready_timeout):
        print(f"[live] нет /clock, /go1_ready=true или истины за {args.ready_timeout} с wall "
              f"(clock={node.now}, ready={node.ready}, gt={len(node.gt)})", flush=True)
        node.sample_graph()
        meta = {'mode': 'live', 'source': args.preset, 'error': 'не дождались готовности сима'}
        return node, rclpy, None, meta
    # решение стартует вместе с симом; даём ему подняться (TODO(уточнить): ждёт ли Judge) — потом всё равно едем
    if not spin_for(rclpy, node, lambda: len(node.od) > 0, args.wait_odom):
        print(f"[live] /odom не появился за {args.wait_odom} с wall — начинаю сценарий без него", flush=True)
    steps, sgn = build_steps(p)
    yaw0 = float(yaw_of(np.array(node.gt[-1][7:11])))
    print(f"[live] готов: sim {node.now:.2f} с, старт истины ({node.gt[-1][4]:.3f}; {node.gt[-1][5]:.3f}), "
          f"курс {math.degrees(yaw0):.1f}°", flush=True)
    ws, s_start = time.monotonic(), node.now
    marks, estop = {}, None
    i, t_step, last_ctl, t_graph = 0, node.now, -1.0, 0.0
    aborted, clk_last, clk_wall = None, node.now, time.monotonic()
    marks[steps[0][0]] = t_step
    print(f"  sim {t_step:8.2f}: {steps[0][0]}", flush=True)
    while i < len(steps):
        rclpy.spin_once(node, timeout_sec=0.005)
        if time.monotonic() - t_graph > 3.0:
            node.sample_graph()
            t_graph = time.monotonic()
        if time.monotonic() - ws > args.wall_limit:
            aborted = f'превышен лимит wall {args.wall_limit} с'
            break
        now = node.now
        if now != clk_last:
            clk_last, clk_wall = now, time.monotonic()
        elif time.monotonic() - clk_wall > args.stall:
            aborted = f'сим остановился: /clock не идёт {args.stall:.0f} с wall (sim {now:.2f} с, шаг {steps[i][0]})'
            print(f"  ПРЕРВАНО: {aborted}", flush=True)
            break
        if now - last_ctl < 0.05:
            continue
        last_ctl = now
        name, kind, dur, yaw_t = steps[i]
        el = now - t_step
        g = node.gt[-1]
        yaw = float(yaw_of(np.array(g[7:11])))
        vx = wz = 0.0
        done = el >= dur
        if kind == 'straight':
            vx = V_LIN
            if p['correct']:
                wz = float(np.clip(1.0 * wrap(yaw0 + yaw_t - yaw), -W_ANG, W_ANG))
        elif kind == 'turn':
            wz = sgn * W_ANG
            if p['correct']:
                done = abs(wrap(yaw0 + yaw_t - yaw)) < math.radians(2.0) or el >= 2 * dur
        if kind in ('straight', 'turn') and estop is None:
            cl, what = clearance(g[4], g[5])
            if cl < ESTOP_CLEAR_M:
                estop = f'sim {now:.2f} с: корпус ({g[4]:.2f}; {g[5]:.2f}) в {cl:.2f} м от «{what}»'
                print(f"  АВАРИЙНАЯ ОСТАНОВКА: {estop}", flush=True)
                i = next(k for k, s in enumerate(steps) if s[0] == 'transition')
                t_step = now
                marks['transition'] = now
                node.send(0, 0)
                continue
        if done:
            i += 1
            t_step = now
            if i < len(steps):
                marks[steps[i][0]] = now
                print(f"  sim {now:8.2f} (wall {time.monotonic() - ws:6.1f}): {steps[i][0]}  истина "
                      f"({g[4]:.2f}; {g[5]:.2f}) курс {math.degrees(wrap(yaw - yaw0)):.1f}°", flush=True)
            node.send(0, 0) if i >= len(steps) or steps[i][1] == 'still' else None
            continue
        node.send(vx, wz)
    for _ in range(5):
        node.send(0, 0)
        rclpy.spin_once(node, timeout_sec=0.02)
    node.sample_graph()
    ph = {'eval0': marks['still0'], 'motion0': marks.get('straight1'), 'motion1': marks.get('transition'),
          'drift0': marks.get('drift'), 'drift1': marks.get('tail', node.now)}
    if ph['drift0'] is None:
        ph['drift0'] = node.now
    meta = {'mode': 'live', 'source': f'preset {args.preset}', 'preset': p, 'marks': marks, 'estop': estop,
            'aborted': aborted,
            'sim_s': node.now - s_start, 'wall_s': time.monotonic() - ws,
            'rtf': (node.now - s_start) / max(time.monotonic() - ws, 1e-6), 'wall_total_s': time.monotonic() - w0}
    return node, rclpy, ph, meta


# ---------------------------------------------------------------- офлайн по bag
BAG_TOPICS = ['/clock', '/imu/data', '/scan', '/joint_states', '/foot_force', '/odom/ground_truth', '/cmd_vel',
              '/cmd_vel_gated', '/tf', '/tf_static', '/go1_ready', '/robot_description']


def dds_env():
    env = dict(os.environ)
    env.setdefault('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    return env


def run_bag(args):
    bag = Path(args.bag).expanduser()
    here = Path(__file__).resolve().parent
    topics = [t for t in BAG_TOPICS if not (args.no_bag_tf and t == '/tf')]
    procs, container = [], None
    rclpy, node = make_node(args)
    env = dds_env()
    try:
        if args.stub:
            procs.append(subprocess.Popen([sys.executable, str(here / 't4_stub_odom.py'), '--mode', args.stub],
                                          env=env, start_new_session=True))
        if args.image:
            container = f'go1-t4-bag-{os.getpid()}'
            xml = os.environ.get('CYCLONEDDS_URI', '')
            if xml.startswith('file://'):
                xml = Path(xml[7:]).read_text()
            cmd = ['docker', 'run', '-d', '--rm', '--name', container, '--platform', 'linux/amd64', '--pull', 'never',
                   '--network', 'host', '--env', f"ROS_DOMAIN_ID={env.get('ROS_DOMAIN_ID', '0')}",
                   '--env', 'RMW_IMPLEMENTATION=rmw_cyclonedds_cpp', '--env', f'CYCLONEDDS_URI={xml}',
                   '--env', 'ROS_LOCALHOST_ONLY=0', '--env', 'GO1_DDS_LOCAL=0', '--env', 'GUI=false']
            if args.cpus:
                cmd += ['--cpus', str(args.cpus)]
            cmd += ['--entrypoint', '/entrypoint.sh', args.image, 'T4']
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        spin_for(rclpy, node, lambda: False, args.startup_wait)
        if args.image:   # ждём, пока решение поднимет издателя /odom (до --odom-pub-wait с wall)
            ok = spin_for(rclpy, node, lambda: bool(node.get_publishers_info_by_topic('/odom')), args.odom_pub_wait, 1.0)
            print(f"[bag] издатель /odom {'есть' if ok else 'НЕ появился за %.0f с' % args.odom_pub_wait}", flush=True)
        print(f'[bag] проигрываю {bag.name} (rate {args.rate}, /clock из bag)', flush=True)
        w0 = time.monotonic()
        play = subprocess.Popen(['ros2', 'bag', 'play', str(bag), '--rate', str(args.rate),
                                 '--read-ahead-queue-size', '5000', '--disable-keyboard-controls', '--topics', *topics],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        procs.append(play)
        t_graph = 0.0
        while play.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.monotonic() - t_graph > 3.0:
                node.sample_graph()
                t_graph = time.monotonic()
        spin_for(rclpy, node, lambda: False, 2.0)
        wall = time.monotonic() - w0
        if container:
            st = subprocess.run(['docker', 'inspect', '-f', '{{.State.Status}}', container], capture_output=True,
                                text=True).stdout.strip()
            print(f'[bag] контейнер решения: {st or "уже завершился"}', flush=True)
    finally:
        for pr in procs:
            if pr.poll() is None:
                os.killpg(pr.pid, signal.SIGINT)
                try:
                    pr.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(pr.pid, signal.SIGKILL)
        if container:
            subprocess.run(['docker', 'stop', '--time', '5', container], capture_output=True)
            subprocess.run(['docker', 'rm', '-f', container], capture_output=True)
    cmd = np.array(node.cmd, float) if node.cmd else np.zeros((0, 3))
    moving = cmd[(np.abs(cmd[:, 1]) > 1e-3) | (np.abs(cmd[:, 2]) > 1e-3)] if len(cmd) else cmd
    t_end = float(node.now or 0)
    if not len(moving):
        # bag без движения (D — мягкий старт): только публикация; RPE и дрейф не применимы
        t_first = float(node.gt[0][0]) if node.gt else t_end
        print('[bag] в bag нет ненулевых /cmd_vel — только частота/NaN/stamp/frame, RPE и дрейф не применимы', flush=True)
        ph = {'eval0': t_first + args.pre, 'motion0': None, 'motion1': None, 'drift0': t_end, 'drift1': t_end,
              'eval1': t_end, 'rpe_na': True, 'drift_na': True}
    else:
        m0, m1 = float(moving[0, 0]), float(moving[-1, 0])
        ph = {'eval0': m0 - args.pre, 'motion0': m0, 'motion1': m1, 'drift0': m1 + args.transition,
              'drift1': m1 + args.transition + 10.0}
        if ph['drift1'] > t_end:
            print(f"[bag] конец интервала дрейфа {ph['drift1']:.1f} позже конца bag {t_end:.1f} с sim — дрейф не применим",
                  flush=True)
            ph['drift_na'] = True
            ph['eval1'] = t_end
    sim_s = (node.now or 0) - (node.gt[0][1] if node.gt else 0)
    meta = {'mode': 'bag', 'source': f"{bag.name} + {('заглушка ' + args.stub) if args.stub else args.image}",
            'rate': args.rate, 'sim_s': sim_s, 'wall_s': wall, 'rtf': sim_s / max(wall, 1e-6)}
    return node, rclpy, ph, meta


# ---------------------------------------------------------------- main
def finish(rec, ph, meta, out):
    out.mkdir(parents=True, exist_ok=True)
    save_rec(rec, ph, meta, out / 'record.npz')
    res = evaluate(rec, ph, meta)
    err = plot(rec, res, out / 'trajectory.png')
    if err:
        res['plot'] = err
    txt = summary_text(res)
    (out / 'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (out / 'summary.txt').write_text(txt + '\n')
    print(txt, flush=True)
    print(f'файлы: {out}/result.json, summary.txt, trajectory.png, record.npz', flush=True)
    return 0 if res['verdict_tz'] == 'PASS' else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    lv = sub.add_parser('live')
    lv.add_argument('--preset', default='tz', choices=sorted(PRESETS))
    lv.add_argument('--side', type=float)
    lv.add_argument('--turn', choices=['left', 'right'])
    lv.add_argument('--correct', dest='correct', action='store_true', default=None)
    lv.add_argument('--no-correct', dest='correct', action='store_false')
    lv.add_argument('--still0', type=float)
    lv.add_argument('--transition', type=float)
    lv.add_argument('--ready-timeout', type=float, default=600.0, help='wall, с')
    lv.add_argument('--wait-odom', type=float, default=60.0, help='ждать первый /odom перед сценарием, wall с')
    lv.add_argument('--stall', type=float, default=60.0, help='/clock не идёт столько wall с → прервать')
    lv.add_argument('--wall-limit', type=float, default=2400.0, help='лимит сценария, wall с')
    bg = sub.add_parser('bag')
    bg.add_argument('bag')
    g = bg.add_mutually_exclusive_group(required=True)
    g.add_argument('--stub', choices=['zero', 'truth', 'rot10', 'drift'])
    g.add_argument('--image')
    bg.add_argument('--cpus', type=float)
    bg.add_argument('--rate', type=float, default=2.0, help='скорость проигрывания относительно записи (wall)')
    bg.add_argument('--pre', type=float, default=3.0, help='начало оценки: за столько с sim до первой команды')
    bg.add_argument('--transition', type=float, default=5.0, help='от последней команды до начала 10 с дрейфа, с sim')
    bg.add_argument('--startup-wait', type=float, default=5.0, help='wall, с: дать решению подняться')
    bg.add_argument('--odom-pub-wait', type=float, default=90.0, help='wall, с: ждать издателя /odom от образа')
    bg.add_argument('--no-bag-tf', action='store_true', help='не проигрывать /tf суставов из bag')
    ra = sub.add_parser('reanalyze')
    ra.add_argument('record')
    for s in (lv, bg, ra):
        s.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.mode == 'reanalyze':
        rec, ph, meta = load_rec(args.record)
        out = Path(args.out or Path(args.record).parent)
        return finish(rec, ph, meta, out)
    out = Path(args.out or f"/tmp/t4_judge_{time.strftime('%Y%m%d_%H%M%S')}")
    if args.mode == 'live':
        p = dict(PRESETS[args.preset])
        for k in ('side', 'turn', 'correct', 'still0', 'transition'):
            if getattr(args, k) is not None:
                p[k] = getattr(args, k)
        node, rclpy, ph, meta = run_live(args, p)
    else:
        node, rclpy, ph, meta = run_bag(args)
    rec = collect(node)
    node.destroy_node()
    rclpy.shutdown()
    if ph is None:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'result.json').write_text(json.dumps({'meta': meta, 'verdict_tz': 'FAIL', 'forbidden': rec['forbidden'],
                                                     'odom_msgs': len(rec['odom']['stamp'])}, ensure_ascii=False))
        print(json.dumps(meta, ensure_ascii=False))
        return 2
    return finish(rec, ph, meta, out)


if __name__ == '__main__':
    sys.exit(main())
