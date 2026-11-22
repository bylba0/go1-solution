#!/usr/bin/env python3
"""Самопроверка одометрии T4 на записанных bag (только для разработки, в образ не входит).

Истина /odom/ground_truth читается ПРЯМО из bag-файла и используется только здесь — для метрик; узел её не видит.

Режимы:
  replay  — прогон go1_odometry.core по входам bag (таймер 200 Гц sim, выход 50 Гц), без ROS-графа (быстро):
            python3 tools/dev_t4_offline_eval.py replay ~/ros2_ws/data/t4_bags/t4_A_square_left \\
                [--cache DIR] [--set k=v ...] [--estimator legs|lidar]
            lidar — go1_odometry.lidar_mode (IMU + /scan + суставы) с той же подачей: на тике 200 Гц — все новые IMU
            и сканы, последнее /joint_states; поза — на тике публикации.
  odom    — оценка /odom, записанного с живого узла (tools/dev_t4_play.sh: bag play входов + узел + record):
            python3 tools/dev_t4_offline_eval.py odom <исходный_bag> <bag_с_/odom>
Метрики: RPE по участкам пути GT ≈ 1 м (|Δp_est − Δp_gt| / длина участка, xy), дрейф на последней стоянке
(окно 10 с и вся стоянка), частота /odom по sim-времени, число NaN.
"""
import argparse
import math
import os
import pathlib
import sys

import numpy as np

ORDER = [f'{leg}_{j}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for j in ('hip', 'thigh', 'calf')]


def _st(s):
    return s.sec + s.nanosec * 1e-9


def read_bag(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    topics = [t for t in topics if t in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    cls = {t: get_message(types[t]) for t in topics}
    while r.has_next():
        t, d, ts = r.read_next()
        yield t, deserialize_message(d, cls[t]), ts


def load_inputs(bag, cache=None, scans=False):
    if cache:
        f = pathlib.Path(cache) / (pathlib.Path(bag).name + ('.scan' if scans else '') + '.npz')
        if f.exists():
            return dict(np.load(f))
    js_t, js_q, imu, gt, scan_t, scan_r, scan_a = [], [], [], [], [], [], []
    topics = ['/joint_states', '/imu/data', '/odom/ground_truth'] + (['/scan'] if scans else [])
    for t, m, _ in read_bag(bag, topics):
        if t == '/scan':
            scan_t.append(_st(m.header.stamp))
            scan_r.append(np.asarray(m.ranges, dtype=np.float32))
            scan_a.append([m.angle_min, m.angle_increment])
        elif t == '/joint_states':
            idx = {n: i for i, n in enumerate(m.name)}
            js_t.append(_st(m.header.stamp))
            js_q.append([m.position[idx[n]] for n in ORDER])
        elif t == '/imu/data':
            o, w = m.orientation, m.angular_velocity
            imu.append([_st(m.header.stamp), o.x, o.y, o.z, o.w, w.x, w.y, w.z])
        else:
            p, o = m.pose.pose.position, m.pose.pose.orientation
            gt.append([_st(m.header.stamp), p.x, p.y, p.z, o.x, o.y, o.z, o.w])
    d = dict(js_t=np.array(js_t), js_q=np.array(js_q), imu=np.array(imu), gt=np.array(gt))
    if scans:
        d.update(scan_t=np.array(scan_t), scan_r=np.array(scan_r), scan_a=np.array(scan_a))
    if cache:
        pathlib.Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez(f, **d)
    return d


def replay(d, params, timer_hz=200.0, pub_hz=50.0):
    from go1_odometry.core import OdometryEstimator
    est = OdometryEstimator(**params)
    js_t, js_q, imu = d['js_t'], d['js_q'], d['imu']
    t0 = min(js_t[0], imu[0, 0])
    t_end = max(js_t[-1], imu[-1, 0])
    ij = ii = 0
    out = []
    n = int((t_end - t0) * timer_hz)
    every = int(round(timer_hz / pub_hz))
    for k in range(1, n + 1):
        t = t0 + k / timer_hz
        while ii < len(imu) and imu[ii, 0] <= t + 1e-9:
            ii += 1
        while ij < len(js_t) and js_t[ij] <= t + 1e-9:
            ij += 1
        if ii:
            r = imu[ii - 1]
            est.set_imu(r[0], r[1:5], r[5:8])
        if ij:
            est.set_joints(js_t[ij - 1], js_q[ij - 1])
        if est.step(t) and k % every == 0:
            out.append([t, *est.position, *est.orientation])
    return np.array(out)


def replay_lidar(d, timer_hz=200.0, pub_hz=50.0, leg_scale=1.0):
    from go1_odometry.lidar_mode import LIDAR_JOINT_NAMES, LidarEstimator
    est = LidarEstimator(leg_scale=leg_scale)
    perm = [ORDER.index(n) for n in LIDAR_JOINT_NAMES]
    js_t, js_q, imu, sc_t = d['js_t'], d['js_q'][:, perm], d['imu'], d['scan_t']
    t0 = min(js_t[0], imu[0, 0])
    t_end = max(js_t[-1], imu[-1, 0])
    ij = ii = isc = 0
    out = []
    n = int((t_end - t0) * timer_hz)
    every = int(round(timer_hz / pub_hz))
    for k in range(1, n + 1):
        t = t0 + k / timer_hz
        ev = []
        while ii < len(imu) and imu[ii, 0] <= t + 1e-9:
            ev.append(('imu', imu[ii, 0], imu[ii, 1:5]))
            ii += 1
        j0 = ij
        while ij < len(js_t) and js_t[ij] <= t + 1e-9:
            ij += 1
        if ij > j0:
            ev.append(('js', js_t[ij - 1], js_q[ij - 1]))
        while isc < len(sc_t) and sc_t[isc] <= t + 1e-9:
            ev.append(('scan', sc_t[isc], d['scan_r'][isc], *d['scan_a'][isc]))
            isc += 1
        est.feed(ev)
        if k % every == 0:
            p = est.pose(t)
            if p is not None:
                out.append([t, p[0][0], p[0][1], 0.0, *p[1]])
    ms = np.array(est.scan_ms)
    if len(ms):
        print(f'сканы: {est.n_scans_in}, сопоставлено {est.core.n_match}, пропущено {est.core.n_skip}, '
              f'опорных {est.core.n_key}, вырожд. {est.core.n_degen}; '
              f'обработка {ms.mean():.1f} мс ср, {ms.max():.1f} макс',
              file=sys.stderr)
    return np.array(out)


def _yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def evaluate(est, gt, seg_len=1.0, still_win=10.0):
    """est, gt: [t, x, y, z, qx, qy, qz, qw]. Оценка приводится к GT в начальный момент (как у Judge)."""
    est = est[np.all(np.isfinite(est), axis=1)]
    nan_count = 0
    t0 = max(est[0, 0], gt[0, 0])
    g0 = gt[np.searchsorted(gt[:, 0], t0)]
    e0 = est[np.searchsorted(est[:, 0], t0)]
    dyaw = _yaw(g0[4:8]) - _yaw(e0[4:8])
    c, s = math.cos(dyaw), math.sin(dyaw)

    def est_at(t):
        i = min(np.searchsorted(est[:, 0], t), len(est) - 1)
        e = est[i]
        dx, dy = e[1] - e0[1], e[2] - e0[2]
        return np.array([g0[1] + c * dx - s * dy, g0[2] + s * dx + c * dy]), _yaw(e[4:8]) + dyaw

    g = gt[gt[:, 0] >= t0]
    step = np.linalg.norm(np.diff(g[:, 1:3], axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(step)])
    errs = []
    i0 = 0
    for i in range(1, len(g)):
        if cum[i] - cum[i0] >= seg_len:
            dg = g[i, 1:3] - g[i0, 1:3]
            de = est_at(g[i, 0])[0] - est_at(g[i0, 0])[0]
            errs.append(np.linalg.norm(de - dg) / (cum[i] - cum[i0]))
            i0 = i
    # последняя стоянка по GT: с конца, пока скорость GT < 2 см/с
    v = np.concatenate([[0], step / np.maximum(np.diff(g[:, 0]), 1e-6)])
    k = len(g) - 1
    while k > 0 and v[k] < 0.02:
        k -= 1
    still_t0 = g[min(k + 20, len(g) - 1), 0]   # +1 с на успокоение
    t_last = min(g[-1, 0], est[-1, 0])

    def drift(ta, tb):
        pa, ya = est_at(ta)
        pb, yb = est_at(tb)
        dy = math.degrees((yb - ya + math.pi) % (2 * math.pi) - math.pi)
        return float(np.linalg.norm(pb - pa)), abs(dy)
    dt = np.diff(est[:, 0])
    return {
        'rpe_mean_pct': 100 * float(np.mean(errs)) if errs else float('nan'),
        'rpe_max_pct': 100 * float(np.max(errs)) if errs else float('nan'),
        'segments': len(errs),
        'path_gt_m': float(cum[-1]),
        'final_err_m': float(np.linalg.norm(est_at(t_last)[0] - gt[np.searchsorted(gt[:, 0], t_last) - 1, 1:3])),
        'still_s': float(t_last - still_t0),
        'drift_last10s': drift(t_last - still_win, t_last),
        'drift_still_all': drift(still_t0, t_last),
        'odom_hz': float(len(dt) / (est[-1, 0] - est[0, 0])),
        'odom_dt_max_ms': 1000 * float(dt.max()),
        'nan': nan_count,
    }


def read_odom(bag):
    """/odom из записи + сверка TF odom→trunk: число сообщений и совпадение stamp/позы с /odom."""
    out, nan, tf = [], 0, {}
    for topic, m, _ in read_bag(bag, ['/odom', '/tf']):
        if topic == '/tf':
            for tr in m.transforms:
                if tr.header.frame_id == 'odom' and tr.child_frame_id == 'trunk':
                    v = tr.transform.translation
                    tf[_st(tr.header.stamp)] = (v.x, v.y, v.z)
            continue
        p, o = m.pose.pose.position, m.pose.pose.orientation
        row = [_st(m.header.stamp), p.x, p.y, p.z, o.x, o.y, o.z, o.w]
        if not all(math.isfinite(v) for v in row):
            nan += 1
        out.append(row)
    same = sum(1 for r in out if tf.get(r[0]) == tuple(r[1:4]))
    print(f'/odom {len(out)}, TF odom→trunk {len(tf)}, совпали stamp+поза {same}')
    return np.array(out), nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['replay', 'odom'])
    ap.add_argument('bag')
    ap.add_argument('odom_bag', nargs='?')
    ap.add_argument('--cache', help='каталог кэша входов (npz)')
    ap.add_argument('--set', action='append', default=[], help='параметр оценщика k=v (legs; lidar: leg_scale=…)')
    ap.add_argument('--estimator', choices=['legs', 'lidar'], default='legs', help='режим replay')
    a = ap.parse_args()
    here = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(here / 'src' / 'go1_odometry'))
    d = load_inputs(a.bag, a.cache, scans=a.mode == 'replay' and a.estimator == 'lidar')
    if a.mode == 'replay':
        params = {}
        for kv in a.set:
            k, v = kv.split('=', 1)
            try:
                v = float(v)
            except ValueError:
                pass
            params[k] = v
        est = replay_lidar(d, **params) if a.estimator == 'lidar' else replay(d, params)
        nan = 0
    else:
        est, nan = read_odom(a.odom_bag)
    res = evaluate(est, d['gt'])
    res['nan'] = nan
    print(os.path.basename(a.bag.rstrip('/')), ' '.join(
        f'{k}={v:.3f}' if isinstance(v, float) else
        f'{k}=({v[0]:.3f} м, {v[1]:.2f}°)' if isinstance(v, tuple) else f'{k}={v}' for k, v in res.items()))


if __name__ == '__main__':
    main()
