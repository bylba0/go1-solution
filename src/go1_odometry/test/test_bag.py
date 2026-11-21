"""Тесты T4 на записанных bag живого сима (docs/T4_RECON.md). Каталог bag — переменная окружения GO1_T4_BAGS
(в нём t4_A_square_left/, t4_D_startup/); без неё или без rosbag2_py тесты пропускаются.
Истина (/odom/ground_truth) здесь НЕ читается: проверяются FK против TF robot_state_publisher, стоянка и мягкий старт."""
import math
import os
import pathlib

import numpy as np
import pytest

from go1_odometry.core import JOINT_NAMES, OdometryEstimator, foot_positions, quat_to_rot, quat_yaw

BAGS = pathlib.Path(os.environ.get('GO1_T4_BAGS', '/nonexistent'))
LEGS = ('FR', 'FL', 'RR', 'RL')


def _bag(name):
    p = BAGS / name
    if not p.is_dir():
        pytest.skip(f'нет bag {p} (задайте GO1_T4_BAGS)')
    pytest.importorskip('rosbag2_py')
    return p


def _read(path, topics, stop=None):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    cls = {t: get_message(types[t]) for t in topics if t in types}
    while r.has_next():
        t, d, _ = r.read_next()
        m = deserialize_message(d, cls[t])
        yield t, m
        if stop is not None and stop(t, m):
            return


def _st(s):
    return s.sec + s.nanosec * 1e-9


def _tf_matrix(tr):
    q = tr.transform.rotation
    T = np.eye(4)
    T[:3, :3] = quat_to_rot((q.x, q.y, q.z, q.w))
    v = tr.transform.translation
    T[:3, 3] = (v.x, v.y, v.z)
    return T


def _joint_vec(m):
    idx = {n: i for i, n in enumerate(m.name)}
    return np.array([m.position[idx[n]] for n in JOINT_NAMES])


def _inputs(path, t_from=-math.inf):
    imu, js = [], []
    for topic, m in _read(path, ['/imu/data', '/joint_states']):
        t = _st(m.header.stamp)
        if t < t_from:
            continue
        if topic == '/imu/data':
            o = m.orientation
            imu.append((t, (o.x, o.y, o.z, o.w), (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)))
        else:
            js.append((t, _joint_vec(m)))
    return imu, js


def _replay(imu, js, t0, t1, rate=200.0):
    """Как узел: на тике таймера — последнее IMU и последние суставы со stamp ≤ t."""
    est = OdometryEstimator()
    ii = ij = 0
    poses = []
    for k in range(int((t1 - t0) * rate) + 1):
        t = t0 + k / rate
        while ii < len(imu) and imu[ii][0] <= t:
            est.set_imu(*imu[ii])
            ii += 1
        while ij < len(js) and js[ij][0] <= t:
            ij += 1
        if ij:
            est.set_joints(*js[ij - 1])
        if est.step(t):
            poses.append((t, est.position.copy(), est.orientation.copy()))
    return poses


def test_fk_matches_robot_state_publisher_tf():
    path = _bag('t4_A_square_left')
    static, joints, tfs = {}, {}, {}
    t_start = [None]

    def stop(topic, m):
        return t_start[0] is not None and len(tfs) > 0 and max(tfs) - t_start[0] > 15.0   # 10 с стоянки + ходьба

    for topic, m in _read(path, ['/tf_static', '/tf', '/joint_states'], stop):
        if topic == '/tf_static':
            for tr in m.transforms:
                static[tr.child_frame_id] = _tf_matrix(tr)
        elif topic == '/tf':
            t = _st(m.transforms[0].header.stamp)
            t_start[0] = t if t_start[0] is None else t_start[0]
            tfs[t] = {tr.child_frame_id: _tf_matrix(tr) for tr in m.transforms}
        else:
            joints[_st(m.header.stamp)] = _joint_vec(m)
    common = sorted(set(tfs) & set(joints))
    assert len(common) > 1000
    worst = 0.0
    spread = []
    for t in common[::5]:
        tf = tfs[t]
        if not all(f'{leg}_{k}' in tf for leg in LEGS for k in ('hip', 'thigh', 'calf')):
            continue
        p = foot_positions(joints[t])
        for i, leg in enumerate(LEGS):
            T = tf[f'{leg}_hip'] @ tf[f'{leg}_thigh'] @ tf[f'{leg}_calf'] @ static[f'{leg}_foot']
            worst = max(worst, float(np.linalg.norm(T[:3, 3] - p[i])))
        spread.append(p[0, 0])
    assert worst < 1e-5, worst
    assert np.ptp(spread) > 0.03          # в окне есть ходьба, а не только стойка


def test_standstill_60s_no_drift():
    path = _bag('t4_A_square_left')
    imu, js = _inputs(path, t_from=0.0)
    t1 = js[-1][0]
    t0 = t1 - 60.0                         # A: последние 75 с — стоянка
    imu = [r for r in imu if r[0] >= t0 - 0.1]
    js = [r for r in js if r[0] >= t0 - 0.1]
    poses = _replay(imu, js, t0, t1)
    p0, q0 = poses[0][1], poses[0][2]
    p1, q1 = poses[-1][1], poses[-1][2]
    drift = float(np.linalg.norm(p1[:2] - p0[:2]))
    dyaw = abs(math.degrees(quat_yaw(q1) - quat_yaw(q0)))
    worst = max(float(np.linalg.norm(p[1][:2] - p0[:2])) for p in poses)
    assert drift < 0.01, drift
    assert worst < 0.02, worst
    assert dyaw < 0.5, dyaw


def test_soft_start_no_xy_jump():
    path = _bag('t4_D_startup')
    imu, js = _inputs(path)
    t0 = min(imu[0][0], js[0][0])
    poses = _replay(imu, js, t0, js[-1][0])
    xy = np.array([p[1][:2] for p in poses])
    z = np.array([p[1][2] for p in poses])
    assert np.all(np.isfinite(xy)) and np.all(np.isfinite(z))
    assert float(np.abs(xy).max()) < 0.03          # подъём/проседание корпуса не дают смещения x/y
    assert 0.25 < z.min() and z.max() < 0.33
