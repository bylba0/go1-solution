"""Unit-тесты оценщика T4 без ROS. Запуск: pytest src/go1_odometry/test -q"""
import math

import numpy as np
import pytest

from go1_odometry.core import (JOINT_NAMES, OdometryEstimator, foot_positions, quat_from_rotvec, quat_mul,
                               quat_to_rot, quat_yaw)

STAND_Q = np.array([0.0, 0.8, -1.6] * 4)


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _fk_reference(leg, q):
    """Цепочка однородных преобразований прямо по URDF (независимо от core.foot_positions)."""
    sx = 1 if leg[0] == 'F' else -1
    sy = -1 if leg[1] == 'R' else 1
    chain = [((sx * 0.1881, sy * 0.04675, 0), 'x', q[0]), ((0, sy * 0.08, 0), 'y', q[1]), ((0, 0, -0.213), 'y', q[2])]
    T = np.eye(4)
    for xyz, axis, a in chain:
        J = np.eye(4)
        J[:3, 3] = xyz
        J[:3, :3] = _rot(axis, a)
        T = T @ J
    return (T @ np.array([0, 0, -0.213, 1]))[:3]


def test_joint_names_order():
    assert JOINT_NAMES[:3] == ('FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint')
    assert len(set(JOINT_NAMES)) == 12


def test_fk_zero_matches_urdf():
    p = foot_positions(np.zeros(12))
    expect = np.array([[0.1881, -0.12675, -0.426], [0.1881, 0.12675, -0.426],
                       [-0.1881, -0.12675, -0.426], [-0.1881, 0.12675, -0.426]])
    assert np.allclose(p, expect, atol=1e-9)


def test_fk_random_matches_reference_chain():
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.uniform([-1.0, -0.6, -2.7] * 4, [1.0, 2.9, -0.9] * 4)
        p = foot_positions(q)
        for i, leg in enumerate(('FR', 'FL', 'RR', 'RL')):
            assert np.allclose(p[i], _fk_reference(leg, q[3 * i:3 * i + 3]), atol=1e-12)


def _imu_quat(yaw, roll=0.0, pitch=0.0):
    q = quat_from_rotvec(np.array([0, 0, yaw]))
    q = quat_mul(q, quat_from_rotvec(np.array([0, pitch, 0])))
    return quat_mul(q, quat_from_rotvec(np.array([roll, 0, 0])))


def test_orientation_relative_to_initial_yaw_and_normalized():
    est = OdometryEstimator()
    est.set_imu(0.0, 2.0 * _imu_quat(0.7), (0, 0, 0))      # ненормированный вход
    est.set_joints(0.0, STAND_Q)
    assert est.step(0.0)
    assert abs(np.linalg.norm(est.orientation) - 1) < 1e-12
    assert abs(quat_yaw(est.orientation)) < 1e-9
    est.set_imu(0.01, _imu_quat(0.7 + 0.3, roll=0.05), (0, 0, 0))
    est.step(0.01)
    assert abs(quat_yaw(est.orientation) - 0.3) < 1e-6
    assert abs(np.linalg.norm(est.orientation) - 1) < 1e-12


def test_no_pose_before_imu_and_initial_height():
    est = OdometryEstimator()
    est.set_joints(0.0, STAND_Q)
    assert not est.step(0.0)
    est.set_imu(0.0, (0, 0, 0, 1), (0, 0, 0))
    assert est.step(0.0)
    assert 0.2 < est.position[2] < 0.4
    assert est.position[0] == 0.0 and est.position[1] == 0.0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nan_inf_inputs_ignored(bad):
    est = OdometryEstimator()
    est.set_imu(0.0, (0, 0, 0, 1), (0, 0, 0))
    est.set_joints(0.0, STAND_Q)
    est.step(0.0)
    pos, ori = est.position.copy(), est.orientation.copy()
    assert not est.set_imu(0.005, (bad, 0, 0, 1), (0, 0, 0))
    assert not est.set_imu(0.005, (0, 0, 0, 0), (0, 0, 0))          # нулевой кватернион
    q = STAND_Q.copy()
    q[4] = bad
    assert not est.set_joints(0.005, q)
    assert not est.set_joints(bad, STAND_Q)
    assert est.step(0.005)
    assert np.all(np.isfinite(est.position)) and np.all(np.isfinite(est.orientation))
    assert np.allclose(est.position, pos) and np.allclose(est.orientation, ori)
    assert est.rejected == 4


def test_gyro_mode_integrates_angular_rate():
    est = OdometryEstimator(orientation_mode='gyro')
    est.set_imu(0.0, (0, 0, 0, 1), (0, 0, 0.5))
    est.step(0.0)                      # начальный курс фиксируется на первом шаге
    for k in range(1, 201):            # 1 с при 0,5 рад/с; кватернион IMU намеренно неподвижен
        est.set_imu(k * 0.005, (0, 0, 0, 1), (0, 0, 0.5))
    est.set_joints(1.0, STAND_Q)
    est.step(1.0)
    assert abs(quat_yaw(est.orientation) - 0.5) < 1e-6


def _ik(feet_trunk, q0=None):
    """Численная обратная кинематика (Ньютон, для синтетики): стопы в trunk → 12 углов."""
    q = (STAND_Q if q0 is None else q0).copy()
    for _ in range(30):
        err = (feet_trunk - foot_positions(q)).reshape(-1)
        if np.abs(err).max() < 1e-12:
            break
        for leg in range(4):
            sl = slice(3 * leg, 3 * leg + 3)
            J = np.zeros((3, 3))
            for k in range(3):
                dq = q.copy()
                dq[3 * leg + k] += 1e-7
                J[:, k] = (foot_positions(dq)[leg] - foot_positions(q)[leg]) / 1e-7
            q[sl] += np.linalg.solve(J, err[sl])
    return q


def test_ik_helper_roundtrip():
    for q in (STAND_Q, np.array([0.1, 0.7, -1.5, -0.1, 0.9, -1.7, 0.05, 0.6, -1.4, 0.0, 1.0, -1.8])):
        assert np.allclose(foot_positions(_ik(foot_positions(q))), foot_positions(q), atol=1e-9)


def test_synthetic_walk_with_fixed_stance_feet():
    """Рысь: корпус проходит ~1 м с поворотом курса на 30°; опорные стопы неподвижны в мире,
    переносные плавно переставляются вперёд с подъёмом 4 см. Оценка должна совпасть с путём корпуса."""
    est = OdometryEstimator()
    dt, v, period = 0.005, 0.2, 50                 # 0,25 с на фазу
    nominal = foot_positions(STAND_Q)
    nominal[:, 2] = -0.30
    body, yaw = np.zeros(3), 0.0
    feet_w = nominal.copy()
    start = feet_w.copy()
    target = feet_w.copy()
    swing = ()
    q = None
    for k in range(int(5.0 / dt)):
        t = k * dt
        ph = k % period
        if ph == 0:
            swing = (0, 3) if (k // period) % 2 == 0 else (1, 2)
            # цель переноса: под корпусом через полфазы вперёд (как у контроллера)
            yaw_n = yaw + 0.5 * period * dt * math.radians(30) / 5.0
            body_n = body + period * dt * v * np.array([math.cos(yaw), math.sin(yaw), 0])
            Rn = quat_to_rot(quat_from_rotvec(np.array([0, 0, yaw_n])))
            for i in swing:
                start[i] = feet_w[i]
                target[i] = body_n + Rn @ nominal[i]
                target[i, 2] = -0.30
        s_ = (ph + 1) / period
        for i in swing:
            feet_w[i] = start[i] + (target[i] - start[i]) * (3 * s_ ** 2 - 2 * s_ ** 3)
            feet_w[i, 2] = -0.30 + 0.04 * math.sin(math.pi * s_)
        R = quat_to_rot(quat_from_rotvec(np.array([0, 0, yaw])))
        feet_t = (feet_w - body) @ R
        q = _ik(feet_t, q)
        est.set_imu(t, quat_from_rotvec(np.array([0, 0, yaw])), (0, 0, 0))
        est.set_joints(t, q)
        est.step(t)
        body = body + dt * v * np.array([math.cos(yaw), math.sin(yaw), 0])
        yaw += dt * math.radians(30) / 5.0
    # курс в начале 0 ⇒ odom = мир; последний шаг корпуса после step не учтён
    body_last = body - dt * v * np.array([math.cos(yaw), math.sin(yaw), 0])
    assert np.linalg.norm(body_last[:2]) > 0.9
    assert np.linalg.norm(est.position[:2] - body_last[:2]) < 0.03 * np.linalg.norm(body_last[:2])
    assert abs(math.degrees(quat_yaw(est.orientation)) - 30) < 0.1


def test_standstill_vibration_no_drift():
    """Стоим 60 с: ориентация дрожит (±0,5°, 200 Гц), суставы слегка шумят — смещение ≈ 0."""
    rng = np.random.default_rng(3)
    est = OdometryEstimator()
    for k in range(12000):
        t = k * 0.005
        rv = np.radians(0.5) * rng.uniform(-1, 1, 3)
        est.set_imu(t, quat_from_rotvec(rv), rng.normal(0, 2.0, 3))     # гироскоп «вибрирует» — не используется
        est.set_joints(t, STAND_Q + rng.normal(0, 2e-5, 12))
        est.step(t)
    assert np.linalg.norm(est.position[:2]) < 0.005
    assert abs(math.degrees(quat_yaw(est.orientation))) < 1.0


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        OdometryEstimator(no_such_param=1)
    with pytest.raises(ValueError):
        OdometryEstimator(orientation_mode='magic')
