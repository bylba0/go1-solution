"""Режим lidar без ROS: раскладка суставов по именам, опора без контактов, мировой курс, NaN/Inf на входе."""
import math

import numpy as np

from go1_odometry import lidar_core
from go1_odometry.core import JOINT_NAMES, foot_positions, quat_yaw
from go1_odometry.lidar_mode import LIDAR_JOINT_NAMES, LidarEstimator, joint_index

STAND = {'hip': 0.0, 'thigh': 0.8, 'calf': -1.6}


def _yaw_quat(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def test_joint_order_by_names_and_kinematics_agree():
    """Ядро lidar ждёт FL, FR, RL, RR; наши JOINT_NAMES — FR, FL, RR, RL; /joint_states — в своём порядке.
    Раскладка по именам даёт одинаковые стопы в обеих кинематиках (каждая — в своём порядке ног)."""
    assert LIDAR_JOINT_NAMES[:3] == ('FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint')
    assert LIDAR_JOINT_NAMES[9:] == ('RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint')
    assert set(LIDAR_JOINT_NAMES) == set(JOINT_NAMES)
    rng = np.random.default_rng(0)
    names = list(JOINT_NAMES)
    rng.shuffle(names)                                   # порядок «как пришёл в /joint_states»
    vals = {n: STAND[n.split('_')[1]] + 0.2 * rng.standard_normal() for n in names}
    pos = [vals[n] for n in names]
    il, ig = joint_index(names), joint_index(names, JOINT_NAMES)
    ql = np.array([pos[i] for i in il])
    qg = np.array([pos[i] for i in ig])
    assert [names[i] for i in il] == list(LIDAR_JOINT_NAMES)
    fl = lidar_core.foot_pos(ql.reshape(4, 3))           # FL, FR, RL, RR
    fg = foot_positions(qg)                              # FR, FL, RR, RL
    assert np.allclose(fl, fg[[1, 0, 3, 2]], atol=1e-12)
    assert joint_index(names[:-1]) is None               # нет сустава — не раскладываем


def _q_lidar(raise_leg=None, dz=0.06):
    """Суставы (порядок ядра) стойки; raise_leg — нога ('FL'…), поднятая на dz (через смену угла бедра/голени)."""
    q = np.array([[STAND['hip'], STAND['thigh'], STAND['calf']]] * 4)
    if raise_leg is not None:
        k = lidar_core.LEGS.index(raise_leg)
        q[k, 2] = -2.0                                    # колено согнуто сильнее — стопа выше
        z0 = lidar_core.foot_pos(q * 0 + [STAND['hip'], STAND['thigh'], STAND['calf']])[k, 2]
        assert lidar_core.foot_pos(q)[k, 2] - z0 > 0.03
    return q.ravel()


def test_stance_by_foot_height_without_contacts():
    """on_contact не вызывается — ядро определяет опору по высоте стоп: поднятая нога (> CONTACT_Z) не опорная."""
    est = LidarEstimator()
    est.set_imu(1.0, _yaw_quat(0.0))
    est.set_joints(1.0, _q_lidar(raise_leg='FR'))
    assert est.core.t_contact == -np.inf                  # контактов не было вовсе
    assert list(est.core.st_prev) == [True, False, True, True]
    est.set_joints(1.004, _q_lidar(raise_leg='RL'))
    assert list(est.core.st_prev) == [True, True, False, True]


def test_heading_is_world_imu_yaw():
    """Курс выхода — мировой yaw IMU с первого сообщения (не обнуляется на старте)."""
    est = LidarEstimator()
    for k in range(50):
        t = 1.0 + k * 0.005
        est.set_imu(t, _yaw_quat(1.2 + 0.01 * k))
        est.set_joints(t, _q_lidar())
    xy, q = est.pose(t)
    assert abs(quat_yaw(q) - (1.2 + 0.49)) < 1e-9
    assert abs(np.linalg.norm(q) - 1) < 1e-12
    assert np.allclose(xy, 0.0)


def test_nan_inf_inputs_rejected_pose_stays_finite():
    est = LidarEstimator()
    assert est.pose(0.0) is None
    est.set_imu(1.0, _yaw_quat(0.3))
    est.set_joints(1.0, _q_lidar())
    good = est.pose(1.0)
    bad = [est.set_imu(1.01, [math.nan, 0, 0, 1]), est.set_imu(1.01, [0, 0, 0, 0]), est.set_imu(math.inf, _yaw_quat(0)),
           est.set_joints(1.01, np.r_[_q_lidar()[:11], math.inf]), est.set_joints(1.01, np.zeros(11)),
           est.set_joints(math.nan, _q_lidar()),
           est.set_scan(1.01, np.ones(720), math.nan, 0.01), est.set_scan(math.nan, np.ones(720), -3.14, 0.01)]
    assert not any(bad) and est.rejected == len(bad)
    # дальности NaN/Inf — допустимы (лучи без отражения): ядро их отсеивает
    assert est.set_scan(1.01, np.r_[np.full(360, np.inf), np.full(360, np.nan)], -math.pi, 2 * math.pi / 720)
    xy, q = est.pose(1.02)
    assert np.all(np.isfinite(xy)) and np.all(np.isfinite(q))
    assert np.allclose(q, good[1])


def test_feed_orders_events_by_stamp():
    """Пакет событий разбирается по штампам: скан со штампом раньше IMU ждёт её в ядре, суставы — после IMU."""
    est = LidarEstimator()
    est.feed([('js', 1.0, _q_lidar()), ('imu', 1.0, _yaw_quat(0.0))])
    assert est.core.f_prev is not None                    # суставы разобраны после IMU (иначе R нет и они пропущены)
