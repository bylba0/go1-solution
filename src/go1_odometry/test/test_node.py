"""Тесты узла T4 с rclpy (без симулятора): белый список подписок, отсутствие запрещённых входов в коде,
формат /odom и TF, NaN на входе. Узел создаётся в процессе, колбэки и таймер вызываются напрямую."""
import math
import os
import pathlib
import re

import pytest

rclpy = pytest.importorskip('rclpy')
from builtin_interfaces.msg import Time  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import Imu, JointState  # noqa: E402

from go1_odometry.core import JOINT_NAMES  # noqa: E402
from go1_odometry.node import ALLOWED_INPUT_TOPICS, OdometryNode  # noqa: E402

PKG_DIR = pathlib.Path(__file__).resolve().parents[1] / 'go1_odometry'
# Запрещённые входы (ТЗ_T4: нет доступа к истине; не воспроизводить командную траекторию)
FORBIDDEN = re.compile(r'cmd_vel|ground_truth|gazebo|go1_contacts|foot_contacts|camera|champ|model_states|link_states'
                       r'|apply_force|go1_ready')


@pytest.fixture(scope='module', autouse=True)
def ros():
    os.environ.setdefault('ROS_DOMAIN_ID', '48')
    rclpy.init()
    yield
    rclpy.shutdown()


def _node(**params):
    overrides = [Parameter(k, value=v) for k, v in params.items()]
    return OdometryNode(parameter_overrides=overrides)


def _subscribed(node):
    return sorted(s.topic_name for s in node.subscriptions)


@pytest.mark.parametrize('use_ff', [False, True])
def test_subscriptions_whitelist(use_ff):
    node = _node(use_sim_time=True, use_foot_force=use_ff)
    try:
        topics = _subscribed(node)
        assert set(topics) <= set(ALLOWED_INPUT_TOPICS), topics
        assert '/imu/data' in topics and '/joint_states' in topics
        assert ('/foot_force' in topics) == use_ff
        assert not [t for t in topics if FORBIDDEN.search(t)]
        assert sorted(ALLOWED_INPUT_TOPICS) == ['/clock', '/foot_force', '/imu/data', '/joint_states']
    finally:
        node.destroy_node()


def test_forbidden_topic_parameter_rejected():
    for bad in ('/cmd_vel', '/odom/ground_truth', '/go1_contacts/FL'):
        with pytest.raises(ValueError):
            _node(imu_topic=bad)


def test_source_has_no_forbidden_inputs():
    """В коде пакета нет запрещённых топиков/типов (кроме строки-фильтра в этом тесте)."""
    for f in PKG_DIR.glob('*.py'):
        text = f.read_text(encoding='utf-8')
        code = '\n'.join(line.split('#', 1)[0] for line in text.splitlines())
        code = re.sub(r'""".*?"""', '', code, flags=re.S)
        assert not FORBIDDEN.search(code), f'{f.name}: {FORBIDDEN.search(code).group(0)}'
        assert 'gazebo_msgs' not in text and 'champ_msgs' not in text
        assert 'create_subscription' not in code or f.name == 'node.py'


def _stamp(t):
    return Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def _imu(t, q=(0.0, 0.0, 0.0, 1.0)):
    m = Imu()
    m.header.stamp = _stamp(t)
    m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = q
    return m


def _joints(t, value=None):
    """Порядок НЕ алфавитный и не как в JOINT_NAMES — узел должен сопоставлять по именам."""
    names = list(reversed(JOINT_NAMES))
    stand = {'hip': 0.0, 'thigh': 0.8, 'calf': -1.6}
    m = JointState()
    m.header.stamp = _stamp(t)
    m.name = names
    m.position = [value if value is not None else stand[n.split('_')[1]] for n in names]
    m.velocity = [0.0] * 12
    m.effort = [float('nan')] * 12
    return m


class _Capture:
    def __init__(self, node):
        self.odom, self.tf = [], []
        node.odom_pub.publish = self.odom.append
        node.tf_pub.publish = self.tf.append


def _run(node, cap, t0, t1, feed):
    """Имитация таймера 200 Гц по sim time: перед каждым тиком — входы feed(t)."""
    from rclpy.clock import ClockType
    from rclpy.time import Time as RTime
    n = int(round((t1 - t0) * 200))
    for k in range(n):
        t = t0 + k * 0.005
        feed(t)
        node.get_clock().now = (lambda tt=t: RTime(nanoseconds=int(tt * 1e9), clock_type=ClockType.ROS_TIME))
        node._on_timer()


def test_odom_and_tf_format_rate_and_nan_inputs():
    node = _node(use_sim_time=True)
    cap = _Capture(node)
    try:
        def good(t):
            node._on_imu(_imu(t))
            node._on_joints(_joints(t))

        def bad(t):
            node._on_imu(_imu(t, (math.nan, 0.0, 0.0, 1.0)))
            node._on_joints(_joints(t, value=math.inf))
        _run(node, cap, 1.0, 3.0, good)
        n_good = len(cap.odom)
        assert 95 <= n_good <= 101                 # 50 Гц sim за 2 с
        _run(node, cap, 3.0, 4.0, bad)
        assert node.est.rejected >= 300
        assert len(cap.odom) - n_good >= 48        # при плохом входе публикация не прекращается
        last_good = cap.odom[n_good - 1]
        for od, tfm in zip(cap.odom, cap.tf):
            assert od.header.frame_id == 'odom' and od.child_frame_id == 'trunk'
            tr = tfm.transforms[0]
            assert tr.header.frame_id == 'odom' and tr.child_frame_id == 'trunk'
            assert tr.header.stamp == od.header.stamp
            p, q = od.pose.pose.position, od.pose.pose.orientation
            assert (tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z) == (p.x, p.y, p.z)
            assert (tr.transform.rotation.x, tr.transform.rotation.w) == (q.x, q.w)
            vals = [p.x, p.y, p.z, q.x, q.y, q.z, q.w] + list(od.pose.covariance) + list(od.twist.covariance)
            assert all(math.isfinite(v) for v in vals)
            assert abs(math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2) - 1) < 1e-9
            assert od.pose.covariance[0] > 0 and od.twist.covariance[35] > 0
        # во время плохого входа поза = последняя валидная
        for od in cap.odom[n_good:]:
            assert od.pose.pose.position == last_good.pose.pose.position
        stamps = [o.header.stamp.sec + o.header.stamp.nanosec * 1e-9 for o in cap.odom]
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
        assert max(b - a for a, b in zip(stamps, stamps[1:])) <= 0.0201
    finally:
        node.destroy_node()


def test_no_publication_before_sim_time():
    node = _node(use_sim_time=True)
    cap = _Capture(node)
    try:
        node._on_imu(_imu(0.0))
        node._on_timer()                          # /clock ещё не пришёл → now == 0
        assert cap.odom == []
    finally:
        node.destroy_node()
