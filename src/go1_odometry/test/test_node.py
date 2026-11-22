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
from sensor_msgs.msg import Imu, JointState, LaserScan  # noqa: E402

from go1_odometry.core import JOINT_NAMES, quat_yaw  # noqa: E402
from go1_odometry.lidar_mode import LIDAR_JOINT_NAMES  # noqa: E402
from go1_odometry.node import ALLOWED_INPUT_TOPICS, ALLOWED_INPUT_TOPICS_LIDAR, OdometryNode  # noqa: E402

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


def test_subscriptions_whitelist_lidar():
    node = _node(use_sim_time=True, **{'estimator.mode': 'lidar', 'lidar.worker_thread': False})
    try:
        topics = _subscribed(node)
        assert set(topics) <= set(ALLOWED_INPUT_TOPICS_LIDAR), topics
        assert {'/imu/data', '/joint_states', '/scan'} <= set(topics)
        assert '/foot_force' not in topics
        assert not [t for t in topics if FORBIDDEN.search(t)]
        assert sorted(ALLOWED_INPUT_TOPICS_LIDAR) == ['/clock', '/imu/data', '/joint_states', '/scan']
        scan = [s for s in node.subscriptions if s.topic_name == '/scan'][0]
        assert scan.qos_profile.reliability == rclpy.qos.ReliabilityPolicy.RELIABLE
    finally:
        node.destroy_node()


def test_default_mode_is_legs():
    node = _node(use_sim_time=True)
    try:
        assert node.mode == 'legs' and node.lidar is None and '/scan' not in _subscribed(node)
    finally:
        node.destroy_node()


@pytest.mark.parametrize('mode', ['legs', 'lidar'])
def test_forbidden_topic_parameter_rejected(mode):
    for bad in ('/cmd_vel', '/cmd_vel_gated', '/odom/ground_truth', '/go1_contacts/FL', '/gazebo/model_states',
                '/camera/color/image_raw'):
        for key in ('imu_topic', 'joint_states_topic') + (('scan_topic',) if mode == 'lidar' else ()):
            with pytest.raises(ValueError):
                _node(**{key: bad, 'estimator.mode': mode, 'lidar.worker_thread': False})
    if mode == 'lidar':
        with pytest.raises(ValueError):          # в lidar /foot_force не используется
            _node(use_foot_force=True, **{'estimator.mode': 'lidar', 'lidar.worker_thread': False})
    with pytest.raises(ValueError):
        _node(**{'estimator.mode': 'judge'})


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
            v = tr.transform.translation
            assert (v.x, v.y, v.z) == (p.x, p.y, p.z)
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


# ---------------------------------------------------------------- режим lidar
LIDAR = {'estimator.mode': 'lidar', 'lidar.worker_thread': False}


def _yawq(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def _scan(t, value=2.0):
    m = LaserScan()
    m.header.stamp = _stamp(t)
    m.angle_min, m.angle_increment = -math.pi, 2 * math.pi / 720
    m.range_min, m.range_max = 0.12, 20.0
    m.ranges = [value] * 720
    return m


def test_lidar_joints_by_name_and_world_heading():
    """Суставы приходят в своём порядке — в ядро lidar идут FL, FR, RL, RR по именам; курс /odom — мировой yaw IMU."""
    node = _node(use_sim_time=True, **LIDAR)
    cap = _Capture(node)
    try:
        js = _joints(1.0)
        js.position = [float(i) for i in range(12)]          # значение = номер в сообщении
        node._on_joints(js)
        node._on_imu(_imu(1.0, _yawq(2.0)))
        node._feed()
        ev = [e for e in node._lidar_in if e[0] == 'js'][0]
        assert ev[2] == [float(js.name.index(n)) for n in LIDAR_JOINT_NAMES]
        node._lidar_in.clear()

        def feed(t):
            node._on_imu(_imu(t, _yawq(2.0)))
            node._on_joints(_joints(t))
        _run(node, cap, 1.0, 1.5, feed)
        q = cap.odom[-1].pose.pose.orientation
        assert abs(quat_yaw((q.x, q.y, q.z, q.w)) - 2.0) < 1e-9
        assert node.lidar.core.t_contact == -float('inf')    # контакты не подаются: опора по высоте стоп
    finally:
        node.destroy_node()


def test_lidar_odom_format_and_nan_inputs():
    node = _node(use_sim_time=True, **LIDAR)
    cap = _Capture(node)
    try:
        def good(t):
            node._on_imu(_imu(t))
            node._on_joints(_joints(t))
            if round(t * 200) % 10 == 0:
                node._on_scan(_scan(t))

        def bad(t):
            node._on_imu(_imu(t, (math.nan, 0.0, 0.0, 1.0)))
            node._on_joints(_joints(t, value=math.inf))
            if round(t * 200) % 10 == 0:
                node._on_scan(_scan(t, value=math.nan))
        _run(node, cap, 1.0, 3.0, good)
        n_good = len(cap.odom)
        assert 95 <= n_good <= 101
        _run(node, cap, 3.0, 4.0, bad)
        assert node.lidar.rejected >= 300
        assert len(cap.odom) - n_good >= 48
        for od, tfm in zip(cap.odom, cap.tf):
            tr = tfm.transforms[0]
            assert od.header.frame_id == 'odom' and od.child_frame_id == 'trunk' and tr.child_frame_id == 'trunk'
            assert tr.header.stamp == od.header.stamp
            p, q = od.pose.pose.position, od.pose.pose.orientation
            v = tr.transform.translation
            assert (v.x, v.y, v.z) == (p.x, p.y, p.z)
            vals = [p.x, p.y, p.z, q.x, q.y, q.z, q.w, od.twist.twist.linear.x, od.twist.twist.angular.z]
            assert all(math.isfinite(v) for v in vals)
            assert abs(math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2) - 1) < 1e-9
            assert 0.2 < p.z < 0.4                            # z — по кинематике ног (legs-оценщик)
        stamps = [o.header.stamp.sec + o.header.stamp.nanosec * 1e-9 for o in cap.odom]
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
        assert max(b - a for a, b in zip(stamps, stamps[1:])) <= 0.0201
    finally:
        node.destroy_node()


def test_lidar_slow_matching_does_not_delay_publication():
    """Рабочий поток занят (сопоставление 0,3 с) — таймер не ждёт: каждый тик короткий, /odom идёт по расписанию."""
    import threading
    import time
    node = _node(use_sim_time=True, **{'estimator.mode': 'lidar', 'lidar.worker_thread': True})
    cap = _Capture(node)
    try:
        real = node.lidar.feed
        busy = threading.Event()

        def slow(events):
            if any(e[0] == 'scan' for e in events):
                busy.set()
                time.sleep(0.3)
            real(events)
        node.lidar.feed = slow
        from rclpy.clock import ClockType
        from rclpy.time import Time as RTime
        worst = 0.0
        for k in range(400):                                   # 2 с sim, ~1 с wall
            t = 1.0 + k * 0.005
            node._on_imu(_imu(t))
            node._on_joints(_joints(t))
            if k % 10 == 0:
                node._on_scan(_scan(t))
            node.get_clock().now = (lambda tt=t: RTime(nanoseconds=int(tt * 1e9), clock_type=ClockType.ROS_TIME))
            w0 = time.perf_counter()
            node._on_timer()
            worst = max(worst, time.perf_counter() - w0)
            time.sleep(0.0025)
        assert busy.is_set()
        assert worst < 0.05, worst                              # тик не ждёт сопоставления (0,3 с)
        # первая поза — после первого (медленного) пакета; дальше /odom без пропусков, хотя поток занят каждые 50 мс sim
        stamps = [o.header.stamp.sec + o.header.stamp.nanosec * 1e-9 for o in cap.odom]
        assert len(stamps) >= (2.995 - stamps[0]) * 50 - 1
        assert max(b - a for a, b in zip(stamps, stamps[1:])) <= 0.0201
    finally:
        node.destroy_node()
