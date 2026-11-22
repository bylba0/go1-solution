"""T4: узел одометрии Go1 → /odom (odom → trunk) и TF odom → trunk.

Режимы (параметр estimator.mode):
  legs  — core.OdometryEstimator: ориентация IMU + кинематика опорных ног;
  lidar — lidar_mode.LidarEstimator (ядро lidar_core): ориентация IMU + сопоставление сканов /scan + прогноз по ногам;
          z и угловая скорость — от legs-оценщика, который в этом режиме тоже считается (кинематика, дёшево).
          Страховка (fallback.LidarFallback): /scan не приходил дольше estimator.scan_timeout с sim (или поза рабочего
          потока старше) — xy ведётся приращениями legs от последнего выхода; сканы вернулись — плавный возврат к lidar.
Подписки (только эти; список проверяется тестом test/test_node.py):
  /imu/data (sensor_msgs/Imu), /joint_states (sensor_msgs/JointState), /clock (через use_sim_time),
  legs:  /foot_force (std_msgs/Float32MultiArray) — только при use_foot_force:=true;
  lidar: /scan (sensor_msgs/LaserScan, RELIABLE — BEST_EFFORT-подписка данных от сима не получает).
Колбэки только сохраняют последнее сообщение (IMU и сканы — в очередь); расчёт legs — на таймере update_rate (sim),
публикация — publish_rate (sim). Режим lidar: сопоставление сканов — в отдельном рабочем потоке; таймер только
передаёт ему входы и публикует последнюю готовую позу, не дожидаясь расчёта.
ROS_DOMAIN_ID / RMW_IMPLEMENTATION / CYCLONEDDS_URI / ROS_LOCALHOST_ONLY не трогаются.
"""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # до импорта numpy: при лимите CPU потоки BLAS только мешают
os.environ.setdefault('OMP_NUM_THREADS', '1')

import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import rclpy.executors  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Imu, JointState, LaserScan  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from .core import DEFAULTS, JOINT_NAMES, LEGS, OdometryEstimator, all_finite, quat_to_rot  # noqa: E402
from .fallback import LidarFallback  # noqa: E402
from .lidar_mode import LIDAR_JOINT_NAMES, LidarEstimator  # noqa: E402

# Разрешённые входы узла по режимам. Любая другая подписка (в т.ч. /cmd_vel, истина, /gazebo/*, контакты Gazebo,
# камеры) — нарушение.
ALLOWED_INPUT_TOPICS = ('/imu/data', '/joint_states', '/clock', '/foot_force')            # legs
ALLOWED_INPUT_TOPICS_LIDAR = ('/imu/data', '/joint_states', '/clock', '/scan')           # lidar
MODES = ('legs', 'lidar')

SENSOR_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.VOLATILE)
TF_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE)


def _stamp_to_sec(s):
    return s.sec + s.nanosec * 1e-9


class OdometryNode(Node):
    def __init__(self, **kwargs):
        super().__init__('go1_odometry', **kwargs)
        dp = self.declare_parameter
        self.imu_topic = dp('imu_topic', '/imu/data').value
        self.joint_topic = dp('joint_states_topic', '/joint_states').value
        self.foot_force_topic = dp('foot_force_topic', '/foot_force').value
        self.odom_topic = dp('odom_topic', '/odom').value
        self.odom_frame = dp('odom_frame', 'odom').value
        self.base_frame = dp('base_frame', 'trunk').value
        self.publish_tf = dp('publish_tf', True).value
        update_rate = float(dp('update_rate', 200.0).value)
        self.pub_period = 1.0 / float(dp('publish_rate', 50.0).value)
        self.use_foot_force = bool(dp('use_foot_force', False).value)
        self.foot_force_order = list(dp('foot_force_order', ['FL', 'FR', 'RL', 'RR']).value)
        self.foot_force_threshold = float(dp('foot_force_threshold', 5.0).value)
        self.pose_cov = list(dp('pose_covariance_diagonal', [0.01, 0.01, 0.01, 0.001, 0.001, 0.003]).value)
        self.twist_cov = list(dp('twist_covariance_diagonal', [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).value)
        self.mode = dp('estimator.mode', 'legs').value
        if self.mode not in MODES:
            raise ValueError(f'estimator.mode: {MODES}, получено {self.mode!r}')
        self.scan_topic = dp('scan_topic', '/scan').value
        leg_scale = float(dp('lidar.leg_scale', 1.0).value)
        self.lidar_threaded = bool(dp('lidar.worker_thread', True).value)   # false — только для тестов (синхронно)
        self.diag_period = float(dp('diag_period', 10.0).value)             # с wall: строка диагностики в журнал
        est_params = {}
        for k, v in DEFAULTS.items():
            est_params[k] = dp('estimator.' + k, v).value
        self.est = OdometryEstimator(**est_params)
        fb_params = dict(
            scan_timeout=float(dp('estimator.scan_timeout', 1.0).value),
            blend_tau=float(dp('estimator.fallback_blend_tau', 1.0).value),
            resume_delay=float(dp('estimator.fallback_resume_delay', 0.5).value),
            max_correction=float(dp('estimator.fallback_max_correction', 0.25).value))
        lidar = self.mode == 'lidar'
        if lidar and self.est.p['heading_reference'] != 'world':
            raise ValueError('estimator.mode=lidar: курс ядра — мировой yaw IMU, нужен heading_reference=world '
                             '(иначе страховка по legs сменит курс)')
        if lidar and self.use_foot_force:
            raise ValueError('estimator.mode=lidar: /foot_force не используется (use_foot_force должен быть false)')
        allowed = ALLOWED_INPUT_TOPICS_LIDAR if lidar else ALLOWED_INPUT_TOPICS
        inputs = (self.imu_topic, self.joint_topic) + ((self.foot_force_topic,) if self.use_foot_force else ()) \
            + ((self.scan_topic,) if lidar else ())
        for topic in inputs:
            if topic not in allowed:
                raise ValueError(f'подписка на {topic} запрещена (разрешены {allowed})')

        self._imu_msgs = deque(maxlen=100)   # все IMU между тиками (для режима gyro)
        self._js_msg = None
        self._ff_msg = None
        self._ff_t = None
        self._js_index = None
        self._js_index_lidar = None
        self._js_names = None
        self.create_subscription(Imu, self.imu_topic, self._on_imu, SENSOR_QOS)
        self.create_subscription(JointState, self.joint_topic, self._on_joints, SENSOR_QOS)
        if self.use_foot_force:
            self.create_subscription(Float32MultiArray, self.foot_force_topic, self._on_foot_force, SENSOR_QOS)
        # режим lidar: входы рабочему потоку — очередь событий; выход — последняя готовая поза (замена кортежа)
        self.lidar = LidarEstimator(leg_scale=leg_scale) if lidar else None
        self.fb = LidarFallback(**fb_params) if lidar else None
        self._t_scan = None            # sim-время прихода последнего скана (по часам узла, не по штампу)
        self._lidar_in = deque(maxlen=20000)
        self._lidar_pose = None        # (t, xy, quat) — последняя поза рабочего потока
        self._lidar_prev = None        # (t, xy) прошлой опубликованной — для линейной скорости
        self._lidar_lin = np.zeros(3)
        self._lidar_errors = 0
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker = None
        if lidar:
            # RELIABLE: сим публикует /scan так, что BEST_EFFORT-подписка не получает ничего (T4_RECON)
            self.create_subscription(LaserScan, self.scan_topic, self._on_scan,
                                     QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20,
                                                reliability=ReliabilityPolicy.RELIABLE,
                                                durability=DurabilityPolicy.VOLATILE))
            if self.lidar_threaded:
                self._worker = threading.Thread(target=self._lidar_loop, name='go1_odometry_lidar', daemon=True)
                self._worker.start()
        # диагностика: паузы между публикациями по wall и по stamp
        self._diag_t = time.monotonic()
        self._pub_wall = None
        self._pub_stamp = None
        self._gap_wall = [0.0, 0.0]    # [за период, за всё время], с
        self._gap_stamp = [0.0, 0.0]
        self._timer_max = [0.0, 0.0]   # длительность колбэка таймера, с wall
        self._age_max = [0.0, 0.0]     # lidar: возраст позы рабочего потока на публикации, с sim
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_pub = self.create_publisher(TFMessage, '/tf', TF_QOS) if self.publish_tf else None
        self._next_pub = None
        self._last_valid = None   # (pos, quat, lin, ang)
        self.published = 0
        self._pub_eps = 0.5 / update_rate   # допуск: публикация на ближайшем тике расчёта
        self.create_timer(1.0 / update_rate, self._on_timer)
        self.get_logger().info(
            f'T4 odometry: estimator={self.mode}, orientation={self.est.p["orientation_mode"]}, '
            f'{update_rate:.0f} Гц расчёт, '
            f'{1 / self.pub_period:.0f} Гц /odom+TF, foot_force={"да" if self.use_foot_force else "нет"}')

    # ------------------------------------------------------------ колбэки: только сохранить последнее
    def _on_imu(self, msg):
        self._imu_msgs.append(msg)

    def _on_joints(self, msg):
        self._js_msg = msg

    def _on_scan(self, msg):
        ts = _stamp_to_sec(msg.header.stamp)
        now = self.get_clock().now().nanoseconds * 1e-9
        self._t_scan = now if now > 0.0 else ts      # время прихода по sim; до первого /clock — штамп скана
        self._lidar_in.append(('scan', ts, np.asarray(msg.ranges, dtype=np.float32),
                               msg.angle_min, msg.angle_increment))
        self._wake.set()

    def _on_foot_force(self, msg):
        self._ff_msg = msg
        self._ff_t = self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ расчёт
    def _joint_vector(self, msg, lidar=False):
        """Углы по ИМЕНАМ: legs — порядок JOINT_NAMES (FR, FL, RR, RL), lidar — LIDAR_JOINT_NAMES (FL, FR, RL, RR)."""
        names = tuple(msg.name)
        if names != self._js_names:
            idx = {n: i for i, n in enumerate(names)}
            self._js_index = [idx[n] for n in JOINT_NAMES] if all(n in idx for n in JOINT_NAMES) else None
            self._js_index_lidar = [idx[n] for n in LIDAR_JOINT_NAMES] \
                if all(n in idx for n in LIDAR_JOINT_NAMES) else None
            self._js_names = names
        index = self._js_index_lidar if lidar else self._js_index
        if index is None or len(msg.position) != len(names):
            return None
        return [msg.position[i] for i in index]

    def _feed(self):
        lidar = self.lidar is not None
        while self._imu_msgs:
            m = self._imu_msgs.popleft()
            o, w = m.orientation, m.angular_velocity
            t = _stamp_to_sec(m.header.stamp)
            self.est.set_imu(t, (o.x, o.y, o.z, o.w), (w.x, w.y, w.z))
            if lidar:
                self._lidar_in.append(('imu', t, (o.x, o.y, o.z, o.w)))
        m, self._js_msg = self._js_msg, None
        if m is not None:
            q = self._joint_vector(m)
            t = _stamp_to_sec(m.header.stamp)
            if q is None:
                self.est.rejected += 1
            else:
                self.est.set_joints(t, q)
            if lidar:
                ql = self._joint_vector(m, lidar=True)
                if ql is None:
                    self.lidar.rejected += 1
                else:
                    self._lidar_in.append(('js', t, ql))
        m, self._ff_msg = self._ff_msg, None
        if m is not None and len(m.data) == 4 and all_finite(m.data):
            force = dict(zip(self.foot_force_order, m.data))
            if set(force) == set(LEGS):
                self.est.set_contacts(self._ff_t, [force[leg] > self.foot_force_threshold for leg in LEGS])

    def _on_timer(self):
        w0 = time.monotonic()
        self._on_timer_body()
        d = time.monotonic() - w0
        self._timer_max[0], self._timer_max[1] = max(self._timer_max[0], d), max(self._timer_max[1], d)

    def _on_timer_body(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        if t <= 0.0:
            return                     # sim time ещё не пришёл
        self._feed()
        ok = self.est.step(t)
        if self.lidar is not None:
            self._lidar_in.append(('tick', t))
            if self.lidar_threaded:
                self._wake.set()        # не ждём: рабочий поток считает, таймер публикует готовое
            else:
                self._lidar_process()
            ok = self._lidar_output(t) if ok else False
        elif ok:
            pos, quat = self.est.position, self.est.orientation
            lin, ang = self.est.lin_vel, self.est.ang_vel
            if all_finite(pos, quat, lin, ang) and abs(float(np.linalg.norm(quat)) - 1.0) < 1e-6:
                self._last_valid = (pos.copy(), quat.copy(), lin.copy(), ang.copy())
        if time.monotonic() - self._diag_t >= self.diag_period:
            self._diag()
        if self._last_valid is None:
            return
        if self._next_pub is None or t + self._pub_eps >= self._next_pub:
            # следующий момент публикации по sim time; при отставании — без «догоняющей» пачки
            self._next_pub = t + self.pub_period if self._next_pub is None or t - self._next_pub > self.pub_period \
                else self._next_pub + self.pub_period
            self._publish(now.to_msg())

    # ------------------------------------------------------------ режим lidar
    def _lidar_loop(self):
        while not self._stop.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            self._lidar_process()

    def _lidar_process(self):
        """Рабочий поток: разбор накопленных входов по штампам, затем поза на время последнего тика таймера."""
        batch, ticks = [], []
        q = self._lidar_in
        while q:
            e = q.popleft()
            (ticks if e[0] == 'tick' else batch).append(e)
        if not batch and not ticks:
            return
        try:
            self.lidar.feed(batch)
            if ticks:
                t = ticks[-1][1]
                p = self.lidar.pose(t)
                if p is not None:
                    self._lidar_pose = (t, p[0], p[1])
        except Exception as e:  # noqa: BLE001 - ошибка расчёта не останавливает публикацию (последняя валидная поза)
            self._lidar_errors += 1
            self.get_logger().error(f'lidar: {type(e).__name__}: {e}', throttle_duration_sec=1.0)

    def _lidar_output(self, t):
        """Поза для публикации: xy — страховки (поза рабочего потока, последняя готовая, или приращения legs без
        сканов), ориентация — кватернион IMU (мировой курс у обоих оценщиков), z и ω — legs-оценщика."""
        snap = self._lidar_pose
        if snap is not None:
            age = t - snap[0]
            self._age_max[0], self._age_max[1] = max(self._age_max[0], age), max(self._age_max[1], age)
        xy_out = self.fb.update(t, None if snap is None else snap[1], self.est.position[:2], self._t_scan,
                                None if snap is None else snap[0])
        if xy_out is None:
            return False
        if self.fb.source == 'legs' or snap is None:
            self._lidar_prev = None
            quat, lin = self.est.orientation, self.est.lin_vel
            pos = np.array([xy_out[0], xy_out[1], self.est.position[2]])
            if all_finite(pos, quat, lin, self.est.ang_vel) and abs(float(np.linalg.norm(quat)) - 1.0) < 1e-6:
                self._last_valid = (pos, np.array(quat), lin.copy(), self.est.ang_vel.copy())
                return True
            return False
        ts, xy, quat = snap
        pos = np.array([xy_out[0], xy_out[1], self.est.position[2]])
        prev = self._lidar_prev
        if prev is not None and ts > prev[0]:
            v_world = np.array([(xy[0] - prev[1][0]) / (ts - prev[0]), (xy[1] - prev[1][1]) / (ts - prev[0]), 0.0])
            v_body = quat_to_rot(quat).T @ v_world
            if all_finite(v_body):
                self._lidar_lin += self.est.p['twist_alpha'] * (v_body - self._lidar_lin)
        if prev is None or ts > prev[0]:
            self._lidar_prev = (ts, xy)
        lin, ang = self._lidar_lin, self.est.ang_vel
        if all_finite(pos, quat, lin, ang) and abs(float(np.linalg.norm(quat)) - 1.0) < 1e-6:
            self._last_valid = (pos, np.array(quat), lin.copy(), ang.copy())
            return True
        return False

    def _diag(self):
        self._diag_t = time.monotonic()
        msg = (f'/odom: {self.published} опубликовано, пауза макс wall {1000 * self._gap_wall[0]:.0f} мс '
               f'(всего {1000 * self._gap_wall[1]:.0f}), по stamp {1000 * self._gap_stamp[0]:.0f} мс '
               f'(всего {1000 * self._gap_stamp[1]:.0f}); таймер макс {1000 * self._timer_max[0]:.1f} мс wall '
               f'(всего {1000 * self._timer_max[1]:.1f})')
        if self.lidar is not None:
            c, ms = self.lidar.core, np.array(self.lidar.scan_ms)
            msg += (f'; сканы {self.lidar.n_scans_in}: сопоставлено {c.n_match}, пропущено {c.n_skip}, '
                    f'опорных {c.n_key}, вырожд. {c.n_degen}; обработка '
                    + (f'{ms.mean():.1f} мс ср, {ms.max():.1f} макс' if len(ms) else '—')
                    + f'; возраст позы макс {1000 * self._age_max[0]:.0f} мс sim (всего {1000 * self._age_max[1]:.0f})'
                    + f'; очередь {len(self._lidar_in)}; отброшено {self.lidar.rejected}; ошибок {self._lidar_errors}')
            fb, t = self.fb, self.get_clock().now().nanoseconds * 1e-9
            msg += (f'; источник {fb.source}, переключений {fb.switches}, без сканов {fb.no_scan_s(t, self._t_scan):.1f} с sim')
        self.get_logger().info(msg)
        self._gap_wall[0] = self._gap_stamp[0] = self._timer_max[0] = self._age_max[0] = 0.0

    def destroy_node(self):
        self._stop.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        return super().destroy_node()

    def _publish(self, stamp):
        pos, quat, lin, ang = self._last_valid
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = self.odom_frame
        od.child_frame_id = self.base_frame
        pp, po = od.pose.pose.position, od.pose.pose.orientation
        pp.x, pp.y, pp.z = (float(v) for v in pos)
        po.x, po.y, po.z, po.w = (float(v) for v in quat)
        tl, ta = od.twist.twist.linear, od.twist.twist.angular   # в системе child_frame_id (trunk)
        tl.x, tl.y, tl.z = (float(v) for v in lin)
        ta.x, ta.y, ta.z = (float(v) for v in ang)
        for i in range(6):
            od.pose.covariance[7 * i] = float(self.pose_cov[i])
            od.twist.covariance[7 * i] = float(self.twist_cov[i])
        self.odom_pub.publish(od)
        if self.tf_pub is not None:
            tf = TransformStamped()
            tf.header = od.header
            tf.child_frame_id = self.base_frame
            tr, rot = tf.transform.translation, tf.transform.rotation
            tr.x, tr.y, tr.z = pp.x, pp.y, pp.z
            rot.x, rot.y, rot.z, rot.w = po.x, po.y, po.z, po.w
            self.tf_pub.publish(TFMessage(transforms=[tf]))
        self.published += 1
        w, s = time.monotonic(), _stamp_to_sec(stamp)
        if self._pub_wall is not None:
            for acc, gap in ((self._gap_wall, w - self._pub_wall), (self._gap_stamp, s - self._pub_stamp)):
                acc[0], acc[1] = max(acc[0], gap), max(acc[1], gap)
        self._pub_wall, self._pub_stamp = w, s


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
