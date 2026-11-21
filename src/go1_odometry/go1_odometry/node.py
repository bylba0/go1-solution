"""T4: узел одометрии Go1 → /odom (odom → trunk) и TF odom → trunk.

Подписки (только эти; список проверяется тестом test/test_node.py):
  /imu/data (sensor_msgs/Imu), /joint_states (sensor_msgs/JointState), /clock (через use_sim_time),
  /foot_force (std_msgs/Float32MultiArray) — только при use_foot_force:=true.
Колбэки только сохраняют последнее сообщение; расчёт — на таймере update_rate (sim), публикация — publish_rate (sim).
ROS_DOMAIN_ID / RMW_IMPLEMENTATION / CYCLONEDDS_URI / ROS_LOCALHOST_ONLY не трогаются.
"""
from collections import deque

import numpy as np
import rclpy
import rclpy.executors
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

from .core import DEFAULTS, JOINT_NAMES, LEGS, OdometryEstimator, all_finite

# Разрешённые входы узла. Любая другая подписка (в т.ч. /cmd_vel, истина, /gazebo/*, контакты Gazebo,
# камеры) — нарушение.
ALLOWED_INPUT_TOPICS = ('/imu/data', '/joint_states', '/clock', '/foot_force')

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
        est_params = {}
        for k, v in DEFAULTS.items():
            est_params[k] = dp('estimator.' + k, v).value
        self.est = OdometryEstimator(**est_params)
        for topic in (self.imu_topic, self.joint_topic) + ((self.foot_force_topic,) if self.use_foot_force else ()):
            if topic not in ALLOWED_INPUT_TOPICS:
                raise ValueError(f'подписка на {topic} запрещена (разрешены {ALLOWED_INPUT_TOPICS})')

        self._imu_msgs = deque(maxlen=100)   # все IMU между тиками (для режима gyro)
        self._js_msg = None
        self._ff_msg = None
        self._ff_t = None
        self._js_index = None
        self._js_names = None
        self.create_subscription(Imu, self.imu_topic, self._on_imu, SENSOR_QOS)
        self.create_subscription(JointState, self.joint_topic, self._on_joints, SENSOR_QOS)
        if self.use_foot_force:
            self.create_subscription(Float32MultiArray, self.foot_force_topic, self._on_foot_force, SENSOR_QOS)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_pub = self.create_publisher(TFMessage, '/tf', TF_QOS) if self.publish_tf else None
        self._next_pub = None
        self._last_valid = None   # (pos, quat, lin, ang)
        self.published = 0
        self._pub_eps = 0.5 / update_rate   # допуск: публикация на ближайшем тике расчёта
        self.create_timer(1.0 / update_rate, self._on_timer)
        self.get_logger().info(
            f'T4 odometry: mode={self.est.p["orientation_mode"]}, {update_rate:.0f} Гц расчёт, '
            f'{1 / self.pub_period:.0f} Гц /odom+TF, foot_force={"да" if self.use_foot_force else "нет"}')

    # ------------------------------------------------------------ колбэки: только сохранить последнее
    def _on_imu(self, msg):
        self._imu_msgs.append(msg)

    def _on_joints(self, msg):
        self._js_msg = msg

    def _on_foot_force(self, msg):
        self._ff_msg = msg
        self._ff_t = self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ расчёт
    def _joint_vector(self, msg):
        names = tuple(msg.name)
        if names != self._js_names:
            idx = {n: i for i, n in enumerate(names)}
            self._js_index = [idx[n] for n in JOINT_NAMES] if all(n in idx for n in JOINT_NAMES) else None
            self._js_names = names
        if self._js_index is None or len(msg.position) != len(names):
            return None
        return [msg.position[i] for i in self._js_index]

    def _feed(self):
        while self._imu_msgs:
            m = self._imu_msgs.popleft()
            o, w = m.orientation, m.angular_velocity
            self.est.set_imu(_stamp_to_sec(m.header.stamp), (o.x, o.y, o.z, o.w), (w.x, w.y, w.z))
        m, self._js_msg = self._js_msg, None
        if m is not None:
            q = self._joint_vector(m)
            if q is None:
                self.est.rejected += 1
            else:
                self.est.set_joints(_stamp_to_sec(m.header.stamp), q)
        m, self._ff_msg = self._ff_msg, None
        if m is not None and len(m.data) == 4 and all_finite(m.data):
            force = dict(zip(self.foot_force_order, m.data))
            if set(force) == set(LEGS):
                self.est.set_contacts(self._ff_t, [force[leg] > self.foot_force_threshold for leg in LEGS])

    def _on_timer(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        if t <= 0.0:
            return                     # sim time ещё не пришёл
        self._feed()
        if self.est.step(t):
            pos, quat = self.est.position, self.est.orientation
            lin, ang = self.est.lin_vel, self.est.ang_vel
            if all_finite(pos, quat, lin, ang) and abs(float(np.linalg.norm(quat)) - 1.0) < 1e-6:
                self._last_valid = (pos.copy(), quat.copy(), lin.copy(), ang.copy())
        if self._last_valid is None:
            return
        if self._next_pub is None or t + self._pub_eps >= self._next_pub:
            # следующий момент публикации по sim time; при отставании — без «догоняющей» пачки
            self._next_pub = t + self.pub_period if self._next_pub is None or t - self._next_pub > self.pub_period \
                else self._next_pub + self.pub_period
            self._publish(now.to_msg())

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
