"""Walking policy node: /lowstate + /cmd_vel -> WalkController -> /lowcmd_bare."""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # before numpy: one BLAS thread, the CPU is shared with the simulator

import gc  # noqa: E402
import math  # noqa: E402
import signal  # noqa: E402

import rclpy  # noqa: E402
from ament_index_python.packages import get_package_share_directory  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rclpy.serialization import deserialize_message, serialize_message  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402
from unitree_go.msg import LowCmd, LowState  # noqa: E402

from .controller import WalkController  # noqa: E402
from .msg_io import (CMD_FIELDS, CMD_SPEC, STATE_SPEC, cmd_from_msg, cmd_to_msg, locate, markers, read, same,  # noqa: E402
                     state_from_msg, state_to_msg, write)


class WalkNode(Node):
    def __init__(self):
        super().__init__('go1_walk', start_parameter_services=False)
        policy = self.declare_parameter('policy', '').value or os.path.join(
            get_package_share_directory('go1_walk'), 'policy', 'policy.npz')
        # kp 45 instead of the trained 50 (training randomizes gains by +-10 %): far fewer torque-limited steps
        kp = self.declare_parameter('kp', 45.0).value
        self.ctrl = WalkController(policy, heading_hold=self.declare_parameter('heading_hold', True).value, kp=kp)
        self.get_logger().info(f'policy {policy}, kp {kp}, kd {float(self.ctrl.kd[0])}')
        self.cmd_vel = None
        self.started = False
        # /lowstate is read from raw CDR bytes: offsets come from a serialized template and are checked once against
        # full deserialization. Commands are written into a serialized template the same way.
        try:
            marks = markers(STATE_SPEC)
            tmpl = serialize_message(state_to_msg(LowState(), marks))
            self.state_layout, self.state_len = locate(tmpl, marks), len(tmpl)
            marks = markers(CMD_SPEC)
            tmpl = serialize_message(cmd_to_msg(LowCmd(), marks))
            self.cmd_layout, self.cmd_buf = locate(tmpl, marks), bytearray(tmpl)
        except ValueError as e:
            self.get_logger().warning(f'raw message layout not found ({e}): full (de)serialization')
            self.state_layout = self.cmd_layout = None
        self.state_checked = self.cmd_checked = False
        self.msg = LowCmd()
        self.pub = self.create_publisher(LowCmd, '/lowcmd_bare',
                                         QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        # /cmd_vel from any publisher: BEST_EFFORT/VOLATILE matches every writer, RELIABLE/TRANSIENT_LOCAL also gets
        # a message published once before this node started (latched). The last received command stays in effect.
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, be)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(LowState, '/lowstate', self.on_state, be, raw=True)

    def on_cmd_vel(self, m):
        cmd = (m.linear.x, m.linear.y, m.angular.z)
        if not all(math.isfinite(v) for v in cmd):
            self.get_logger().warning('non-finite /cmd_vel ignored', throttle_duration_sec=1.0)
            return
        if self.cmd_vel is None:
            self.get_logger().info(f'first /cmd_vel: {cmd[0]:.3f} {cmd[1]:.3f} {cmd[2]:.3f}')
        self.cmd_vel = cmd

    def parse(self, raw):
        if self.state_layout is not None and len(raw) != self.state_len:
            self.get_logger().warning(f'/lowstate of {len(raw)} bytes, expected {self.state_len}: full deserialization')
            self.state_layout = None
        if self.state_layout is None:
            return state_from_msg(deserialize_message(raw, LowState))
        s = read(raw, self.state_layout)
        if not self.state_checked:
            self.state_checked = True
            if not same(state_from_msg(deserialize_message(raw, LowState)), s):
                self.get_logger().warning('raw /lowstate fields do not match: full deserialization')
                self.state_layout = None
                return self.parse(raw)
        return s

    def publish(self, cmd):
        if self.cmd_layout is not None:
            write(self.cmd_buf, self.cmd_layout, {f: cmd[f] for f in CMD_FIELDS})
            data = bytes(self.cmd_buf)
            if not self.cmd_checked:
                self.cmd_checked = True
                if not same(cmd_from_msg(cmd_to_msg(LowCmd(), cmd)), cmd_from_msg(deserialize_message(data, LowCmd))):
                    self.get_logger().warning('raw /lowcmd_bare fields do not match: full serialization')
                    self.cmd_layout = None
                    return self.publish(cmd)
            self.pub.publish(data)
        else:
            self.pub.publish(cmd_to_msg(self.msg, cmd))

    def on_state(self, raw):
        s = self.parse(raw)
        cmd = self.ctrl.on_lowstate(s, self.cmd_vel, int(s['tick']) / 1000.0)
        if cmd is None:
            return
        self.publish(cmd)
        if not self.started:
            self.started = True
            self.get_logger().info(f'first command at tick {int(s["tick"])}')
            gc.collect()
            gc.freeze()  # keep start-up objects out of later collections (shorter GC pauses)


def main():
    # Own signal handling: rclpy's handler shuts the context down mid-take and the node exits with an error.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    node = WalkNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    while not stop:
        executor.spin_once(timeout_sec=0.05)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
