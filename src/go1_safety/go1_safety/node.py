"""ROS 2 узел Safety Block (T2): /lowcmd_bare → проверка → /lowcmd. Логика — core.SafetyCore, решения — docs/T2_DESIGN.md."""
import math
import os
import time

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd, LowState

from .collision import CollisionChecker
from .core import MAX_STEP_RAD, MODE_SERVO, N_MOTORS, SafetyCore

FIELDS = ('q', 'dq', 'kp', 'kd', 'tau')


def read_fields(msg):
    cmds = msg.motor_cmd
    return {
        'q': [c.q for c in cmds], 'dq': [c.dq for c in cmds], 'kp': [c.kp for c in cmds],
        'kd': [c.kd for c in cmds], 'tau': [c.tau for c in cmds],
    }


class SafetyBlock(Node):
    def __init__(self):
        super().__init__('go1_safety_block')
        model_dir = os.environ.get('GO1_MODEL_DIR', '/opt/go1/models/unitree_go1')
        p = self.declare_parameter
        model_path = p('model_path', os.path.join(model_dir, 'go1.xml')).value
        rate = float(p('rate_hz', 200.0).value)
        timeout = float(p('input_timeout', 0.3).value)
        max_step_deg = float(p('max_step_deg', math.degrees(MAX_STEP_RAD)).value)
        threshold = float(p('collision_threshold', 0.0005).value)
        in_topic = p('input_topic', '/lowcmd_bare').value
        out_topic = p('output_topic', '/lowcmd').value
        state_topic = p('state_topic', '/lowstate').value
        self.stats_period = float(p('stats_period', 10.0).value)

        self.checker = CollisionChecker(model_path, threshold=threshold, stop_threshold=threshold / 2)
        self.core = SafetyCore(self.checker, max_step=math.radians(max_step_deg), input_timeout=timeout)

        # BEST_EFFORT-подписка совместима и с RELIABLE, и с BEST_EFFORT издателем Judge; выход RELIABLE — с любой подпиской
        sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        pub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(LowCmd, out_topic, pub_qos)
        self.create_subscription(LowCmd, in_topic, self._on_cmd, sub_qos)
        self.state_sub = self.create_subscription(LowState, state_topic, self._on_state, sub_qos)

        self.out = LowCmd()
        self.out.head = [0xFE, 0xEF]
        self.out.level_flag = 0xFF
        for i in range(N_MOTORS, len(self.out.motor_cmd)):   # 12..19 не используются симом — нули
            self.out.motor_cmd[i].mode = 0

        # Таймер на steady-часах: /clock идёт 100 Гц и может замереть; сим MuJoCo не обгоняет реальное время,
        # поэтому 200 Гц wall ≥ 200 Гц sim. Тайм-аут входа считается по sim time (self.get_clock()).
        self.create_timer(1.0 / rate, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.create_timer(self.stats_period, self._log_stats, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.pending = None
        self.pending_t = 0.0
        self.n_in = 0
        self.n_pub = 0
        self.tick_cpu = 0.0
        self.tick_cpu_max = 0.0
        self.was_holding = None
        self.get_logger().info(
            f'Safety Block: {in_topic} → {out_topic}, {rate:.0f} Гц (steady), тайм-аут входа {timeout} с sim, '
            f'шаг ≤ {max_step_deg}°, коллизии: 655 пар, порог {threshold * 1000:.2f} мм')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(N_MOTORS)], dtype=float)
        if not np.all(np.isfinite(q)):
            return
        self.core.latch_measured(q)
        self.get_logger().info(f'Стартовая поза из /lowstate: {np.round(self.core.out_q, 3).tolist()}')
        self.destroy_subscription(self.state_sub)   # дальше /lowstate (500 Гц) не нужен
        self.state_sub = None

    def _on_cmd(self, msg):
        # Только запоминаем: разбор 20×5 полей в Python дорог при частом входе (500 Гц → ~30% ядра).
        # Такт обрабатывает самое свежее сообщение; невалидное отклоняется целиком (ТЗ), промежуточные валидные
        # цели заменяются более новой — выход всё равно идёт к последней принятой цели рампой.
        self.pending = msg
        self.pending_t = self._now()
        self.n_in += 1

    def _process_pending(self):
        msg, t_in = self.pending, self.pending_t
        self.pending = None
        if len(msg.motor_cmd) != 20:
            self.core.stats.rejected_shape += 1
            return
        if not self.core.on_command(read_fields(msg), t_in):
            self.get_logger().warn('Команда отклонена (NaN/Inf или неверная форма) — держу последнюю безопасную цель',
                                   throttle_duration_sec=2.0)

    def _tick(self):
        t0 = time.process_time()
        if self.pending is not None:
            self._process_pending()
        out = self.core.step(self._now())
        if out is None:
            return
        mc = self.out.motor_cmd
        for i in range(N_MOTORS):
            c = mc[i]
            c.mode = MODE_SERVO
            c.q = float(out.q[i])
            c.dq = float(out.dq[i])
            c.kp = float(out.kp[i])
            c.kd = float(out.kd[i])
            c.tau = float(out.tau[i])
        self.pub.publish(self.out)
        self.n_pub += 1
        if out.holding != self.was_holding:
            self.was_holding = out.holding
            self.get_logger().info('Удержание последней принятой цели (нет входа)' if out.holding else 'Вход есть — отрабатываю команды')
        dt = time.process_time() - t0
        self.tick_cpu += dt
        self.tick_cpu_max = max(self.tick_cpu_max, dt)

    def _log_stats(self):
        s = self.core.stats
        self.get_logger().info(
            f'stats: in {self.n_in / self.stats_period:.0f} Гц, pub {self.n_pub / self.stats_period:.0f} Гц, CPU такта {100 * self.tick_cpu / self.stats_period:.1f}% ядра '
            f'(макс {self.tick_cpu_max * 1000:.2f} мс), проверок {self.checker.evals}, принято {s.accepted}, '
            f'NaN/Inf {s.rejected_nonfinite}, клип q {s.clipped_q}, упоры {s.blocked}')
        self.n_in = 0
        self.n_pub = 0
        self.tick_cpu = 0.0
        self.tick_cpu_max = 0.0
        self.checker.evals = 0


def main(args=None):
    rclpy.init(args=args)
    node = SafetyBlock()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
