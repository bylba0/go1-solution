"""ROS 2 узел Safety Block (T2): /lowcmd_bare → проверка → /lowcmd. Логика — core.SafetyCore, решения — docs/T2_DESIGN.md.

Потоки (п. 4.5 ТЗ_T2: пауза между /lowcmd ≤ 100 мс sim time — должна держаться и на медленном CPU):
  * executor (один поток): таймер публикации 200 Гц на steady-часах + приём сообщений. Таймер НИКОГДА не считает
    коллизии и не разбирает вход — он отправляет последнюю уже проверенную безопасную позу (готовые байты, под lock).
    Вход принимается raw-подпиской (байты без десериализации) — в слот «последнее сообщение».
  * рабочий поток: просыпается после каждой публикации, разбирает только последнее входное сообщение, двигает цель
    и проверяет пути с бюджетом по ВРЕМЕНИ (compute_budget_ms за порцию, продолжение — в следующей порции).
    Не успел — таймер публикует прежнюю безопасную позу. Следующая поза считается только после того, как предыдущая
    опубликована, поэтому шаг между соседними /lowcmd ≤ max_step при любой загрузке.
  * немедленная публикация (immediate_publish, по умолчанию да): колбэк входа сразу делает шаг ядра для принятой
    команды и публикует результат — безопасная команда выходит без задержки и без изменений даже при входе 500 Гц
    (раньше до выхода доходила лишь та, что совпала с тиком 200 Гц). Перед этим колбэк выпускает ещё не опубликованную
    позу рабочего потока: публикации идут строго в порядке расчёта ядром (каждая следующая ≤ max_step от предыдущей,
    путь проверен). Все публикации — под out_lock; таймер 200 Гц остаётся для ровности потока и удержания.
"""
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd, LowState

from .collision import CollisionChecker
from .core import MAX_STEP_RAD, N_MOTORS, SafetyCore
from .lowcmd_codec import LowCmdCodec


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
        self.budget_s = float(p('compute_budget_ms', 1.2).value) / 1000.0
        self.immediate = bool(p('immediate_publish', True).value)
        in_topic = p('input_topic', '/lowcmd_bare').value
        out_topic = p('output_topic', '/lowcmd').value
        state_topic = p('state_topic', '/lowstate').value
        self.stats_period = float(p('stats_period', 10.0).value)
        self.period = 1.0 / rate

        # Чаще отдавать GIL рабочим потоком — публикация не ждёт дольше ~1 мс.
        sys.setswitchinterval(0.001)

        self.checker = CollisionChecker(model_path, threshold=threshold, stop_threshold=threshold / 2)
        self.core = SafetyCore(self.checker, max_step=math.radians(max_step_deg), input_timeout=timeout,
                               budget_now_s=self.budget_s, budget_tick_s=self.budget_s)
        self.codec = LowCmdCodec()

        self.core_lock = threading.Lock()      # SafetyCore: рабочий поток и латч /lowstate
        self.in_lock = threading.Lock()        # слот входа
        self.out_lock = threading.Lock()       # слот готовой выходной позы
        self.in_raw, self.in_t, self.n_in = None, 0.0, 0
        self.out_bytes, self.out_seq, self.pub_seq = None, 0, 0
        self.out_holding = None
        self.published = threading.Event()
        self.running = True
        self.n_pub = 0
        self.max_pub_gap = 0.0
        self.last_pub_wall = None
        self.work_cpu = 0.0
        self.work_max = 0.0
        self.cb_cpu = 0.0
        self.n_cb_pub = 0
        self.pub_call_max = 0.0     # самый долгий вызов publish (диагностика блокировки DDS)
        self.in_gap_max = 0.0       # самый большой промежуток между колбэками входа (диагностика простоя executor)
        self.last_in_wall = None
        self.n_slow_decode = 0

        # BEST_EFFORT-подписка совместима и с RELIABLE, и с BEST_EFFORT издателем Judge; выход RELIABLE — с любой подпиской
        sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        pub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(LowCmd, out_topic, pub_qos)
        self.create_subscription(LowCmd, in_topic, self._on_cmd_raw, sub_qos, raw=True)
        self.state_sub = self.create_subscription(LowState, state_topic, self._on_state, sub_qos)

        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(self.period, self._publish, clock=steady)
        self.create_timer(self.stats_period, self._log_stats, clock=steady)
        self.worker = threading.Thread(target=self._work_loop, name='safety_worker', daemon=True)
        self.worker.start()
        self.get_logger().info(
            f'Safety Block: {in_topic} → {out_topic}, публикация {rate:.0f} Гц (steady, отдельно от расчётов), '
            f'расчёт ≤ {self.budget_s * 1000:.1f} мс за порцию, тайм-аут входа {timeout} с sim, шаг ≤ {max_step_deg}°, '
            f'коллизии: 655 пар, порог {threshold * 1000:.2f} мм')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ executor: только приём и публикация
    def _on_state(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(N_MOTORS)], dtype=float)
        if not np.all(np.isfinite(q)):
            return
        with self.core_lock:
            self.core.latch_measured(q)
            start = self.core.out_q.copy()
        self.get_logger().info(f'Стартовая поза из /lowstate: {np.round(start, 3).tolist()}')
        self.destroy_subscription(self.state_sub)   # дальше /lowstate (500 Гц) не нужен
        self.state_sub = None
        self.published.set()                        # разбудить рабочий поток: можно готовить первую позу

    def _on_cmd_raw(self, raw):
        t = self._now()
        w = time.monotonic()
        if self.last_in_wall is not None:
            self.in_gap_max = max(self.in_gap_max, w - self.last_in_wall)
        self.last_in_wall = w
        with self.in_lock:
            self.n_in += 1
            if not self.immediate:
                self.in_raw, self.in_t = raw, t     # старый режим: разбирает рабочий поток, только последнее
                return
        c0 = time.thread_time()
        deadline = time.perf_counter() + self.budget_s
        fields = self._decode(raw)
        try:
            with self.core_lock:
                with self.out_lock:                 # сначала — поза рабочего потока, если ещё не ушла (порядок расчёта)
                    if self.out_bytes is not None and self.pub_seq != self.out_seq:
                        self._publish_locked()
                if fields is None:
                    self.core.stats.rejected_shape += 1
                    self._warn_rejected()
                    return
                if not self.core.on_command(fields, t, deadline):
                    self._warn_rejected()
                    return
                out = self.core.step(self._now(), deadline)
                if out is None:
                    return
                data = self.codec.encode(out.q, out.dq, out.kp, out.kd, out.tau)
                with self.out_lock:
                    self.out_bytes = data
                    self.out_seq += 1
                    self._publish_locked()
                    self.n_cb_pub += 1
                self._log_holding(out.holding)
        finally:
            self.cb_cpu += time.thread_time() - c0

    def _warn_rejected(self):
        self.get_logger().warn('Команда отклонена (NaN/Inf, форма или нет безопасного пути) — держу последнюю безопасную цель',
                               throttle_duration_sec=2.0)

    def _log_holding(self, holding):
        if holding != self.out_holding:
            self.out_holding = holding
            self.get_logger().info('Удержание последней принятой цели (нет входа)' if holding
                                   else 'Вход есть — отрабатываю команды')

    def _publish_locked(self):
        """Опубликовать текущую позу слота. Вызывать только под out_lock — так порядок публикаций = порядок расчёта."""
        p0 = time.perf_counter()
        self.pub.publish(self.out_bytes)            # готовые байты LowCmd, без вычислений
        self.pub_call_max = max(self.pub_call_max, time.perf_counter() - p0)
        self.pub_seq = self.out_seq
        self.n_pub += 1
        w = time.monotonic()
        if self.last_pub_wall is not None:
            self.max_pub_gap = max(self.max_pub_gap, w - self.last_pub_wall)
        self.last_pub_wall = w
        self.published.set()

    def _publish(self):
        """Таймер 200 Гц: ровный поток и удержание. Пропускает тик, если только что публиковали и новой позы нет."""
        with self.out_lock:
            if self.out_bytes is None:
                return
            if (self.pub_seq == self.out_seq and self.last_pub_wall is not None
                    and time.monotonic() - self.last_pub_wall < 0.8 * self.period):
                return
            self._publish_locked()

    # ------------------------------------------------------------------ рабочий поток: разбор входа, движение, коллизии
    def _decode(self, raw):
        fields = self.codec.decode(raw)
        if fields is None:
            self.n_slow_decode += 1
            try:
                fields = self.codec.decode_slow(raw)
            except Exception:  # noqa: BLE001 — битое сообщение = отклонить
                fields = None
        return fields

    def _work_loop(self):
        while self.running and rclpy.ok():
            self.published.wait(timeout=self.period)
            self.published.clear()
            c0 = time.thread_time()
            deadline = time.perf_counter() + self.budget_s
            with self.in_lock:
                raw, t_in = self.in_raw, self.in_t
                self.in_raw = None
            try:
                with self.core_lock:
                    if raw is not None:                 # только при immediate_publish:=false
                        fields = self._decode(raw)
                        ok = fields is not None and self.core.on_command(fields, t_in, deadline)
                        if fields is None:
                            self.core.stats.rejected_shape += 1
                        if not ok:
                            self._warn_rejected()
                    with self.out_lock:
                        ready = self.out_bytes is None or self.pub_seq == self.out_seq
                    if ready:
                        out = self.core.step(self._now(), deadline)
                        if out is not None:
                            data = self.codec.encode(out.q, out.dq, out.kp, out.kd, out.tau)
                            with self.out_lock:
                                # та же поза, что уже опубликована, — не новая: не удваиваем поток (таймер повторит сам)
                                if data != self.out_bytes:
                                    self.out_bytes = data
                                    self.out_seq += 1
                            self._log_holding(out.holding)
                    elif self.core.search is not None:
                        self.core.think(self._now(), deadline)   # предыдущая поза ещё не ушла — только поиск пути
            except Exception as e:  # noqa: BLE001 — рабочий поток не должен умирать: публикуется прежняя поза
                self.get_logger().error(f'Ошибка расчёта: {e!r}', throttle_duration_sec=2.0)
            dt = time.thread_time() - c0
            self.work_cpu += dt
            self.work_max = max(self.work_max, dt)

    def _log_stats(self):
        s = self.core.stats
        gap, self.max_pub_gap = self.max_pub_gap, 0.0
        self.get_logger().info(
            f'stats: in {self.n_in / self.stats_period:.0f} Гц, pub {self.n_pub / self.stats_period:.0f} Гц '
            f'(сразу из колбэка {self.n_cb_pub / self.stats_period:.0f} Гц, макс. пауза {gap * 1000:.0f} мс wall, '
            f'макс. publish {self.pub_call_max * 1000:.1f} мс, макс. промежуток входа {self.in_gap_max * 1000:.0f} мс), '
            f'колбэк {100 * self.cb_cpu / self.stats_period:.1f}% ядра, расчёт {100 * self.work_cpu / self.stats_period:.1f}% ядра '
            f'(макс порция {self.work_max * 1000:.2f} мс), проверок {self.checker.evals}, принято {s.accepted}, '
            f'NaN/Inf {s.rejected_nonfinite}, клип q {s.clipped_q}, отказ-коллизия {s.rejected_collision}, '
            f'отходов {s.retreats}, упоры {s.blocked}, медл. разбор {self.n_slow_decode}')
        self.n_in = self.n_pub = self.n_cb_pub = 0
        self.pub_call_max = self.in_gap_max = 0.0
        self.work_cpu = self.work_max = self.cb_cpu = 0.0
        self.checker.evals = 0

    def destroy_node(self):
        self.running = False
        self.published.set()
        self.worker.join(timeout=1.0)
        return super().destroy_node()


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
