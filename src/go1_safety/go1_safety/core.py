"""Логика Safety Block T2 без ROS (тестируется pytest). Обоснование чисел — docs/T2_DESIGN.md.

Порядок моторов Unitree: 0..2 FR (hip, thigh, calf), 3..5 FL, 6..8 RR, 9..11 RL; в LowCmd 20 слотов, используются 0..11.
"""
import math
from dataclasses import dataclass

import numpy as np

from .collision import BLOCKED, CollisionChecker  # noqa: F401  (CollisionChecker — для аннотаций/импорта извне)

N_MOTORS = 12
N_SLOTS = 20
MODE_SERVO = 0x0A

# ТЗ_T2, таблица 4.3 (MuJoCo Menagerie go1, rev 0059d43): hip, thigh, calf
Q_MIN_TZ = np.array([-0.863, -0.686, -2.818] * 4)
Q_MAX_TZ = np.array([0.863, 4.501, -0.888] * 4)
Q_MARGIN = 0.01                                   # внутренний запас, рад
Q_MIN = Q_MIN_TZ + Q_MARGIN
Q_MAX = Q_MAX_TZ - Q_MARGIN

MAX_STEP_RAD = math.radians(4.5)                  # ТЗ: ≤ 5° между соседними командами
TORQUE_LIMIT = np.array([23.7, 23.7, 35.55] * 4)  # пределы моментов симулятора (core.TORQUE_LIMIT)
KP_MAX = 100.0
KD_MAX = 10.0
DQ_MAX = 20.0
HOLD_KP = 40.0                                    # как удержание в симуляторе
HOLD_KD = 2.0
HOLD_KP_MIN = 5.0                                 # «удерживать» с меньшим kp нельзя — берём HOLD_KP/HOLD_KD
INPUT_TIMEOUT = 0.3                               # с sim time (ТЗ: ≤ 0.5 с)
HOME_Q = np.array([0.0, 0.9, -1.8] * 4)           # keyframe 'home' go1.xml


@dataclass
class Output:
    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray
    holding: bool


@dataclass
class Stats:
    accepted: int = 0
    rejected_nonfinite: int = 0
    rejected_shape: int = 0
    clipped_q: int = 0
    blocked: int = 0


def all_finite(fields):
    """True, если во ВСЕХ управляющих полях всех 20 моторов нет NaN/±Inf."""
    for name in ('q', 'dq', 'kp', 'kd', 'tau'):
        if not np.all(np.isfinite(np.asarray(fields[name], dtype=float))):
            return False
    return True


class SafetyCore:
    """Состояние и решения Safety Block. Время `now` — sim time в секундах."""

    def __init__(self, checker, max_step=MAX_STEP_RAD, input_timeout=INPUT_TIMEOUT):
        self.checker = checker
        self.max_step = float(max_step)
        self.input_timeout = float(input_timeout)
        self.out_q = None            # последняя опубликованная цель (12)
        self.goal = None             # последняя принятая (клипованная) цель
        self.cmd = None              # последние принятые dq/kp/kd/tau (клипованные), dict
        self.last_input_t = None
        self.blocked = False         # путь к goal упёрся в коллизию — стоим до новой цели
        self.stats = Stats()

    # ------------------------------------------------------------------ старт
    def _safe_start(self, q):
        q = np.clip(np.asarray(q, dtype=float), Q_MIN, Q_MAX)
        if self.checker.min_dist(q) >= -self.checker.stop_threshold:
            return q
        return np.clip(HOME_Q, Q_MIN, Q_MAX)

    def latch_measured(self, q_measured):
        """Первый /lowstate: удерживаем измеренную позу (один раз, без слежения — иначе поза «оседает»)."""
        if self.out_q is None:
            self.out_q = self._safe_start(q_measured)
            self.goal = self.out_q.copy()

    # ------------------------------------------------------------------ вход
    def on_command(self, fields, now):
        """fields: dict массивов длины 20 (q, dq, kp, kd, tau). Возвращает True, если команда принята."""
        try:
            arr = {k: np.asarray(fields[k], dtype=float) for k in ('q', 'dq', 'kp', 'kd', 'tau')}
        except (KeyError, TypeError, ValueError):
            self.stats.rejected_shape += 1
            return False
        if any(a.shape != (N_SLOTS,) for a in arr.values()):
            self.stats.rejected_shape += 1
            return False
        if not all_finite(arr):
            self.stats.rejected_nonfinite += 1
            return False
        q_in = arr['q'][:N_MOTORS]
        goal = np.clip(q_in, Q_MIN, Q_MAX)
        if np.any(goal != q_in):
            self.stats.clipped_q += 1
        self.cmd = {
            'dq': np.clip(arr['dq'][:N_MOTORS], -DQ_MAX, DQ_MAX),
            'kp': np.clip(arr['kp'][:N_MOTORS], 0.0, KP_MAX),
            'kd': np.clip(arr['kd'][:N_MOTORS], 0.0, KD_MAX),
            'tau': np.clip(arr['tau'][:N_MOTORS], -TORQUE_LIMIT, TORQUE_LIMIT),
        }
        if self.out_q is None:                       # команда раньше /lowstate: стартуем из home
            self.out_q = self._safe_start(HOME_Q)
            self.goal = self.out_q.copy()
        if self.goal is None or not np.array_equal(goal, self.goal):
            self.goal = goal
            self.blocked = False
        self.last_input_t = float(now)
        self.stats.accepted += 1
        return True

    # ------------------------------------------------------------------ такт
    def holding(self, now):
        return self.cmd is None or self.last_input_t is None or (float(now) - self.last_input_t) > self.input_timeout

    def _hold_gains(self):
        if self.cmd is not None:
            kp, kd = self.cmd['kp'].copy(), self.cmd['kd'].copy()
            weak = kp < HOLD_KP_MIN
            kp[weak] = HOLD_KP
            kd[weak] = HOLD_KD
            return kp, kd
        return np.full(N_MOTORS, HOLD_KP), np.full(N_MOTORS, HOLD_KD)

    def step(self, now):
        """Следующая выходная команда (или None, если безопасная поза ещё неизвестна)."""
        if self.out_q is None:
            return None
        at_goal = True
        if not self.blocked and not np.array_equal(self.out_q, self.goal):
            delta = self.goal - self.out_q
            span = float(np.max(np.abs(delta)))
            frac = min(1.0, self.max_step / span)
            cand = self.goal.copy() if frac >= 1.0 else self.out_q + delta * frac
            q_new, status = self.checker.advance(self.out_q, cand)
            if status == BLOCKED:
                self.blocked = True
                self.stats.blocked += 1
            # страховка от численных хвостов: шаг строго в пределах и не больше max_step
            q_new = np.clip(q_new, Q_MIN, Q_MAX)
            q_new = self.out_q + np.clip(q_new - self.out_q, -self.max_step, self.max_step)
            self.out_q = q_new
            at_goal = np.array_equal(self.out_q, self.goal)
        hold = self.holding(now)
        if hold:
            kp, kd = self._hold_gains()
            dq = np.zeros(N_MOTORS)
            tau = np.zeros(N_MOTORS)
        else:
            kp, kd = self.cmd['kp'].copy(), self.cmd['kd'].copy()
            tau = self.cmd['tau'].copy()
            # целевая скорость имеет смысл только у цели; во время рампы/остановки у препятствия — 0
            dq = self.cmd['dq'].copy() if at_goal else np.zeros(N_MOTORS)
        return Output(q=self.out_q.copy(), dq=dq, kp=kp, kd=kd, tau=tau, holding=hold)
