"""Логика Safety Block T2 без ROS (тестируется pytest). Обоснование чисел — docs/T2_DESIGN.md.

Порядок моторов Unitree: 0..2 FR (hip, thigh, calf), 3..5 FL, 6..8 RR, 9..11 RL; в LowCmd 20 слотов, используются 0..11.
"""
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .collision import BLOCKED, DONE, CollisionChecker  # noqa: F401  (CollisionChecker — для импорта извне)

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
# Допустимые значения LowCmd для 12 приводов — официальное уточнение организаторов к ТЗ_T2:
#   kp 0..1000, kd 0..100, tau −100..+100, dq −20..+20. Это фильтр некорректных команд, а не физические пределы:
#   момент симулятор ограничивает сам (hip/thigh ±23.7, calf ±35.55 Н·м). Вне диапазона — клиппинг.
KP_MAX = 1000.0
KD_MAX = 100.0
TAU_MAX = 100.0
DQ_MAX = 20.0
HOLD_KP = 40.0                                    # как удержание в симуляторе
HOLD_KD = 2.0
HOLD_KP_MIN = 5.0                                 # «удерживать» с меньшим kp нельзя — берём HOLD_KP/HOLD_KD
INPUT_TIMEOUT = 0.3                               # с sim time (ТЗ: ≤ 0.5 с)
HOME_Q = np.array([0.0, 0.9, -1.8] * 4)           # keyframe 'home' go1.xml
HISTORY_LEN = 4000                                # поз в истории (сжатой: отрезки ≤ шага, проверенные)
# Бюджеты по ВРЕМЕНИ (не числу вычислений): на медленном CPU порция не растягивается, публикация не ждёт.
SEARCH_BUDGET_NOW_S = 0.0012                      # с: сразу при новой цели
SEARCH_BUDGET_TICK_S = 0.0012                     # с: на каждой следующей порции, пока поиск пути не закончен
SEARCH_BUDGET_TOTAL = 20000                       # вычислений всего на одну цель (≈ 0.4–0.6 с CPU), дальше — отказ
EVAL_CHUNK = 8                                    # вычислений между проверками часов
RETREAT_CANDIDATES = 40                           # сколько точек истории пробуем как начало обходного пути
SNAP_EPS = 1e-9                                   # рад: «дошли до цели» — присваиваем цель ровно
REPLAN_PERIOD = 0.2                               # с sim: та же отклонённая цель перепроверяется не чаще


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
    rejected_collision: int = 0      # цель в коллизии или к ней нет безопасного пути
    retreats: int = 0                # путь к безопасной цели перекрыт — отход назад по истории поз
    blocked: int = 0                 # упор на такте (при проверке всего пути заранее быть не должно)
    path_evals_max: int = 0          # макс. вычислений коллизий на одну новую цель
    search_ticks_max: int = 0        # макс. тактов, за которые решился поиск пути


def all_finite(fields):
    """True, если во ВСЕХ управляющих полях всех 20 моторов нет NaN/±Inf."""
    for name in ('q', 'dq', 'kp', 'kd', 'tau'):
        if not np.all(np.isfinite(np.asarray(fields[name], dtype=float))):
            return False
    return True


class PathSearch:
    """Пошаговый поиск безопасного пути к цели (бюджет вычислений за вызов — такт не затягивается).

    Кандидаты на начало прямого пути к цели: текущая выходная поза, затем точки истории от новых к старым.
    Отрезки истории уже были проверены при публикации, поэтому отход назад по ним безопасен.
    """

    def __init__(self, checker, goal, starts):
        self.checker = checker
        self.goal = goal
        self.starts = starts          # [(k, q)]: k = -1 — текущая поза, иначе индекс в снимке истории
        self.i = 0
        self.pos = None
        self.evals = 0
        self.ticks = 0

    def run(self, deadline):
        """Работает до deadline (time.perf_counter). ('found', k) | ('fail', None) | ('pending', None)."""
        ch = self.checker
        e0 = ch.evals
        self.ticks += 1
        try:
            while self.i < len(self.starts):
                if self.pos is None:
                    self.pos = self.starts[self.i][1].copy()
                while True:
                    if time.perf_counter() >= deadline:
                        return 'pending', None
                    q, st = ch.advance(self.pos, self.goal, max_evals=EVAL_CHUNK)
                    self.pos = q
                    if st == DONE:
                        return 'found', self.starts[self.i][0]
                    if st == BLOCKED:
                        break
                self.i += 1
                self.pos = None
            return 'fail', None
        finally:
            self.evals += ch.evals - e0


class SafetyCore:
    """Состояние и решения Safety Block. Время `now` — sim time в секундах.

    Новая цель принимается, только если к ней есть доказуемо безопасный путь:
      1) цель сама без коллизий (проникновение ≤ stop_threshold), иначе — отказ сразу (вариант 1 ТЗ);
      2) прямой путь от текущей выходной позы до цели проверяется ЦЕЛИКОМ; свободен — исполняем рампой ≤ max_step;
      3) перекрыт — страховка: ищем точку истории опубликованных поз, из которой прямой путь к цели свободен,
         отходим к ней назад по истории (каждый отрезок уже проверен) и идём к цели;
      4) не нашли — отказ: держим последнюю принятую цель.
    Поиск пошаговый, бюджет по времени: сразу SEARCH_BUDGET_NOW_S, затем по SEARCH_BUDGET_TICK_S за порцию; пока он идёт,
    выход стоит на месте и публикуется без пауз. Так блок не застревает у препятствия и не затягивает такт.
    """

    def __init__(self, checker, max_step=MAX_STEP_RAD, input_timeout=INPUT_TIMEOUT, history_len=HISTORY_LEN,
                 budget_now_s=SEARCH_BUDGET_NOW_S, budget_tick_s=SEARCH_BUDGET_TICK_S, budget_total=SEARCH_BUDGET_TOTAL):
        self.checker = checker
        self.max_step = float(max_step)
        self.input_timeout = float(input_timeout)
        self.budget_now_s, self.budget_tick_s = float(budget_now_s), float(budget_tick_s)
        self.budget_total = int(budget_total)
        self.out_q = None            # последняя опубликованная цель (12)
        self.goal = None             # последняя принятая (клипованная, достижимая) цель
        self.cmd = None              # последние принятые dq/kp/kd/tau (клипованные), dict
        self.last_input_t = None
        self._awaiting_safe_command = True
        self.blocked = False         # упор на такте (страховка; при проверке пути заранее не возникает)
        self.waypoints = deque()     # отход назад по истории перед движением к goal
        self.history = deque(maxlen=int(history_len))
        self.search = None           # PathSearch в процессе
        self.search_cmd = None       # коэффициенты ожидающей команды
        self.search_hist = None      # снимок истории для поиска
        self._reject_goal = None     # последняя отклонённая цель и время: повтор перепроверяем не чаще REPLAN_PERIOD
        self._reject_t = -1e9
        self.stats = Stats()

    # ------------------------------------------------------------------ старт
    def _safe_start(self, q):
        q = np.clip(np.asarray(q, dtype=float), Q_MIN, Q_MAX)
        if self.checker.min_dist(q) >= -self.checker.stop_threshold:
            return q
        return np.clip(HOME_Q, Q_MIN, Q_MAX)

    def _init_pose(self, q):
        self.out_q = self._safe_start(q)
        self.goal = self.out_q.copy()
        self.history.clear()
        self.history.append(self.out_q.copy())

    def latch_measured(self, q_measured):
        """Первый /lowstate: удерживаем измеренную позу (один раз, без слежения — иначе поза «оседает»)."""
        if self.out_q is None:
            self._init_pose(q_measured)

    # ------------------------------------------------------------------ история опубликованных поз
    def _remember(self, q, deadline=None):
        """Сжатая история: соседние записи соединены проверенным прямым отрезком длиной ≤ max_step.
        Нет времени на проверку сжатия — просто добавляем (это всегда безопасно)."""
        h = self.history
        if h and np.array_equal(h[-1], q):
            return
        if len(h) >= 2 and float(np.max(np.abs(q - h[-2]))) <= self.max_step \
                and (deadline is None or time.perf_counter() < deadline) \
                and self.checker.path_clear(h[-2], q, budget=20) is True:
            h[-1] = q.copy()
        else:
            h.append(q.copy())

    # ------------------------------------------------------------------ поиск пути
    def _start_search(self, goal, cmd, now, deadline):
        hist = list(self.history)
        n = len(hist)
        starts = [(-1, self.out_q.copy())]
        if n >= 2:
            stride = max(1, n // RETREAT_CANDIDATES)
            idx = list(range(n - 2, -1, -stride))
            if 0 not in idx:
                idx.append(0)
            starts += [(k, hist[k]) for k in idx]
        self.search = PathSearch(self.checker, goal, starts)
        self.search_cmd = cmd
        self.search_hist = hist
        return self._continue_search(deadline, now)

    def _continue_search(self, deadline, now):
        s = self.search
        res, k = s.run(deadline)
        if res == 'pending' and s.evals < self.budget_total:
            return 'pending'
        self.stats.path_evals_max = max(self.stats.path_evals_max, s.evals)
        self.stats.search_ticks_max = max(self.stats.search_ticks_max, s.ticks)
        goal, cmd, hist = s.goal, self.search_cmd, self.search_hist
        self.search = self.search_cmd = self.search_hist = None
        if res == 'found':
            self.goal = goal
            self.cmd = cmd
            self._awaiting_safe_command = False
            self.blocked = False
            self.waypoints = deque()
            if k >= 0:                               # отход: hist[n-2], ..., hist[k] (текущая поза = hist[n-1])
                self.waypoints = deque(hist[j].copy() for j in range(len(hist) - 2, k - 1, -1))
                self.stats.retreats += 1
            self._reject_goal = None
            return 'found'
        self._reject(goal, now)
        return 'fail'

    def _reject(self, goal, now):
        self._reject_goal, self._reject_t = goal.tobytes(), float(now)
        self.stats.rejected_collision += 1

    # ------------------------------------------------------------------ вход
    def on_command(self, fields, now, deadline=None):
        """fields: dict массивов длины 20 (q, dq, kp, kd, tau).
        Возвращает True — команда принята (или ждёт окончания поиска пути), False — отклонена."""
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
        cmd = {
            'dq': np.clip(arr['dq'][:N_MOTORS], -DQ_MAX, DQ_MAX),
            'kp': np.clip(arr['kp'][:N_MOTORS], 0.0, KP_MAX),
            'kd': np.clip(arr['kd'][:N_MOTORS], 0.0, KD_MAX),
            'tau': np.clip(arr['tau'][:N_MOTORS], -TAU_MAX, TAU_MAX),
        }
        if self.out_q is None:                       # команда раньше /lowstate: стартуем из home
            self._init_pose(HOME_Q)
        now = float(now)
        self._expire_input(now)
        if np.array_equal(goal, self.goal) and self.search is None:
            self.cmd = cmd                           # та же цель — обновляем коэффициенты
            self._awaiting_safe_command = False
        elif self.search is not None and np.array_equal(goal, self.search.goal):
            self.search_cmd = cmd                    # та же цель ещё в поиске
        else:
            if goal.tobytes() == self._reject_goal and now - self._reject_t < REPLAN_PERIOD:
                self.stats.rejected_collision += 1
                return False
            self.search = None                       # новая цель отменяет незаконченный поиск
            if self.checker.min_dist(goal) < -self.checker.stop_threshold:
                self._reject(goal, now)              # сама цель в коллизии — отказ целиком
                return False
            if deadline is None:
                deadline = time.perf_counter() + self.budget_now_s
            if self._start_search(goal, cmd, now, deadline) == 'fail':
                return False
        self.last_input_t = now
        self.stats.accepted += 1
        return True

    # ------------------------------------------------------------------ такт
    def _expire_input(self, now):
        if self.last_input_t is None or now < self.last_input_t + self.input_timeout:
            return
        # Freeze the last checked output, not the requested goal: that goal
        # may still need a slew or a retreat through multiple waypoints.
        self.last_input_t = None
        self._awaiting_safe_command = True
        self.goal = self.out_q
        self.waypoints.clear()
        self.search = self.search_cmd = self.search_hist = None
        self.blocked = False

    def holding(self, now):
        return (self._awaiting_safe_command or self.last_input_t is None
                or float(now) >= self.last_input_t + self.input_timeout)

    def _hold_gains(self):
        if self.cmd is not None:
            kp, kd = self.cmd['kp'].copy(), self.cmd['kd'].copy()
            weak = kp < HOLD_KP_MIN
            kp[weak] = HOLD_KP
            kd[weak] = HOLD_KD
            return kp, kd
        return np.full(N_MOTORS, HOLD_KP), np.full(N_MOTORS, HOLD_KD)

    def _move_toward(self, target, deadline):
        """Шаг ≤ max_step по прямой к target с проверкой отрезка (порциями до deadline); True — дошли."""
        delta = target - self.out_q
        span = float(np.max(np.abs(delta)))
        if span == 0.0:
            return True
        frac = min(1.0, self.max_step / span)
        cand = target.copy() if frac >= 1.0 else self.out_q + delta * frac
        q = self.out_q
        while True:
            q, status = self.checker.advance(q, cand, max_evals=EVAL_CHUNK)
            if status != 'partial' or time.perf_counter() >= deadline:
                break                                  # не успели — публикуем уже проверенную часть отрезка
        if status == BLOCKED:
            self.blocked = True
            self.waypoints.clear()
            self.stats.blocked += 1
        # страховка от численных хвостов: шаг строго в пределах и не больше max_step
        q_new = np.clip(q, Q_MIN, Q_MAX)
        q_new = self.out_q + np.clip(q_new - self.out_q, -self.max_step, self.max_step)
        # a + (b − a) в плавающей точке не всегда даёт ровно b (~1e-17 рад): дошли — ставим ровно цель, иначе
        # at_goal=false и на выходе dq=0 вместо заданного. Цель (или точка истории) уже проверена; 1e-9 рад ≪ запаса шага.
        if float(np.max(np.abs(q_new - target))) < SNAP_EPS:
            q_new = target.copy()
        self.out_q = q_new
        self._remember(self.out_q, deadline)
        return np.array_equal(self.out_q, target)

    def think(self, now, deadline):
        """Только продвинуть незаконченный поиск пути (без движения выхода) — когда предыдущая поза ещё не опубликована."""
        self._expire_input(now)
        if self.search is not None:
            self._continue_search(deadline, now)

    def step(self, now, deadline=None):
        """Следующая выходная команда (или None, если безопасная поза ещё неизвестна).
        deadline (time.perf_counter) ограничивает вычисления этой порции."""
        if self.out_q is None:
            return None
        if deadline is None:
            deadline = time.perf_counter() + self.budget_tick_s
        self.think(now, deadline)
        if self.last_input_t is not None and self.search is None and not self.blocked:
            if self.waypoints:
                if self._move_toward(self.waypoints[0], deadline):
                    self.waypoints.popleft()
            elif not np.array_equal(self.out_q, self.goal):
                self._move_toward(self.goal, deadline)
        at_goal = self.search is None and not self.waypoints and np.array_equal(self.out_q, self.goal)
        hold = self.holding(now)
        if hold:
            kp, kd = self._hold_gains()
            dq = np.zeros(N_MOTORS)
            tau = np.zeros(N_MOTORS)
        else:
            kp, kd = self.cmd['kp'].copy(), self.cmd['kd'].copy()
            tau = self.cmd['tau'].copy()
            # целевая скорость имеет смысл только у цели; во время рампы/отхода/поиска — 0
            dq = self.cmd['dq'].copy() if at_goal else np.zeros(N_MOTORS)
        return Output(q=self.out_q.copy(), dq=dq, kp=kp, kd=kd, tau=tau, holding=hold)
