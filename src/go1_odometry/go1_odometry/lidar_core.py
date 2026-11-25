"""T4, режим lidar: ядро одометрии Go1 по IMU, лидару и ногам (без ROS, numpy).

Разработано командой «Кафедра проблем», BSD-3-Clause (см. ниже).

Правки при переносе (алгоритм не менялся):
  1. этот заголовок (источник, лицензия, список правок); исходная строка документации — ниже, дословно;
  2. больше ничего: проверка входов на NaN/Inf, раскладка суставов по именам, z, twist и потоки — в lidar_mode.py
     и node.py этого пакета. on_contact в нашем узле не вызывается (/go1_contacts не используем) — опора всегда
     по высоте стоп (CONTACT_Z), см. on_joints.

Лицензия этого файла — BSD-3-Clause, как у пакета-источника (тег <license> в его package.xml; строки copyright и
файла LICENSE в источнике нет): разрешено распространение и изменение с сохранением этого уведомления; без гарантий.
Остальной пакет go1_odometry — MIT.

Исходная документация модуля:
Одометрия Go1 без ROS.

Ориентация (курс, крен, тангаж) — из IMU. Перенос в плоскости — сопоставление скана лидара с опорным сканом
по расстоянию «точка–прямая»; неизвестен только сдвиг (2 числа), поворот точек берётся из IMU. Между сканами
и вдоль вырожденных направлений скана — прогноз: перемещение по кинематике опорных ног или последняя скорость.
Стоянка (ориентация IMU не меняется за 0,3 с) замораживает выход. Время — штампы сообщений; штамп назад — сброс.
"""
from collections import deque

import numpy as np

# --- параметры ---
LIDAR_XYZ = np.array([0.22, 0.0, 0.11])      # base_scan относительно trunk (go1.urdf.xacro)
HIP_X, HIP_Y = 0.1881, 0.04675               # leg_offset_x/y (const.xacro)
THIGH_Y, THIGH_L, CALF_L = 0.08, 0.213, 0.213  # thigh_offset, thigh_length, calf_length (const.xacro)
LEGS = ('FL', 'FR', 'RL', 'RR')
JOINTS = tuple(f'{g}_{j}_joint' for g in LEGS for j in ('hip', 'thigh', 'calf'))  # порядок q, dq в on_joints
FLOOR_Z = -0.20         # точка ниже корпуса больше чем на 0,20 м — пол (корпус на высоте ≈0,29 м)
MIN_RANGE, MAX_RANGE = 0.15, 10.0  # дальше лучи под острым углом к стенам дают ложные пары; 10 м при тангаже 6° — подъём 1 м
NORMAL_R = 0.2          # нормаль ячейки — по сырым точкам её и соседних ячеек отрезка в пределах этого пути вдоль скана, м
NORMAL_BASE = 0.05      # ... если окрестность вдоль поверхности длиннее этого (≫ шума дальности 1 см), м
NORMAL_MIN = 5          # ... и в ней не меньше стольких точек
NORMAL_FLAT = 0.1       # отношение собственных чисел ковариации окрестности: больше — угол или шум, нормали нет
PAIR_DIST = 0.3         # наибольшее расстояние до ближайшей точки опорного скана, м (сдвиг за скан ≤ 3 см)
ROBUST = 0.03           # масштаб весов Коши по невязке, м (шум лидара σ = 1 см)
CELL = 0.08             # точки скана усредняются по ячейкам такой длины дуги, м: у близкой стены шаг лучей — миллиметры,
                        # и нормали по соседним точкам задаёт шум дальности (1 см), а не стена
CELL_JUMP = 0.1         # скачок дальности больше этого начинает новую ячейку (край предмета), м
ITERS = 8               # итерации сопоставления (сходится за 2–4 при хорошем прогнозе)
DEGEN = 0.05            # доля информации по слабой оси (λ / Σw): меньше — ось не наблюдаема, берём прогноз
MIN_PAIRS = 30          # меньше пар — скан не используется
KEY_DIST = 0.5          # смена опорного скана при удалении от него, м
KEY_OVERLAP = 0.3       # ... или при доле точек скана с невязкой < 9 см ниже этой
STILL_WIN = 0.3         # окно признака стоянки, с
STILL_ATT = np.radians(0.5)  # размах крена, тангажа и курса в окне меньше — стоянка: стоя ориентация почти не меняется
                        # (≤ 0,2°, и при мелкой дрожи корпуса, когда гироскоп и акселерометр шумят), на ходу рысь качает ≥ 0,75°
BLEND_TAU = 0.2         # постоянная времени, с которой поправка скана и выход из стоянки входят в выход, с
HIST = 1.0              # история IMU для ориентации на момент штампа скана (с запасом на паузу IMU), с
PENDING = 40            # сканов в очереди до прихода IMU (2 с при 20 Гц): пауза IMU не теряет сканы
PENDING_AGE = 0.5       # после паузы IMU разбираются сканы не старше этого (если есть новее), с
MHIST = 2.0             # история пути по ногам: прогноз скана — на его штамп, с
MAX_DT = 0.045          # через больший разрыв в суставах путь по ногам не считается (стопа могла успеть шагнуть); 45 мс — один
                        # пропуск при 50 Гц (40 мс) ещё считается: фаза переноса рыси длиннее, с
JS_STALE = 0.1          # суставов нет дольше этого — путь по ногам не ведётся: прогноз и выход — по последней скорости, с
VEL_MAX_HOLD = 2.0      # ... но не дольше этого (дальше — стоим), с
VEL_HOLD = 0.3          # без сканов дольше этого выход ведётся по ногам (скорость сканов не экстраполируется), с
CONTACT_HOLD = 0.5      # признак контакта старше этого — опора по кинематике: стопы не выше нижней на CONTACT_Z, с
CONTACT_Z = 0.02        # м


def rot_q(q):
    """Матрица поворота из кватерниона (x, y, z, w)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def foot_pos(q):
    """Положения стоп в осях корпуса, q — (4, 3): hip, thigh, calf для FL, FR, RL, RR (оси x, y, y по URDF)."""
    sx = np.array([1, 1, -1, -1])
    sy = np.array([1, -1, 1, -1])
    q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
    x = -THIGH_L * np.sin(q2) - CALF_L * np.sin(q2 + q3)
    z = -THIGH_L * np.cos(q2) - CALF_L * np.cos(q2 + q3)
    y0 = sy * THIGH_Y
    return np.stack([sx * HIP_X + x, sy * HIP_Y + np.cos(q1) * y0 - np.sin(q1) * z,
                     np.sin(q1) * y0 + np.cos(q1) * z], 1)


def cells(p, r, idx, dth, cell=CELL):
    """Точки скана по порядку лучей (p — xy, r — дальности, idx — номера лучей) → средние по ячейкам длиной cell вдоль
    дуги (сумма r·Δθ) и нормали ячеек. Новую ячейку и новый отрезок начинают скачок дальности > CELL_JUMP и пропуск лучей.
    Нормаль ячейки — главная ось ковариации СЫРЫХ точек ячеек того же отрезка в пределах NORMAL_R пути (моменты по
    ячейкам, суммы окон — по накопленным суммам); nan, если окрестность короткая, редкая или неплоская (угол, шум)."""
    if len(p) < 2:
        return p, np.full_like(p, np.nan)
    brk = np.r_[False, (np.abs(np.diff(r)) > CELL_JUMP) | (np.diff(idx) > 1)]
    seg = np.cumsum(brk)
    step = np.maximum(r * dth, np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1)])   # вдоль стены под острым
    a = np.cumsum(step) / cell + seg                               # углом шаг лучей длиннее дуги — ячейка по пути
    start = np.flatnonzero(np.r_[True, np.diff(np.floor(a)) > 0])
    n = np.diff(np.r_[start, len(p)]).astype(float)
    s1 = np.add.reduceat(p, start, axis=0)
    s2 = np.add.reduceat(np.stack([p[:, 0] ** 2, p[:, 0] * p[:, 1], p[:, 1] ** 2], 1), start, axis=0)
    m = s1 / n[:, None]
    cs = seg[start]
    gap = np.r_[0.0, np.linalg.norm(np.diff(m, axis=0), axis=1) * (np.diff(cs) == 0)]
    s = np.cumsum(gap) + cs * 1e6                                 # путь вдоль отрезка; отрезки разнесены
    lo = np.searchsorted(s, s - NORMAL_R, 'left')
    hi = np.searchsorted(s, s + NORMAL_R, 'right')
    acc = lambda x: np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)])
    N, S1, S2 = acc(n), acc(s1), acc(s2)
    k = N[hi] - N[lo]
    mu = (S1[hi] - S1[lo]) / k[:, None]
    cxx, cxy, cyy = ((S2[hi] - S2[lo]) / k[:, None] - np.stack([mu[:, 0] ** 2, mu[:, 0] * mu[:, 1], mu[:, 1] ** 2], 1)).T
    h, d = (cxx + cyy) / 2, np.hypot((cxx - cyy) / 2, cxy)
    lmax, lmin = h + d, np.maximum(h - d, 0.0)
    ok = (k >= NORMAL_MIN) & (np.sqrt(12 * np.maximum(lmax, 0)) >= NORMAL_BASE) & (lmin < NORMAL_FLAT * lmax)
    th = 0.5 * np.arctan2(2 * cxy, cxx - cyy)                      # направление поверхности
    nrm = np.stack([-np.sin(th), np.cos(th)], 1)
    nrm[~ok] = np.nan
    return m, nrm


class Odometry:
    def __init__(self, use_scan=True, use_legs=True, leg_scale=1.0):
        """leg_scale — множитель пути по ногам (калибровка: при поднятом корпусе ноги занижают путь)."""
        self.use_scan, self.use_legs, self.leg_scale = use_scan, use_legs, float(leg_scale)
        self.reset()

    def reset(self):
        self.last = {}                 # последний штамп по каждому потоку
        self.R = None                  # ориентация корпуса в мире (из IMU)
        self.t_imu, self.q_imu = None, None
        self.hist = deque()            # (t, кватернион) IMU за HIST — ориентация на момент штампа скана
        self.pending = deque(maxlen=PENDING)  # сканы, ждущие IMU со штампом не раньше своего (по порядку)
        self.t_contact = -np.inf       # штамп последнего признака контакта
        self.win = deque()             # (t, крен/тангаж/курс) за STILL_WIN
        self.still = False
        self.contact = np.ones(4, bool)
        self.f_prev, self.st_prev = None, None  # стопы и опора на прошлом сообщении суставов
        self.m = np.zeros(2)           # накопленное перемещение по ногам
        self.mhist = deque()           # (t, m) — путь по ногам на штампы суставов
        self.c, self.t_c, self.m_c = np.zeros(2), None, np.zeros(2)  # позиция по последнему скану, её штамп, m тогда
        self.v = np.zeros(2)           # скорость по сканам
        self.key = None                # опорный скан: точки в мире, нормали, позиция
        self.degen = False
        self.n_match = self.n_degen = self.n_skip = self.n_key = self.n_out = self.n_legs_out = 0   # для диагностики
        self.out, self.frozen, self.off, self.t_off, self.t_pose = None, None, np.zeros(2), 0.0, -np.inf
        self.mode = None               # источник выхода на прошлом pose: True — ноги, False — скорость сканов

    def _fresh(self, name, t):
        """Порядок штампов в потоке: назад — сброс, повтор — пропуск."""
        if name in self.last and t < self.last[name]:
            self.reset()
        elif self.last.get(name) == t:
            return False
        self.last[name] = t
        return True

    def _rot(self, t):
        """Ориентация в момент t: интерполяция между соседними IMU (за пределами истории — крайняя)."""
        h = self.hist
        k = next((i for i in range(len(h)) if h[i][0] >= t), None)
        if k is None:
            return rot_q(h[-1][1])
        if k == 0:
            return rot_q(h[0][1])
        (t0, q0), (t1, q1) = h[k - 1], h[k]
        q1 = q1 if np.dot(q0, q1) >= 0 else -q1
        return rot_q(q0 + (q1 - q0) * (t - t0) / (t1 - t0))

    def on_imu(self, t, quat, gyro=None, acc=None):
        """Ориентация (x, y, z, w). Гироскоп и ускорение не нужны: в симуляторе ориентация точная, а при мелкой
        дрожи корпуса стоя гироскоп и ускорение шумят сильнее, чем на ходу."""
        if not self._fresh('imu', t):
            return
        q = np.asarray(quat, float)
        self.R, self.t_imu, self.q_imu = rot_q(q), t, q
        if self.t_c is None:                 # отсчёт «давно ли был скан» — с первой IMU
            self.t_c = t
        self.hist.append((t, q))
        while self.pending and t >= self.pending[0][0]:      # очередь — до обрезки истории: нужна IMU до паузы
            s = self.pending.popleft()
            if not (self.pending and self.pending[0][0] <= t and s[0] < t - PENDING_AGE):  # после долгой паузы —
                self._scan(*s)                                                      # только свежие сканы
        while self.hist[0][0] < t - HIST:
            self.hist.popleft()
        R = self.R
        att = np.array([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])])
        self.win.append((t, att))
        while self.win[0][0] <= t - STILL_WIN:
            self.win.popleft()
        d = np.array([w[1] for w in self.win]) - att
        d = (d + np.pi) % (2 * np.pi) - np.pi
        self.still = bool(t - self.win[0][0] >= STILL_WIN * 0.8 and np.ptp(d, 0).max() < STILL_ATT)

    def on_contact(self, t, leg, on):
        self.contact[leg], self.t_contact = bool(on), t

    def on_joints(self, t, q, dq=None):
        """Перемещение корпуса по опорным стопам: стопа в контакте неподвижна в мире, поэтому сдвиг корпуса —
        минус сдвиг R·p_стопы между соседними сообщениями. dq не используется: он шумнее разности q."""
        prev = self.last.get('js')
        if not self._fresh('js', t) or self.R is None:
            return
        f = self.R @ foot_pos(np.asarray(q, float).reshape(4, 3)).T          # стопы в мировых осях, (3, 4)
        if t - self.t_contact <= CONTACT_HOLD:
            stance = self.contact.copy()
        else:                                                                 # контактов нет — по высоте стоп
            stance = f[2] < f[2].min() + CONTACT_Z
        if prev is not None and self.f_prev is not None and t - prev <= MAX_DT:
            both = stance & self.st_prev
            if both.any() and not self.still:
                self.m = self.m - self.leg_scale * (f - self.f_prev)[:2, both].mean(1)
        self.mhist.append((t, self.m.copy()))
        while self.mhist[0][0] < t - MHIST:
            self.mhist.popleft()
        self.f_prev, self.st_prev = f, stance

    def m_at(self, t):
        """Путь по ногам на момент t: последнее значение не позже t (для скана, разобранного после паузы IMU)."""
        for tm, m in reversed(self.mhist):
            if tm <= t:
                return m
        return self.mhist[0][1] if self.mhist else self.m

    def _legs_ok(self, t):
        return self.use_legs and t - self.last.get('js', -np.inf) < JS_STALE

    def _predict(self, t, legs=None):
        """Позиция в момент t по последнему скану и прогнозу после него: по ногам или по последней скорости."""
        if self.t_c is None:                 # до первого скана отсчёт ведётся от начала пути по ногам (m_c = 0)
            self.t_c = t
        if legs == 'out':          # выход: между сканами — по скорости (ноги дёрганее), без свежих сканов — по ногам
            legs = not self.use_scan or (self.use_legs and t - self.t_c > VEL_HOLD)
        if (self.use_legs if legs is None else legs) and (self._legs_ok(t) or not self.use_scan):
            return self.c + self.m_at(t) - self.m_c
        hold = VEL_HOLD if self._legs_ok(t) or not self.use_legs else VEL_MAX_HOLD   # ног нет — скорость дольше
        return self.c + self.v * min(t - self.t_c, hold)

    def on_scan(self, t, ranges, angle_min, angle_increment):
        if not self._fresh('scan', t) or self.R is None or not self.use_scan:
            return
        if self.t_imu >= t:
            self._scan(t, ranges, angle_min, angle_increment)
        else:
            self.pending.append((t, ranges, angle_min, angle_increment))

    def _scan(self, t, ranges, angle_min, angle_increment):
        r = np.asarray(ranges, float)
        a = angle_min + angle_increment * np.arange(len(r))
        ok = np.isfinite(r) & (r > MIN_RANGE) & (r < MAX_RANGE)
        p = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], 1)[ok] + LIDAR_XYZ
        p = p @ self._rot(t).T                                     # в мировых осях относительно корпуса
        up = p[:, 2] > FLOOR_Z
        d, nrm = cells(p[up, :2], r[ok][up], np.flatnonzero(ok)[up], angle_increment, CELL)
        c = self._predict(t)
        t_ref = max(t, self.t_pose)                             # скан мог прийти после выдачи поз с более поздним штампом
        c0 = self._predict(t_ref, legs='out')                   # то, что выход показывал бы без этого скана
        if self.key is None:
            self._set_key(c, d, nrm)
        else:
            c, inl = self._match(c, d)
            if np.linalg.norm(c - self.key[2]) > KEY_DIST or inl < KEY_OVERLAP:
                self._set_key(c, d, nrm)
        if not self.still and t > self.t_c:
            self.v = 0.5 * self.v + 0.5 * (c - self.c) / (t - self.t_c)
        if self.still:
            self.v = np.zeros(2)
        self.c, self.t_c, self.m_c = c, t, self.m_at(t).copy()
        # поправка входит в выход плавно, начиная с последней выданной позы
        self.off = self.off * np.exp(-max(t_ref - self.t_off, 0.0) / BLEND_TAU) - (self._predict(t_ref, legs='out') - c0)
        self.t_off = t_ref

    def _set_key(self, c, d, n):
        self.n_key += 1
        ok = np.isfinite(n[:, 0])
        self.key = (c + d[ok], n[ok], c.copy())

    def _match(self, c, d):
        K, N, _ = self.key
        if len(K) < MIN_PAIRS or len(d) < MIN_PAIRS:
            self.n_skip += 1
            return c, 0.0
        kk = (K ** 2).sum(1)
        for _ in range(ITERS):
            P = c + d
            D2 = (P ** 2).sum(1)[:, None] + kk - 2 * P @ K.T
            j = D2.argmin(1)
            res = ((P - K[j]) * N[j]).sum(1)
            use = D2[np.arange(len(P)), j] < PAIR_DIST ** 2
            if use.sum() < MIN_PAIRS:
                self.n_skip += 1
                return c, 0.0
            n, res = N[j[use]], res[use]
            w = 1 / (1 + (res / ROBUST) ** 2)
            H = (w[:, None, None] * n[:, :, None] * n[:, None, :]).sum(0)
            g = -(w[:, None] * n * res[:, None]).sum(0)
            lam, E = np.linalg.eigh(H)
            good = lam > DEGEN * w.sum()                           # вдоль слабой оси остаётся прогноз
            self.degen = not good.all()
            step = E[:, good] @ ((E[:, good].T @ g) / lam[good])
            c = c + step
            if np.linalg.norm(step) < 1e-4:
                break
        self.n_match += 1
        self.n_degen += self.degen
        return c, float(np.mean(use & (np.abs(((c + d - K[j]) * N[j]).sum(1)) < 3 * ROBUST)))

    def pose(self, t):
        """(x, y, yaw, roll, pitch) в момент t или None до первой IMU. Стоя выход заморожен; поправки скана и выход
        из стоянки входят в выход плавно (с постоянной времени BLEND_TAU)."""
        if self.R is None:
            return None
        self.t_pose = max(self.t_pose, t)
        raw = self._predict(t, legs='out')
        mode = not self.use_scan or (self.use_legs and t - self.t_c > VEL_HOLD)
        self.n_out += 1
        self.n_legs_out += bool(mode)
        if mode != self.mode:                                   # смена источника выхода (сканы пропали или вернулись)
            if self.mode is not None:
                self.off = self.off * np.exp(-max(t - self.t_off, 0.0) / BLEND_TAU) - (raw - self._predict(t, legs=self.mode))
                self.t_off = t
            self.mode = mode
        if self.still:
            if self.frozen is None:
                self.frozen = raw + self.off * np.exp(-(t - self.t_off) / BLEND_TAU)
            xy = self.frozen
        else:
            if self.frozen is not None:
                self.off, self.t_off, self.frozen = self.frozen - raw, t, None
            xy = raw + self.off * np.exp(-max(t - self.t_off, 0.0) / BLEND_TAU)
        R = self.R
        yaw, pitch, roll = np.arctan2(R[1, 0], R[0, 0]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[2, 1], R[2, 2])
        return float(xy[0]), float(xy[1]), float(yaw), float(roll), float(pitch)
