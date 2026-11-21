"""T4: оценщик одометрии Go1 без ROS (numpy) — ориентация из IMU, положение по опорным ногам.

Идея (кинематическая одометрия ног):
  * ориентация корпуса R (odom ← trunk) — кватернион /imu/data (мировой, без шума), курс — относительно начального;
    imu_link закреплён на trunk без поворота, поэтому ориентация trunk = ориентация IMU (смещение IMU на ориентацию
    не влияет; ω одинакова для всех точек твёрдого тела). Ускорения IMU не используются вовсе.
  * стопы в системе trunk — прямая кинематика по URDF сима; f_i = R · p_i — стопа относительно trunk в осях odom;
  * опорная нога неподвижна в мире ⇒ смещение корпуса за шаг Δx = −(f_i(t) − f_i(t−1)) (это ∫ −(ṗ + ω × p) dt);
    берём среднее по ногам, опорным в обоих тиках (низкие по высоте в горизонтальной системе), с отбраковкой выбросов;
  * ZUPT: суставы почти неподвижны и корпус не поворачивается ⇒ смещение 0.
Знание траектории, команд, истины не используется.
"""
import math

import numpy as np

LEGS = ('FR', 'FL', 'RR', 'RL')
JOINTS_PER_LEG = ('hip', 'thigh', 'calf')
JOINT_NAMES = tuple(f'{leg}_{j}_joint' for leg in LEGS for j in JOINTS_PER_LEG)

# URDF сима (docs/t4/go1_sim.urdf): trunk→hip, hip→thigh, thigh→calf, calf→foot
HIP_X = 0.1881
HIP_Y = 0.04675
THIGH_Y = 0.08
L_THIGH = 0.213
L_CALF = 0.213
# знаки (x, y) для FR, FL, RR, RL
LEG_SIGNS = np.array([[1, -1], [1, 1], [-1, -1], [-1, 1]], dtype=float)
HIP_OFFSETS = np.column_stack([LEG_SIGNS[:, 0] * HIP_X, LEG_SIGNS[:, 1] * HIP_Y, np.zeros(4)])
THIGH_OFFSETS_Y = LEG_SIGNS[:, 1] * THIGH_Y


def foot_positions(q):
    """q: (4, 3) [hip, thigh, calf] в порядке LEGS → (4, 3) положения стоп (центр сферы) в системе trunk."""
    q = np.asarray(q, dtype=float).reshape(4, 3)
    q0, q1, q2 = q[:, 0], q[:, 1], q[:, 2]
    # в системе thigh: стопа = Ry(q1) · ([0,0,−L1] + Ry(q2) · [0,0,−L2])
    # Ry(a)·[0,0,−L] = [−L sin a, 0, −L cos a]
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    x = -L_THIGH * np.sin(q1) - L_CALF * s12
    z = -L_THIGH * np.cos(q1) - L_CALF * c12
    y = THIGH_OFFSETS_Y
    # Rx(q0) · [x, y, z]
    c0, s0 = np.cos(q0), np.sin(q0)
    return HIP_OFFSETS + np.column_stack([x, c0 * y - s0 * z, s0 * y + c0 * z])


# ---------------------------------------------------------------- кватернионы (x, y, z, w)

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-6:
        return None
    return q / n


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_from_yaw(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def quat_from_rotvec(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.array([0.5 * v[0], 0.5 * v[1], 0.5 * v[2], 1.0])
    s = math.sin(a / 2) / a
    return np.array([v[0] * s, v[1] * s, v[2] * s, math.cos(a / 2)])


def quat_to_rotvec(q):
    q = q if q[3] >= 0 else -q
    v = np.asarray(q[:3])
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return 2 * v
    return v * (2 * math.atan2(s, q[3]) / s)


def all_finite(*arrs):
    return all(np.all(np.isfinite(np.asarray(a, dtype=float))) for a in arrs)


DEFAULTS = {
    'orientation_mode': 'imu',      # imu — кватернион /imu/data; gyro — интегрирование угловой скорости (запасной)
    'heading_reference': 'world',   # world — курс = мировой yaw IMU с первого сообщения (не зависит от того, совмещает ли
                                    # Judge только положение или и курс); initial — курс относительно начального (старт с 0)
    'stance_height': 0.005,         # м: вес опоры ноги w = 1 − Δh / stance_height, Δh — высота над самой низкой стопой
    'zupt_foot_speed': 0.01,        # м/с: макс. скорость стопы относительно trunk для «стоим»
    'zupt_ang_speed': 0.02,         # рад/с: макс. угловая скорость (по разности ориентаций) для «стоим»
    'max_dt': 0.1,                  # с: при большем разрыве данных шаг не интегрируется (только пересброс)
    'foot_radius': 0.02,            # м: радиус сферы стопы (высота trunk над полом = высота над центром стопы + r)
    'initial_height': 0.29,         # м: z до первых /joint_states
    'height_alpha': 0.02,           # фильтр высоты trunk (доля нового значения за тик)
    'twist_alpha': 0.2,             # фильтр скоростей для twist
    'contact_timeout': 0.1,         # с: маска контактов (необязательная, /foot_force) старше — не используется
}


class OdometryEstimator:
    """Без ROS. set_imu / set_joints / set_contacts — сохраняют последнее, step(t) — расчёт (на таймере)."""

    def __init__(self, **params):
        p = dict(DEFAULTS)
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'unknown params: {sorted(unknown)}')
        p.update(params)
        if p['orientation_mode'] not in ('imu', 'gyro'):
            raise ValueError("orientation_mode: 'imu' | 'gyro'")
        if p['heading_reference'] not in ('world', 'initial'):
            raise ValueError("heading_reference: 'world' | 'initial'")
        self.p = p
        # входы (последние валидные)
        self._imu_q = None
        self._q = None
        self._js_t = None
        self._contacts = None
        self._contacts_t = None
        # состояние
        self._yaw0_inv = None          # кватернион −yaw0 (initial) или единичный (world)
        self._gyro_q = None            # ориентация в режиме gyro
        self._gyro_t = None
        self.orientation = None        # (x, y, z, w) odom ← trunk
        self.position = np.array([0.0, 0.0, p['initial_height']])
        self._height = None
        self._prev = None              # (t, f, p, h, orient) прошлого шага по суставам
        self.lin_vel = np.zeros(3)     # в системе trunk
        self.ang_vel = np.zeros(3)     # в системе trunk
        self.stationary = False
        self.weights = np.zeros(4)
        self.rejected = 0              # счётчик отброшенных (NaN/Inf) входов

    # ------------------------------------------------------------ входы
    def set_imu(self, t, quat, gyro=None):
        qn = quat_normalize(quat) if all_finite(quat) else None
        if qn is None or not math.isfinite(t):
            self.rejected += 1
            return False
        if self.p['orientation_mode'] == 'gyro':
            if gyro is None or not all_finite(gyro):
                self.rejected += 1
                return False
            if self._gyro_q is None:
                self._gyro_q = qn
            elif self._gyro_t is not None and 0 < t - self._gyro_t <= self.p['max_dt']:
                # ω в системе тела: q ← q ⊗ exp(ω dt)
                dq = quat_from_rotvec(np.asarray(gyro, dtype=float) * (t - self._gyro_t))
                self._gyro_q = quat_normalize(quat_mul(self._gyro_q, dq))
            self._gyro_t = t
        self._imu_q = qn
        return True

    def set_joints(self, t, q):
        """q: 12 значений в порядке JOINT_NAMES."""
        q = np.asarray(q, dtype=float)
        if q.shape != (12,) or not all_finite(q) or not math.isfinite(t):
            self.rejected += 1
            return False
        self._q = q
        self._js_t = t
        return True

    def set_contacts(self, t, mask):
        """Необязательно: маска опоры (4, порядок LEGS), например по /foot_force."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (4,) or not math.isfinite(t):
            self.rejected += 1
            return False
        self._contacts, self._contacts_t = mask, t
        return True

    # ------------------------------------------------------------ расчёт
    def step(self, t):
        """Обновляет оценку. Возвращает True, если поза определена (есть ориентация)."""
        qw = self._gyro_q if self.p['orientation_mode'] == 'gyro' else self._imu_q
        if qw is None:
            return False
        if self._yaw0_inv is None:
            self._yaw0_inv = (quat_from_yaw(-quat_yaw(qw)) if self.p['heading_reference'] == 'initial'
                              else np.array([0.0, 0.0, 0.0, 1.0]))
        orient = quat_normalize(quat_mul(self._yaw0_inv, qw))
        if orient is None:
            return self.orientation is not None
        self.orientation = orient
        if self._q is None:
            return True

        js_t = self._js_t
        if self._prev is not None and js_t <= self._prev[0]:
            return True                # новых суставов нет — поза прежняя
        p = foot_positions(self._q.reshape(4, 3))
        f = p @ quat_to_rot(orient).T  # стопы относительно trunk в осях odom
        h = f[:, 2] - f[:, 2].min()
        if self._prev is not None and js_t - self._prev[0] <= self.p['max_dt']:
            self._integrate(t, js_t - self._prev[0], f, p, h, orient)
        else:
            self.weights = (h < self.p['stance_height']).astype(float)
        sup = self.weights > 0
        body_h = -float(np.mean(f[sup, 2] if sup.any() else f[:, 2])) + self.p['foot_radius']
        self._height = body_h if self._height is None else \
            self._height + self.p['height_alpha'] * (body_h - self._height)
        self.position[2] = self._height
        self._prev = (js_t, f, p, h, orient)
        return True

    def _integrate(self, t, dt, f, p, h, orient):
        _, f0, p0, h0, o0 = self._prev
        foot_speed = np.linalg.norm(p - p0, axis=1) / dt
        w_body = quat_to_rotvec(quat_mul(quat_conj(o0), orient)) / dt
        a = self.p['twist_alpha']
        self.ang_vel += a * (w_body - self.ang_vel)
        self.stationary = (float(foot_speed.max()) < self.p['zupt_foot_speed']
                           and float(np.linalg.norm(w_body)) < self.p['zupt_ang_speed'])
        # вес опоры: нога низко в обоих тиках (мягкий порог по высоте над самой низкой стопой)
        w = np.clip(1.0 - np.maximum(h, h0) / self.p['stance_height'], 0.0, 1.0)
        if self._contacts is not None and t - self._contacts_t <= self.p['contact_timeout'] \
                and (w * self._contacts).sum() > 1e-6:
            w = w * self._contacts
        if w.sum() < 1e-6:
            w = (h == 0).astype(float)
        self.weights = w
        if self.stationary:
            disp = np.zeros(3)
        else:
            # опорная стопа неподвижна в мире ⇒ корпус сместился на −Δf (= ∫ −(ṗ + ω × p) dt)
            disp = (w[:, None] * -(f - f0)).sum(axis=0) / w.sum()
        self.position[0] += disp[0]
        self.position[1] += disp[1]
        v_body = quat_to_rot(orient).T @ (disp / dt)
        self.lin_vel += a * (v_body - self.lin_vel)
