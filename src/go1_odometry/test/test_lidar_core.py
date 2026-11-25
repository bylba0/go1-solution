"""Ядро режима lidar (go1_odometry.lidar_core) на синтетике: прямоугольная комната, известное движение, коридор,
стоянка, пропуски, время назад, ноги.

Тесты ядра; разработано командой «Кафедра проблем», BSD-3-Clause.
Правки: импорт go1_odom.core → go1_odometry.lidar_core; убран test_recording (нужна
запись .npz формата источника с контактами; на наших bag ядро проверяется tools/dev_t4_offline_eval.py --estimator lidar).
Порядок суставов здесь — порядок ядра (JOINTS: FL, FR, RL, RR); раскладка по именам из /joint_states — test_lidar_mode.py.
"""
import numpy as np
import pytest

from go1_odometry.lidar_core import LIDAR_XYZ, Odometry

ANG = -np.pi + 2 * np.pi / 720 * np.arange(720)


def quat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


def rot3(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def scan(xy, yaw, room=(-3.0, 5.0, -2.0, 4.0), walls_y_only=False, roll=0.0, pitch=0.0, height=0.41):
    """Дальности лучей лидара (наклонённого вместе с корпусом) до вертикальных стен комнаты (x0, x1, y0, y1) и пола."""
    R = rot3(roll, pitch, yaw)
    o = np.asarray(xy) + (R @ LIDAR_XYZ)[:2]
    d3 = np.stack([np.cos(ANG), np.sin(ANG), np.zeros_like(ANG)], 1) @ R.T
    d = d3[:, :2]
    with np.errstate(divide='ignore'):
        tx = np.where(d[:, 0] > 0, (room[1] - o[0]) / d[:, 0], (room[0] - o[0]) / d[:, 0])
        ty = np.where(d[:, 1] > 0, (room[3] - o[1]) / d[:, 1], (room[2] - o[1]) / d[:, 1])
    r = np.minimum(np.abs(tx), np.abs(ty))
    if walls_y_only:
        r = np.where(np.abs(ty) <= np.abs(tx), np.abs(ty), np.inf)
    with np.errstate(divide='ignore'):
        floor = np.where(d3[:, 2] < 0, height / -d3[:, 2], np.inf)
    return np.minimum(r, floor)


def drive(path, rate=200, scan_every=10, sway=0.0, noise=0.0, walls_y_only=False, keep=None, od=None, imu_pause=None):
    """path(t) -> (x, y, yaw); подаёт IMU 200 Гц и сканы 20 Гц, возвращает ядро и последнюю позу."""
    od = od or Odometry(use_legs=False)
    rng = np.random.default_rng(1)
    t = 0.0
    for k in range(int(path.T * rate)):
        t = k / rate
        x, y, yaw = path(t)
        roll, pitch = sway * np.sin(2 * np.pi * 2.5 * t), sway * np.cos(2 * np.pi * 2.5 * t)
        if (keep is None or keep(rng)) and not (imu_pause and imu_pause[0] <= t < imu_pause[1]):
            od.on_imu(t, quat(roll, pitch, yaw), np.zeros(3), np.zeros(3))
        if k % scan_every == 0 and (keep is None or keep(rng)):
            od.on_scan(t, scan((x, y), yaw, walls_y_only=walls_y_only, roll=roll, pitch=pitch)
                       + noise * rng.standard_normal(720), ANG[0], ANG[1] - ANG[0])
    return od, od.pose(t), path(t)


class Path:
    def __init__(self, T, v=(0.3, 0.1), w=0.3):
        self.T, self.v, self.w = T, np.array(v), w

    def __call__(self, t):
        yaw = self.w * t
        return self.v[0] * t, self.v[1] * t + 0.2 * np.sin(t), yaw


def test_room_shift_rotation_sway():
    """Сдвиги и поворот с качанием корпуса ±6° (лучи наклонены, часть уходит в пол): ошибка конца < 1 % пути."""
    od, est, true = drive(Path(10.0), sway=np.radians(6), noise=0.01)
    assert np.hypot(est[0] - true[0], est[1] - true[1]) < 0.008    # без приведения к горизонту — 1,6 см
    assert abs(est[2] - np.arctan2(np.sin(true[2]), np.cos(true[2]))) < 1e-3


def test_corridor_degenerate_keeps_prediction():
    """Видны только две стены y = const: вдоль x сдвиг не наблюдаем — скан его не трогает, поперёк — исправляет."""
    path = Path(6.0, v=(0.3, 0.0), w=0.0)
    od, est, true = drive(path, walls_y_only=True)
    assert od.degen
    assert abs(od.c[1] - path(od.t_c)[1]) < 0.01      # оценка на последнем скане; выход догоняет её плавно
    assert abs(est[1] - true[1]) < 0.05
    assert abs(est[0] - true[0]) > 0.5           # без ног вдоль коридора движения не видно


def test_still_freezes_output():
    """Стоя выход не меняется, даже если скан шумит."""
    od, _, _ = drive(Path(3.0, v=(0, 0), w=0.0), noise=0.02)
    a = od.pose(3.0)
    jitter = np.radians(0.1)                     # стоя ориентация дрожит на десятые доли градуса
    for k in range(1, 40):
        t = 3.0 + k * 0.05
        od.on_imu(t, quat(jitter * np.sin(7 * k), jitter * np.cos(5 * k), 0), np.zeros(3), np.zeros(3))
        assert k < 7 or od.still
        od.on_scan(t, scan((0.0, 0.0), 0.0) + 0.02 * np.random.default_rng(k).standard_normal(720), ANG[0], ANG[1] - ANG[0])
    b = od.pose(t)
    assert a[:3] == b[:3]


def test_dropped_messages():
    """Потеря 30 % сообщений IMU и сканов: ошибка конца < 2 см."""
    od, est, true = drive(Path(8.0), keep=lambda rng: rng.random() > 0.3)
    assert np.hypot(est[0] - true[0], est[1] - true[1]) < 0.02


def test_time_backwards_resets():
    od, _, _ = drive(Path(4.0))
    assert np.hypot(*od.pose(4.0)[:2]) > 0.5
    od.on_imu(0.0, quat(0, 0, 0), np.zeros(3), np.zeros(3))
    od.on_scan(0.0, scan((0.0, 0.0), 0.0), ANG[0], ANG[1] - ANG[0])
    assert np.hypot(*od.pose(0.0)[:2]) < 1e-9


def test_no_pose_before_imu():
    assert Odometry().pose(0.0) is None


# --- ноги: синтетическая походка по обратной кинематике той же модели (своя таблица ног, не из ядра) ---
LEG_TABLE = {'FL': (1, 1), 'FR': (1, -1), 'RL': (-1, 1), 'RR': (-1, -1)}       # знаки x и y бедра (go1.urdf.xacro)


def leg_ik(p, sx, sy):
    """Углы hip, thigh, calf для стопы p (оси корпуса), колено назад (calf < 0)."""
    L1 = L2 = 0.213
    x, y, z = p[0] - sx * 0.1881, p[1] - sy * 0.04675, p[2]
    y0 = sy * 0.08
    zl = -np.sqrt(y * y + z * z - y0 * y0)
    q1 = np.arctan2(z, y) - np.arctan2(zl, y0)
    q3 = -np.arccos(np.clip((x * x + zl * zl - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1))
    q2 = np.arctan2(-x, -zl) - np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
    return q1, q2, q3


def test_leg_ik_matches_core_kinematics():
    from go1_odometry.lidar_core import foot_pos
    p = np.array([[0.2, 0.13, -0.3], [0.17, -0.14, -0.28], [-0.2, 0.12, -0.29], [-0.15, -0.13, -0.3]])
    q = np.array([leg_ik(p[k], *LEG_TABLE[g]) for k, g in enumerate(('FL', 'FR', 'RL', 'RR'))])
    assert np.allclose(foot_pos(q), p, atol=1e-9)


def legs_drive(T=1.0, v=(0.08, -0.05), w=0.1, scans=False, trot=False, contacts=True, js_every=4, corridor=None,
               drift_y=0.0, leg_scale=1.0):
    """Корпус едет и поворачивается; опорные стопы стоят на месте в мире, переносимые идут вместе с корпусом.
    trot — рысь: опора попеременно FL+RR и FR+RL по 0,25 с, иначе все четыре в опоре. Суставы 250 Гц, IMU 200 Гц."""
    from go1_odometry.lidar_core import JOINTS
    nom = {g: np.array([sx * 0.19, sy * 0.13, -0.3]) for g, (sx, sy) in LEG_TABLE.items()}
    feet_w, phase = {}, None
    od = Odometry(use_scan=scans, leg_scale=leg_scale)
    for k in range(int(T * 1000) + 1):
        t = k / 1000
        c, yaw = np.r_[np.array(v) * t, 0.0], w * t
        R = rot3(0, 0, yaw)
        ph = int(t / 0.25) % 2 if trot else 0
        stance = (('FL', 'RR'), ('FR', 'RL'))[ph] if trot else tuple(LEG_TABLE)
        if ph != phase:                                   # касание: стопа ставится чуть впереди и стоит в мире
            phase = ph
            for g in stance:
                feet_w[g] = c + R @ (nom[g] + np.r_[v, 0.0] * (0.125 if trot else 0.0))
        if k % 5 == 0:
            od.on_imu(t, quat(0, 0, yaw), np.zeros(3), np.zeros(3))
        if corridor is not None and k % 50 == 0 and od.R is not None:   # коридор вдоль x: стена в corridor м справа
            rng = np.random.default_rng(k)
            od.on_scan(t, scan(c[:2], yaw, room=(-50.0, 50.0, -corridor, 1.0 - corridor)) + 0.01 * rng.standard_normal(720),
                       ANG[0], ANG[1] - ANG[0])
        if drift_y and k % js_every == 0:                # ноги уводят вбок drift_y м/с (проскальзывание)
            od.m = od.m + np.array([0.0, drift_y * js_every / 1000])
        if k % js_every == 0 and od.R is not None:
            for i, g in enumerate(('FL', 'FR', 'RL', 'RR')):
                if contacts:
                    od.on_contact(t, i, g in stance)
            q = {}
            for g, (sx, sy) in LEG_TABLE.items():
                pb = R.T @ (feet_w[g] - c) if g in stance else nom[g] + np.r_[0.0, 0.0, 0.06]   # перенос — стопа поднята
                q[g] = leg_ik(pb, sx, sy)
            od.on_joints(t, np.array([q[j[:2]][('hip', 'thigh', 'calf').index(j[3:-6])] for j in JOINTS]))
    return od, od.pose(T), np.array(v) * T


def test_legs_follow_body_with_rotation():
    """Только ноги + IMU: путь корпуса по неподвижным опорным стопам при сдвиге и повороте — ошибка < 1 мм."""
    od, est, true = legs_drive()
    assert np.hypot(est[0] - true[0], est[1] - true[1]) < 1e-3


def test_no_scans_output_follows_legs():
    """Сканов нет совсем (лидар не прислал ни одного) — выход ведётся по ногам, а не стоит."""
    od, est, true = legs_drive(scans=True)
    assert np.hypot(est[0] - true[0], est[1] - true[1]) < 0.02


def test_legs_trot_partial_stance():
    """Рысь: в опоре только диагональная пара, переносимые стопы движутся с корпусом. На смене пар один шаг суставов
    (4 мс) без общей опоры не считается — 0,4 мм на смену, отсюда допуск 3 мм."""
    od, est, true = legs_drive(trot=True)
    assert np.hypot(est[0] - true[0], est[1] - true[1]) < 3e-3


def test_legs_stance_by_foot_height_without_contacts():
    """Топиков контактов нет: опора — нижние стопы (переносимые подняты на 6 см); ошибка как с контактами."""
    od, est, true = legs_drive(trot=True, contacts=False)
    assert np.hypot(est[0] - true[0], est[1] - true[1]) < 3e-3


def test_imu_pause_with_rotation():
    """Пауза IMU 1,5 с во время поворота: сканы паузы ждут IMU; после паузы поза сходится (ошибка конца < 2 см)."""
    od, est, true = drive(Path(8.0), keep=None, od=None, imu_pause=(3.0, 4.5))
    assert np.hypot(est[0] - true[0], est[1] - true[1]) < 0.02


def test_sparse_joints_never_backward():
    """Голодание: суставы раз в 80 мс (между отсчётами проходит шаг, опорная стопа уже другая). Путь по ногам через
    такой разрыв не считается — поза не уходит назад (прежде: назад на десятки процентов пути)."""
    od, est, true = legs_drive(T=2.0, v=(0.2, 0.0), w=0.0, trot=True, js_every=80)
    assert abs(est[0]) < 0.005 or abs(est[0] - true[0]) < 0.02   # не считает через разрыв или считает верно



def test_cells_average_near_keep_far_and_split_at_range_jumps():
    from go1_odometry.lidar_core import cells
    dth = 2 * np.pi / 720
    near = np.full(60, 0.3)                       # шаг лучей 2,6 мм — ячейка 0,08 м собирает ~30 точек
    far = np.full(5, 9.0)                         # шаг 7,9 см < 8 см: по точке-две на ячейку
    r = np.r_[near, far]
    a = dth * np.arange(len(r))
    p = np.c_[r * np.cos(a), r * np.sin(a)]
    q, n = cells(p, r, np.arange(len(r)), dth, 0.08)
    assert len(q) == 7                            # 60 ближних — в 2 ячейки, 5 дальних — каждая своя
    assert np.all(np.abs(np.linalg.norm(q[:2], axis=1) - 0.3) < 0.003)   # среднее лежит на дуге
    assert np.allclose(q[2:], p[60:])
    r2 = np.r_[np.full(10, 1.0), np.full(10, 1.5)]   # скачок дальности 0,5 м — ячейка не смешивает края
    a2 = dth * np.arange(20)
    q2, _ = cells(np.c_[r2 * np.cos(a2), r2 * np.sin(a2)], r2, np.arange(20), dth, 0.08)
    assert np.allclose(np.linalg.norm(q2, axis=1), [1.0, 1.0, 1.5, 1.5], atol=1e-3)
    idx = np.r_[np.arange(10), np.arange(13, 23)]    # пропуск трёх лучей (проём в пустоту) — без «фантома» в проёме
    r3 = np.full(20, 1.0)
    q3, _ = cells(np.c_[r3 * np.cos(dth * idx), r3 * np.sin(dth * idx)], r3, idx, dth, 0.08)
    ang = np.arctan2(q3[:, 1], q3[:, 0]) / dth
    assert not np.any((ang > 9.5) & (ang < 12.5))


def test_normals_wall_at_grazing_angle_and_corner():
    """Нормали — по сырым точкам метрической окрестности: прямая стена вдоль x (в т. ч. под острым углом к лучам) —
    нормаль (0, ±1) у десятков ячеек; угол комнаты — без нормали; шум дальности 1 см нормали не портит."""
    from go1_odometry.lidar_core import cells
    dth = 2 * np.pi / 720
    a = -np.pi + dth * np.arange(720)
    with np.errstate(divide='ignore'):
        r = np.where(np.sin(a) < -1e-6, -0.3 / np.sin(a), np.inf)   # стена y = −0,3 (лучи вниз)
    r = r + 0.01 * np.random.default_rng(0).standard_normal(720)
    ok = np.isfinite(r) & (r < 10)
    p = np.c_[r * np.cos(a), r * np.sin(a)][ok]
    q, n = cells(p, r[ok], np.flatnonzero(ok), dth, 0.08)
    good = np.isfinite(n[:, 0])
    assert good.sum() >= 50
    assert np.all(np.abs(np.abs(n[good, 1]) - 1) < 0.02)
    c = np.r_[np.c_[np.linspace(0, 1, 50), np.zeros(50)], np.c_[np.ones(50), np.linspace(0, 1, 50)]]   # угол
    rc = np.linalg.norm(c - [0.5, 0.5], axis=1)
    _, nc = cells(c, rc, np.arange(100), 0.02, 0.08)
    assert not np.isfinite(nc[len(nc) // 2 - 1:len(nc) // 2 + 1, 0]).any()


def test_corridor_near_wall_follows_legs():
    """Коридор 1 м вдоль x, робот в 0,23 м от стены, шум дальности 1 см: вдоль оси скан ничего не знает, оценка идёт за
    ногами (без усреднения по ячейкам шумные нормали ближних точек тянули позу к опорному скану)."""
    od, est, true = legs_drive(T=4.0, v=(0.25, 0.0), w=0.05, scans=True, trot=True, corridor=0.23, drift_y=0.02)
    assert abs(est[0] - true[0]) < 0.05                # вдоль оси — ноги (скан ничего не знает и не тянет назад)
    assert abs(est[1]) < 0.01                          # бок: ноги уводят на 8 см, скан исправляет
    assert od.n_match > 70 and od.n_skip == 0          # сопоставления не пропускаются
    assert len(od.key[0]) >= 100


def test_leg_scale_multiplies_leg_path():
    """Калибровка пути по ногам: множитель 1,13 удлиняет путь ровно во столько раз (курс не меняется)."""
    od, est, true = legs_drive(trot=True, leg_scale=1.13)
    assert np.hypot(est[0] - 1.13 * true[0], est[1] - 1.13 * true[1]) < 3e-3


def test_normals_need_base_and_stay_within_segment():
    """Короткий обрывок (3 см) у самого лидара — без нормали: на базе короче 5 см шум дальности задаёт направление
    (прежняя ловушка);
    две длинные стены, разделённые скачком дальности (проём), — у ячеек на краях нормали свои, окно не смешивает стены."""
    from go1_odometry.lidar_core import cells
    rng = np.random.default_rng(3)
    x = np.linspace(0, 0.03, 12)                                   # обрывок 3 см на расстоянии 0,2 м: даже почти
    p = np.c_[x, np.full(12, 0.2) + 0.001 * rng.standard_normal(12)]   # без шума база мала — направление не задано
    r = np.linalg.norm(p, axis=1)
    _, n = cells(p, r, np.arange(12), 0.002, 0.08)
    assert not np.isfinite(n[:, 0]).any()
    xa, xb = np.linspace(-1.0, -0.02, 50), np.linspace(0.02, 1.0, 50)   # стена y = 1 и за проёмом стена y = 1,5
    p2 = np.r_[np.c_[xa, np.ones(50)], np.c_[xb, np.full(50, 1.5)]]
    r2 = np.linalg.norm(p2, axis=1)
    m2, n2 = cells(p2, r2, np.arange(100), 0.02, 0.08)
    edge = np.argmin(np.abs(m2[:, 0]))                             # ячейки у проёма с обеих сторон
    for k in (edge - 1, edge, edge + 1):
        if 0 <= k < len(n2):
            assert np.isfinite(n2[k, 0]) and abs(abs(n2[k, 1]) - 1) < 0.02


def test_normals_at_range_step_and_inf_gap():
    """Ступенька стены 0,15 м (лучи идут вдоль перпендикулярной стенки и её не видят) — у ячеек при скачке нормали
    наклонены не больше 2°; стена с 10 лучами inf посередине — ни одна ячейка не уходит от стены дальше 1 см."""
    from go1_odometry.lidar_core import cells
    dth = 2 * np.pi / 720
    a = np.pi / 2 + dth * np.arange(-80, 81)                       # лучи вверх, ±40°
    y = np.where(np.cos(a) > 0, 1.15, 1.0)                         # справа стена дальше на 0,15 м
    r = y / np.sin(a)
    p = np.c_[r * np.cos(a), r * np.sin(a)]
    m, n = cells(p, r, np.arange(len(r)), dth, 0.08)
    k = np.argsort(np.abs(m[:, 0]))[:4]                            # ячейки у ступеньки
    ok = np.isfinite(n[k, 0])
    assert ok.any() and np.all(np.degrees(np.arccos(np.minimum(np.abs(n[k][ok, 1]), 1))) <= 2.0)
    r2 = 1.0 / np.sin(a)
    r2[75:85] = np.inf
    keep = np.isfinite(r2)
    p2 = np.c_[r2 * np.cos(a), r2 * np.sin(a)][keep]
    m2, _ = cells(p2, r2[keep], np.flatnonzero(keep), dth, 0.08)
    assert np.all(np.abs(m2[:, 1] - 1.0) < 0.01)
