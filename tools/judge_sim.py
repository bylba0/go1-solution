#!/usr/bin/env python3
"""judge_sim — имитатор Judge для T2 (ТЗ_T2, раздел 4.5). НЕ официальный Judge: сценарии и пороги — по тексту ТЗ.

Публикует /clock (sim time = wall от старта, 100 Гц, как MuJoCo-сим), /lowstate (поза стоя), /lowcmd_bare по сценариям;
записывает /lowcmd и выставляет баллы по таблице ТЗ_T2:
  1. частота /lowcmd ≥ 50 Гц в окне 5 с sim и максимальный интервал ≤ 100 мс sim           — 3
  2. угловые ограничения 12 суставов                                                        — 2
  3. нет самоколлизий > 1 мм (путь между соседними командами с шагом 0.1°) + обработка команд — 3
  4. удержание последней принятой цели через 0.5 с после потери входа                        — 2
  5. нет NaN/Inf в выходных командах                                                        — 2
Проверка коллизий — своя (исходный go1.xml, mj_collision, 655 пар), независимая от кода Safety Block.

Запуск (в контейнере образа решения рядом с `/entrypoint.sh T2`, одинаковые ROS_DOMAIN_ID/CYCLONEDDS_URI):
  python3 judge_sim.py [--model /opt/go1/models/unitree_go1/go1.xml] [--cmd-rate 100] [--json out.json]
"""
import argparse
import json
import math
import sys
import time

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from unitree_go.msg import LowCmd, LowState

DEG = math.pi / 180
Q_MIN = np.array([-0.863, -0.686, -2.818] * 4)        # ТЗ_T2, таблица 4.3
Q_MAX = np.array([0.863, 4.501, -0.888] * 4)
HOME = np.array([0.0, 0.9, -1.8] * 4)
STAND_MEASURED = np.array([0.0, 0.82, -1.82] * 4)     # как /lowstate у сима после оседания
# 8 суставов вне диапазонов; после клипа путь из стойки и обратно без самоколлизий (проверено) — проверяем именно клип.
# (Цели, прямой путь к которым проходит через самоколлизию, — отдельная фаза «поза с самоколлизией».)
OUT_OF_RANGE = np.array([-1.5, -1.2, -0.3, 1.5, 0.9, -3.5, 0.0, 0.9, -0.3, 0.0, 5.0, -3.5])
JOINTS = [f'{leg}_{k}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for k in ('hip', 'thigh', 'calf')]
FIELDS = ('q', 'dq', 'kp', 'kd', 'tau')


# ---------------------------------------------------------------------------- коллизии (как у Judge)
class JudgeCollision:
    def __init__(self, path):
        self.m = mujoco.MjModel.from_xml_path(path)       # исходные margin = 1 мм, без пола → ровно 655 пар
        self.d = mujoco.MjData(self.m)
        key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, key)
        self.adr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS]

    def penetration(self, q):
        self.d.qpos[self.adr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        if self.d.ncon == 0:
            return 0.0
        return max(0.0, -float(np.min(self.d.contact.dist[:self.d.ncon])))

    def path_max(self, qa, qb, step=0.1 * DEG):
        n = max(1, int(math.ceil(float(np.max(np.abs(qb - qa))) / step)))
        return max(self.penetration(qa + (qb - qa) * (k / n)) for k in range(n + 1))

    def find_colliding(self, depth=0.005, seed=11):
        rng = np.random.default_rng(seed)
        for _ in range(50000):
            q = rng.uniform(Q_MIN + 0.02, Q_MAX - 0.02)
            if self.penetration(q) > depth:
                return q
        raise RuntimeError('нет позы с самоколлизией')


# ---------------------------------------------------------------------------- сценарии
CYCLES = 50
CYCLE_DANGER, CYCLE_SAFE = 0.35, 1.0                  # с sim: случайная (часто опасная) цель → безопасная рядом с home


def build_cycles(jc, seed=2026):
    """Ловушка из ревью: случайная цель в пределах ТЗ (часто опасная) → безопасная поза рядом с home."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(CYCLES):
        danger = rng.uniform(Q_MIN, Q_MAX)
        while True:
            safe = np.clip(HOME + rng.uniform(-0.15, 0.15, 12), Q_MIN + 0.011, Q_MAX - 0.011)
            if jc.penetration(safe) == 0.0:
                break
        out.append((danger, safe))
    return out


def build_scenario(collide_q, cycles):
    """Список фаз: (имя, длительность sim, функция t_rel → (q12 | None, поле_с_NaN | None))."""
    jump = HOME.copy()
    jump[1] += 30 * DEG
    jump[2] -= 30 * DEG
    jump[7] += 30 * DEG
    out_of_range = OUT_OF_RANGE

    period = CYCLE_DANGER + CYCLE_SAFE

    def cycle(t):
        i = min(int(t // period), len(cycles) - 1)
        return (cycles[i][0] if t - i * period < CYCLE_DANGER else cycles[i][1]), None

    def sine(t):
        return HOME + np.array([0.08, 0.25, -0.3] * 4) * math.sin(2 * math.pi * 0.5 * t)

    return [
        ('старт', 2.0, lambda t: (None, None)),
        ('безопасная траектория', 7.0, lambda t: (sine(t), None)),
        ('выход за пределы', 1.5, lambda t: (out_of_range, None)),
        ('домой', 1.0, lambda t: (HOME, None)),
        ('скачок 30°', 1.0, lambda t: (jump, None)),
        ('домой 2', 1.0, lambda t: (HOME, None)),
        ('поза с самоколлизией', 2.0, lambda t: (collide_q, None)),
        ('домой после коллизии', 1.5, lambda t: (HOME, None)),
        (f'опасная → безопасная ×{len(cycles)}', len(cycles) * period, cycle),
        ('домой 3', 1.5, lambda t: (HOME, None)),
        ('NaN/Inf', 1.0, lambda t: (HOME + np.array([0.0, 0.3, -0.3] * 4), 'bad')),
        ('потеря входа', 1.5, lambda t: (None, None)),
        ('возобновление', 1.5, lambda t: (HOME + np.array([0.0, 0.15, -0.15] * 4), None)),
    ]


class Judge(Node):
    def __init__(self, args):
        super().__init__('t2_judge_sim')
        qos_out = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        qos_in = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.clock_pub = self.create_publisher(Clock, '/clock', qos_out)
        self.state_pub = self.create_publisher(LowState, '/lowstate', qos_out)
        self.cmd_pub = self.create_publisher(LowCmd, '/lowcmd_bare', qos_out)
        self.create_subscription(LowCmd, '/lowcmd', self._on_out, qos_in)
        self.sim_t = 0.0
        self.ext_clock = None         # --external-sim: время сима из /clock
        self.ext_t0 = None
        self.meas = []                # --external-sim: (sim_t, q12 измеренные)
        self.height = []              # --external-sim: (sim_t, z корпуса по /odom/ground_truth)
        if args.external_sim:
            self.create_subscription(Clock, '/clock', self._on_clock, qos_profile_sensor_data)
            self.create_subscription(LowState, '/lowstate', self._on_state, qos_profile_sensor_data)
            self.create_subscription(Odometry, '/odom/ground_truth', self._on_odom, qos_profile_sensor_data)
        self.rec = []                 # (sim_t, dict полей 20×) для каждого /lowcmd
        self.sent = []                # (sim_t, q12 | None, bad)
        self.bad_k = 0

    def _on_clock(self, m):
        self.ext_clock = m.clock.sec + m.clock.nanosec * 1e-9
        if self.ext_t0 is not None:
            self.sim_t = self.ext_clock - self.ext_t0

    def _on_state(self, m):
        if self.ext_t0 is not None and (not self.meas or self.sim_t - self.meas[-1][0] >= 0.01):
            self.meas.append((self.sim_t, np.array([m.motor_state[i].q for i in range(12)])))

    def _on_odom(self, m):
        if self.ext_t0 is not None and (not self.height or self.sim_t - self.height[-1][0] >= 0.02):
            self.height.append((self.sim_t, m.pose.pose.position.z))

    def _on_out(self, m):
        mc = m.motor_cmd
        self.rec.append((self.sim_t, {
            'mode': np.array([c.mode for c in mc]), **{f: np.array([getattr(c, f) for c in mc], dtype=float) for f in FIELDS}}))

    def publish_clock(self, t):
        c = Clock()
        c.clock.sec = int(t)
        c.clock.nanosec = int((t - int(t)) * 1e9)
        self.clock_pub.publish(c)

    def publish_state(self):
        s = LowState()
        for i in range(12):
            s.motor_state[i].mode = 10
            s.motor_state[i].q = float(STAND_MEASURED[i])
        self.state_pub.publish(s)

    def publish_cmd(self, q12, bad):
        m = LowCmd()
        for i in range(12):
            c = m.motor_cmd[i]
            c.mode, c.q, c.kp, c.kd = 10, float(q12[i]), 40.0, 2.0
        if bad:
            variants = [('q', 2, float('nan')), ('kp', 7, float('inf')), ('tau', 15, float('-inf')),
                        ('dq', 0, float('nan')), ('kd', 11, float('-inf'))]
            f, i, v = variants[self.bad_k % len(variants)]
            setattr(m.motor_cmd[i], f, v)
            self.bad_k += 1
        self.cmd_pub.publish(m)


def run(args):
    jc = JudgeCollision(args.model)
    collide_q = jc.find_colliding()
    cycles = build_cycles(jc)
    phases = build_scenario(collide_q, cycles)
    rclpy.init()
    j = Judge(args)
    total = sum(p[1] for p in phases)
    bounds, t_acc = [], 0.0
    for name, dur, _ in phases:
        bounds.append((name, t_acc, t_acc + dur))
        t_acc += dur

    print(f'[judge_sim] сценарий {total:.1f} с sim, вход {args.cmd_rate:.0f} Гц'
          f'{", внешний сим (время из /clock)" if args.external_sim else ""}', file=sys.stderr)
    if args.external_sim:
        deadline = time.monotonic() + 60
        while j.ext_clock is None and time.monotonic() < deadline:
            rclpy.spin_once(j, timeout_sec=0.05)
        if j.ext_clock is None:
            raise RuntimeError('нет /clock от симулятора')
        j.ext_t0 = j.ext_clock
        j.sim_t = 0.0
    w0 = time.monotonic()
    next_clock = next_state = 0.0
    next_cmd_w = 0.0
    while True:
        if args.external_sim:
            now = j.sim_t                    # sim time настоящего симулятора
        else:
            now = time.monotonic() - w0      # sim time = wall (как MuJoCo-сим, RTF 1)
            if now >= next_clock:
                j.sim_t = round(now, 2)      # квантование как у /clock 100 Гц
                j.publish_clock(j.sim_t)
                next_clock += 0.01
            if now >= next_state:
                j.publish_state()
                next_state += 0.01
        wall = time.monotonic() - w0
        if wall >= next_cmd_w:
            next_cmd_w += 1.0 / args.cmd_rate
            for (name, a, b), (_, _, fn) in zip(bounds, phases):
                if a <= now < b:
                    q, bad = fn(now - a)
                    if q is not None:
                        j.publish_cmd(q, bad == 'bad')
                        j.sent.append((j.sim_t, None if bad else np.asarray(q, float), bool(bad)))
                    break
        if now >= total:
            break
        rclpy.spin_once(j, timeout_sec=0.0005)
    for _ in range(50):
        rclpy.spin_once(j, timeout_sec=0.01)
    rec, meas, height = j.rec, j.meas, j.height
    j.destroy_node()
    rclpy.try_shutdown()
    rep, ok = evaluate(rec, bounds, jc, collide_q, cycles)
    if args.external_sim:
        rep['physics'] = physics_report(rec, meas, height, bounds)
    return rep, ok


def physics_report(rec, meas, height, bounds):
    """Справочно (не критерий ТЗ_T2): как настоящий MuJoCo-сим отработал наши /lowcmd."""
    out = {}
    if meas and rec:
        tc = np.array([r[0] for r in rec])
        qc = np.array([r[1]['q'][:12] for r in rec])
        a, b = [(x, y) for n, x, y in bounds if n == 'безопасная траектория'][0]
        errs = []
        for tm, qm in meas:
            if a + 1.0 <= tm < b:
                k = np.searchsorted(tc, tm) - 1
                if k >= 0:
                    errs.append(float(np.max(np.abs(qm - qc[k]))))
        if errs:
            out['tracking_err_rad'] = {'mean': float(np.mean(errs)), 'max': float(np.max(errs))}
    if height:
        z = np.array([h[1] for h in height])
        out['trunk_z_m'] = {'start': float(z[0]), 'min': float(z.min()), 'end': float(z[-1])}
    out['lowstate_samples'] = len(meas)
    return out


# ---------------------------------------------------------------------------- оценка
def evaluate(rec, bounds, jc, collide_q, cycles):
    ph = {name: (a, b) for name, a, b in bounds}
    res, notes = {}, []
    if len(rec) < 10:
        print('[judge_sim] нет выходных сообщений /lowcmd', file=sys.stderr)
        return {'score': 0, 'max': 12, 'rows': []}, False
    t = np.array([r[0] for r in rec])
    Q = np.array([r[1]['q'][:12] for r in rec])

    # 1. частота и паузы (по sim time), окно 5 с в безопасной траектории + скользящие окна по всему прогону после старта
    a, b = ph['безопасная траектория']
    w0 = a + 1.0
    in_w = (t >= w0) & (t < w0 + 5.0)
    rate = in_w.sum() / 5.0
    after = t[t >= a]
    gaps = np.diff(after) if after.size > 1 else np.array([1e9])
    max_gap = float(gaps.max())
    min_win = min(((t >= s) & (t < s + 5.0)).sum() / 5.0 for s in np.arange(a, t[-1] - 5.0, 0.5))
    ok1 = rate >= 50 and max_gap <= 0.1 and min_win >= 50
    mean_rate = (after.size - 1) / float(after[-1] - after[0]) if after.size > 1 else 0.0
    res[1] = (ok1, 3, f'{rate:.0f} Гц в окне 5 с, средн. {mean_rate:.0f} Гц, мин. по скользящим окнам {min_win:.0f} Гц, '
              f'макс. пауза {max_gap * 1000:.0f} мс')

    # 5. NaN/Inf во всех 20 моторах и всех полях
    nonfinite = sum(int(not np.all(np.isfinite(r[1][f]))) for r in rec for f in FIELDS)
    res[5] = (nonfinite == 0, 2, f'сообщений с NaN/Inf: {nonfinite}')
    mode_bad = sum(int(np.any(r[1]['mode'][:12] != 10)) for r in rec)
    if mode_bad:
        notes.append(f'ВНИМАНИЕ: в {mode_bad} сообщениях mode != 0x0A для моторов 0–11')

    # 2. угловые пределы
    viol = int(np.sum(np.any((Q < Q_MIN - 1e-9) | (Q > Q_MAX + 1e-9), axis=1)))
    res[2] = (viol == 0, 2, f'сообщений вне пределов: {viol} из {len(Q)}')

    # 3. коллизии, шаг ≤ 5°, обработка безопасных/опасных команд
    steps = np.max(np.abs(np.diff(Q, axis=0)), axis=1) if len(Q) > 1 else np.zeros(1)
    worst_pen = 0.0
    for k in range(len(Q)):
        if k == 0 or not np.array_equal(Q[k], Q[k - 1]):
            worst_pen = max(worst_pen, jc.path_max(Q[k - 1], Q[k]) if k else jc.penetration(Q[0]))

    def q_at(name, frac=1.0):
        a, b = ph[name]
        tt = a + (b - a) * frac
        idx = np.searchsorted(t, tt) - 1
        return Q[max(0, min(idx, len(Q) - 1))]

    a, b = ph['безопасная траектория']
    sel = (t >= a + 0.5) & (t < b)
    track = 0.0
    for tk, qk in zip(t[sel], Q[sel]):
        ref = HOME + np.array([0.08, 0.25, -0.3] * 4) * math.sin(2 * math.pi * 0.5 * (tk - a))
        track = max(track, float(np.max(np.abs(qk - ref))))
    jump = HOME.copy()
    jump[1] += 30 * DEG
    jump[2] -= 30 * DEG
    jump[7] += 30 * DEG
    ca, _ = ph[f'опасная → безопасная ×{len(cycles)}']
    period = CYCLE_DANGER + CYCLE_SAFE
    cyc_ok = 0
    for i, (_, safe) in enumerate(cycles):
        idx = np.searchsorted(t, ca + (i + 1) * period - 0.005) - 1
        cyc_ok += bool(idx >= 0 and np.max(np.abs(Q[idx] - safe)) < 1e-3)
    handled = {
        'траектория отслеживается (≤ 0.05 рад)': track <= 0.05,
        'скачок 30° отработан рампой': float(np.max(np.abs(q_at('скачок 30°') - jump))) < 1e-3,
        'выход за пределы → у границы': float(np.max(np.abs(q_at('выход за пределы') - np.clip(OUT_OF_RANGE, Q_MIN, Q_MAX)))) < 0.02,
        'поза с коллизией не достигнута': float(np.max(np.abs(q_at('поза с самоколлизией') - collide_q))) > 1e-3,
        'после коллизии вернулся домой': float(np.max(np.abs(q_at('домой после коллизии') - HOME))) < 1e-3,
        f'безопасные цели после опасных: {cyc_ok}/{len(cycles)}': cyc_ok == len(cycles),
        'возобновление отработано': float(np.max(np.abs(q_at('возобновление') - (HOME + np.array([0.0, 0.15, -0.15] * 4))))) < 1e-3,
    }
    ok3 = worst_pen <= 0.001 and float(steps.max()) <= 5 * DEG + 1e-9 and all(handled.values())
    failed = [k for k, v in handled.items() if not v]
    res[3] = (ok3, 3, f'проникновение макс {worst_pen * 1000:.3f} мм, шаг макс {math.degrees(float(steps.max())):.2f}°'
              + ('' if not failed else '; НЕ: ' + ', '.join(failed)))

    # 4. потеря входа: через 0.5 с — q = последняя принятая цель, dq = 0, tau = 0, kp/kd допустимы, поток идёт
    a, b = ph['потеря входа']
    last_target = q_at('NaN/Inf')     # NaN-сообщения отклоняются → цель = поза «домой после коллизии»
    sel = [r for r in rec if a + 0.5 <= r[0] < b]
    hold_ok = bool(sel)
    for _, f in sel:
        hold_ok &= bool(np.allclose(f['q'][:12], last_target, atol=1e-6) and np.all(f['dq'][:12] == 0)
                        and np.all(f['tau'][:12] == 0) and np.all(f['kp'][:12] > 0) and np.all(f['kp'][:12] <= 1000)
                        and np.all(f['kd'][:12] >= 0))
    loss_t = t[(t >= a) & (t < b)]
    loss_gap = float(np.max(np.diff(loss_t))) if loss_t.size > 1 else 1e9
    res[4] = (hold_ok and loss_gap <= 0.1, 2, f'{len(sel)} сообщений в удержании, пауза макс {loss_gap * 1000:.0f} мс')
    nan_target_ok = float(np.max(np.abs(q_at('NaN/Inf') - HOME))) < 1e-6
    if not nan_target_ok:
        notes.append('ВНИМАНИЕ: во время NaN/Inf-команд цель изменилась')
        res[5] = (False, 2, res[5][2] + '; цель изменилась от NaN-команд')

    names = {1: 'Частота /lowcmd ≥ 50 Гц (5 с), пауза ≤ 100 мс', 2: 'Угловые ограничения 12 суставов',
             3: 'Нет самоколлизий + обработка безопасных/опасных команд', 4: 'Удержание при потере входа (0.5 с)',
             5: 'Нет NaN/Inf в выходных командах'}
    rows, score = [], 0
    for k in sorted(res):
        ok, pts, detail = res[k]
        score += pts if ok else 0
        rows.append({'n': k, 'criterion': names[k], 'ok': bool(ok), 'points': pts if ok else 0, 'max': pts, 'detail': detail})
    report = {'score': score, 'max': 12, 'rows': rows, 'notes': notes, 'messages': len(rec)}
    return report, score == 12


def print_table(rep):
    print()
    print('| # | Критерий ТЗ_T2 | Баллы | Детали |')
    print('|---|---|---|---|')
    for r in rep['rows']:
        print(f"| {r['n']} | {r['criterion']} | {r['points']}/{r['max']} | {r['detail']} |")
    print(f"| | **Итого** | **{rep['score']}/{rep['max']}** | сообщений /lowcmd: {rep.get('messages', 0)} |")
    for n in rep.get('notes', []):
        print(n)
    if 'physics' in rep:
        ph = rep['physics']
        te = ph.get('tracking_err_rad', {})
        tz = ph.get('trunk_z_m', {})
        print(f"Физика сима (справочно): ошибка слежения суставов на безопасной траектории — "
              f"средн. {te.get('mean', float('nan')):.3f} рад, макс. {te.get('max', float('nan')):.3f} рад; "
              f"высота корпуса старт {tz.get('start', float('nan')):.3f} м, мин. {tz.get('min', float('nan')):.3f} м, "
              f"конец {tz.get('end', float('nan')):.3f} м; кадров /lowstate {ph.get('lowstate_samples', 0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/opt/go1/models/unitree_go1/go1.xml')
    ap.add_argument('--cmd-rate', type=float, default=100.0)
    ap.add_argument('--json', default='')
    ap.add_argument('--external-sim', action='store_true',
                    help='не публиковать /clock и /lowstate: время и состояние — от настоящего симулятора организаторов')
    args = ap.parse_args()
    rep, ok = run(args)
    print_table(rep)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
