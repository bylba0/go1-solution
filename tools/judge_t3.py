#!/usr/bin/env python3
"""Стенд-судья T3 (ходьба RL-политикой) на живом MuJoCo-симуляторе организаторов.

Решение — чёрный ящик: публичные топики ROS 2. Истина — /odom/ground_truth публичного сима (только в стенде).
Правила — ТЗ_T3 разд. 5.3 (docs/TZ_SUMMARY.md, раздел T3):
  * /cmd_vel: linear.x = preferred_speed_mps, остальное 0, весь эпизод;
  * прогрев 1,5 с sim, затем зачётное окно ≤ 90 с sim (отсчёт после прогрева);
  * успех: ≥ 3 м вперёд относительно начального направления (поза в конце прогрева) без падения;
  * падение: z корпуса < 0,16 м, |roll| > 50°, |pitch| > 50° непрерывно > 0,5 с sim (возврат в норму — сброс).
Строже ТЗ (TODO(уточнить) у организаторов): падение ищется с начала прогрева, а не только в окне; эпизод не обрывается
на 3 м — идёт все 90 с, и отдельно отмечается «падение после 3 м» (success_strict).
Дополнительно (требования T2 к Safety Block, ТЗ_T3 5.2): частота /lowcmd по sim time, NaN/Inf, mode, пределы суставов,
шаг цели ≤ 5° между соседними командами; граф: /lowcmd публикует не политика напрямую, на истину никто не подписан.

Запуск (внутри образа сима, где есть unitree_go; см. tools/run_t3_with_organizers_sim.sh):
  python3 judge_t3.py --out DIR [--speed 0.3] [--window 90] [--start first_lowcmd|publisher]
Код выхода: 0 — успех эпизода, 1 — неуспех, 2 — стенд не дождался сима/решения.
"""
import argparse
import csv
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from unitree_go.msg import LowCmd

# ---- ТЗ_T3 5.3
WARMUP_S = 1.5
WINDOW_S = 90.0
GOAL_M = 3.0
FALL_Z = 0.16
FALL_ANG = math.radians(50.0)
FALL_HOLD = 0.5
# ---- ТЗ_T2 (Safety Block обязан им соответствовать, ТЗ_T3 5.2)
RATE_MIN_HZ = 50.0          # средняя частота за 5 с sim
GAP_MAX_S = 0.100           # макс. интервал /lowcmd
STEP_MAX_DEG = 5.0          # шаг цели сустава между соседними командами
Q_MIN = np.array([-0.863, -0.686, -2.818] * 4)
Q_MAX = np.array([0.863, 4.501, -0.888] * 4)
LIM_EPS = 1e-6


def rpy(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Judge(Node):
    def __init__(self, a):
        super().__init__('judge_t3_stand')
        self.a = a
        self.lock = threading.Lock()
        self.clock = None            # последний sim time
        self.clock_wall = None       # wall, когда он пришёл
        self.gt = []                 # (t_sim, x, y, z, qw, qx, qy, qz)
        self.cmd_t = []              # (t_sim_est, t_wall)
        self.cmd_q = []              # q[0:12]
        self.bare_q = None           # последняя /lowcmd_bare (вход Safety Block) — для «изменено блоком»
        self.bare_n = 0
        self.cmp = []                # (t_sim, max|q_out - q_bare|) для каждой /lowcmd после первой /lowcmd_bare
        self.bad_nan = 0
        self.bad_mode = 0
        self.kp_rng = [math.inf, -math.inf]
        self.kd_rng = [math.inf, -math.inf]
        self.create_subscription(Clock, '/clock', self.on_clock, QoSProfile(depth=50))
        self.create_subscription(Odometry, '/odom/ground_truth', self.on_gt, QoSProfile(depth=200))
        self.create_subscription(LowCmd, '/lowcmd', self.on_cmd, QoSProfile(depth=500))
        self.create_subscription(LowCmd, '/lowcmd_bare', self.on_bare, QoSProfile(depth=500))
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Twist, '/cmd_vel', qos)

    def sim_now(self):
        with self.lock:
            if self.clock is None:
                return None
            # /clock идёт 100 Гц по wall сима (RTF≈1): уточняем внутри такта, не дальше 10 мс
            return self.clock + min(time.monotonic() - self.clock_wall, 0.010)

    def on_clock(self, m):
        with self.lock:
            self.clock = m.clock.sec + m.clock.nanosec * 1e-9
            self.clock_wall = time.monotonic()

    def on_gt(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.gt.append((t, p.x, p.y, p.z, o.w, o.x, o.y, o.z))

    def on_bare(self, m):
        q = np.array([m.motor_cmd[i].q for i in range(12)])
        self.bare_q = q if self.bare_q is None else np.vstack([self.bare_q, q])[-5:]   # 5 последних входов
        self.bare_n += 1

    def on_cmd(self, m):
        ts = self.sim_now()
        self.cmd_t.append((ts if ts is not None else float('nan'), time.monotonic()))
        q = np.empty(12)
        for i in range(12):
            c = m.motor_cmd[i]
            vals = (c.q, c.dq, c.tau, c.kp, c.kd)
            if not all(math.isfinite(v) for v in vals):
                self.bad_nan += 1
            if c.mode != 0x0A:
                self.bad_mode += 1
            q[i] = c.q
            self.kp_rng = [min(self.kp_rng[0], c.kp), max(self.kp_rng[1], c.kp)]
            self.kd_rng = [min(self.kd_rng[0], c.kd), max(self.kd_rng[1], c.kd)]
        self.cmd_q.append(q)
        if self.bare_q is not None:
            self.cmp.append((self.cmd_t[-1][0], float(np.abs(np.atleast_2d(self.bare_q) - q).max(axis=1).min())))

    def graph(self):
        def nodes(infos):
            return sorted({f'{i.node_namespace.rstrip("/")}/{i.node_name}' for i in infos} - {'/judge_t3_stand'})
        out = {}
        for t in ('/odom/ground_truth', '/cmd_vel', '/lowcmd', '/lowcmd_bare', '/lowstate'):
            out[t] = {'pub': nodes(self.get_publishers_info_by_topic(t)),
                      'sub': nodes(self.get_subscriptions_info_by_topic(t))}
        pubs = self.get_publishers_info_by_topic('/lowcmd')
        out['/lowcmd']['pub_qos'] = [str(p.qos_profile.reliability).split('.')[-1] for p in pubs]
        return out


def windows_min_count(t, lo, hi, w):
    """Минимум числа сообщений в окне [s, s+w) для s с шагом 10 мс в [lo, hi-w]."""
    t = np.sort(t)
    starts = np.arange(lo, hi - w + 1e-9, 0.01)
    if len(starts) == 0:
        return None
    c = np.searchsorted(t, starts + w) - np.searchsorted(t, starts)
    return int(c.min()), float(starts[int(c.argmin())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--speed', type=float, default=0.3, help='preferred_speed_mps из ping')
    ap.add_argument('--window', type=float, default=WINDOW_S)
    ap.add_argument('--start', choices=['first_lowcmd', 'publisher'], default='first_lowcmd',
                    help='когда начинать эпизод: по первой /lowcmd решения или через 3 с после появления издателя')
    ap.add_argument('--wait', type=float, default=240.0, help='сколько ждать сим и решение, с wall')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rclpy.init()
    j = Judge(a)
    ex = SingleThreadedExecutor()
    ex.add_node(j)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    res = {'stand': 'judge_t3.py', 'speed': a.speed, 'warmup_s': WARMUP_S, 'window_s': a.window, 'start': a.start}

    def finish(code, msg):
        res['verdict_note'] = msg
        with open(os.path.join(a.out, 'result.json'), 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(msg, flush=True)
        sys.stdout.flush()
        os._exit(code)          # исполнитель крутится в потоке-демоне: обычный выход роняет rclpy (terminate, rc 139)

    w0 = time.monotonic()
    while j.sim_now() is None or len(j.gt) < 5:
        if time.monotonic() - w0 > a.wait:
            finish(2, 'СТЕНД: нет /clock или /odom/ground_truth')
        time.sleep(0.05)
    print(f'сим жив: t={j.sim_now():.2f}', flush=True)
    while not j.get_publishers_info_by_topic('/lowcmd'):
        if time.monotonic() - w0 > a.wait:
            finish(2, 'СТЕНД: решение не создало издателя /lowcmd')
        time.sleep(0.1)
    t_pub = time.monotonic()
    print(f'издатель /lowcmd появился через {t_pub - w0:.1f} с wall', flush=True)
    if a.start == 'first_lowcmd':
        while not j.cmd_t and time.monotonic() - t_pub < 30.0:
            time.sleep(0.02)
        if not j.cmd_t:
            print('/lowcmd до /cmd_vel не идёт 30 с — начинаю эпизод всё равно', flush=True)
    else:
        time.sleep(3.0)
    res['lowcmd_before_cmdvel'] = len(j.cmd_t)

    tw = Twist()
    tw.linear.x = float(a.speed)
    t0 = j.sim_now()
    res['t0_sim'] = t0
    t_end = t0 + WARMUP_S + a.window
    print(f'эпизод: /cmd_vel x={a.speed} с t0={t0:.3f}, конец {t_end:.3f} (sim)', flush=True)
    last_clock_wall = time.monotonic()
    last_clock = j.sim_now()
    next_print = t0 + 10
    graph_done = False
    while True:
        j.pub.publish(tw)                   # TRANSIENT_LOCAL + 20 Гц: дойдёт при любой QoS подписчика
        time.sleep(0.05)
        now = j.sim_now()
        if now != last_clock:
            last_clock, last_clock_wall = now, time.monotonic()
        elif time.monotonic() - last_clock_wall > 20.0:
            res['error'] = 'сим остановил /clock'
            break
        if not graph_done and now > t0 + 10:
            res['graph'] = j.graph()
            graph_done = True
        if now >= next_print and j.gt:
            g = j.gt[-1]
            print(f'  t+{now - t0:5.1f} x={g[1]:.2f} y={g[2]:.2f} z={g[3]:.3f}  /lowcmd {len(j.cmd_t)}', flush=True)
            next_print += 10
        if now >= t_end:
            break
    tw.linear.x = 0.0
    for _ in range(3):
        j.pub.publish(tw)
        time.sleep(0.02)
    if not graph_done:
        res['graph'] = j.graph()

    # ---------------- оценка по ТЗ
    gt = np.array(sorted(j.gt))
    gt = gt[(gt[:, 0] >= t0 - 1e-9) & (gt[:, 0] <= t_end + 1e-9)]
    t = gt[:, 0]
    ang = np.array([rpy(*r[4:8]) for r in gt])
    roll, pitch, yaw = ang[:, 0], ang[:, 1], ang[:, 2]
    z = gt[:, 3]
    bad = (z < FALL_Z) | (np.abs(roll) > FALL_ANG) | (np.abs(pitch) > FALL_ANG)
    fall_onset = None
    run_start = None
    for i in range(len(t)):
        if bad[i]:
            run_start = t[i] if run_start is None else run_start
            if t[i] - run_start > FALL_HOLD:
                fall_onset = run_start
                break
        else:
            run_start = None
    iw = int(np.searchsorted(t, t0 + WARMUP_S - 1e-9))
    c, s = math.cos(yaw[iw]), math.sin(yaw[iw])
    d = gt[:, 1:3] - gt[iw, 1:3]
    along, lateral = d @ [c, s], d @ [-s, c]
    win = np.arange(iw, len(t))
    hit = win[along[win] >= GOAL_M]
    i3 = int(hit[0]) if len(hit) else None
    t3 = None if i3 is None else float(t[i3] - t[iw])
    success = i3 is not None and (fall_onset is None or t[i3] < fall_onset)
    res.update({
        'success': bool(success),
        'success_strict': bool(i3 is not None and fall_onset is None),
        't_3m_s': t3,
        'fall': fall_onset is not None,
        'fall_onset_s': None if fall_onset is None else float(fall_onset - t[iw]),
        'path_90s_m': float(along[-1]),
        'along_max_m': float(along[win].max()),
        'lateral_end_m': float(lateral[-1]),
        'heading_change_end_deg': float(math.degrees((yaw[-1] - yaw[iw] + math.pi) % (2 * math.pi) - math.pi)),
        'roll_max_deg': float(np.degrees(np.abs(roll).max())),
        'pitch_max_deg': float(np.degrees(np.abs(pitch).max())),
        'z_min_m': float(z.min()),
        'z_mean_m': float(z[iw:].mean()),
        'gt_samples': int(len(t)),
        'gt_last_t_rel': float(t[-1] - t0),
        'sim_window_covered_s': float(t[-1] - t[iw]),
    })
    # ---------------- /lowcmd (T2-требования к Safety Block)
    n = min(len(j.cmd_t), len(j.cmd_q))      # исполнитель ещё крутится — фиксируем снимок
    ct = np.array([x[0] for x in j.cmd_t[:n]])
    cw = np.array([x[1] for x in j.cmd_t[:n]])
    m = (ct >= t0) & (ct <= t_end)
    te = ct[m]
    q = np.array(j.cmd_q[:n])
    lc = {'count_total': int(len(ct)), 'count_episode': int(m.sum())}
    if len(te) > 2:
        gaps = np.diff(te)
        mn1 = windows_min_count(te, t0, t_end, 1.0)
        mn5 = windows_min_count(te, t0, t_end, 5.0)
        lc.update({
            'mean_hz': float((len(te) - 1) / (te[-1] - te[0])),
            'min_hz_1s': mn1[0] if mn1 else None, 'min_1s_at_rel': (mn1[1] - t0) if mn1 else None,
            'min_hz_5s_mean': (mn5[0] / 5.0) if mn5 else None,
            'max_gap_ms_sim': float(gaps.max() * 1000), 'max_gap_at_rel': float(te[int(gaps.argmax())] - t0),
            'gaps_over_100ms': int((gaps > GAP_MAX_S).sum()),
            'first_gap_ms_from_t0': float((te[0] - t0) * 1000),
            'last_gap_ms_to_end': float((t_end - te[-1]) * 1000),
            'max_gap_ms_wall': float(np.diff(cw[m]).max() * 1000),
        })
    lc['nan_inf_msgs_fields'] = int(j.bad_nan)
    lc['mode_not_0x0A'] = int(j.bad_mode)
    if len(q):
        qe = q[m]
        lc['out_of_limits'] = int(((qe < Q_MIN - LIM_EPS) | (qe > Q_MAX + LIM_EPS)).sum())
        if len(qe) > 1:
            st = np.degrees(np.abs(np.diff(qe, axis=0)))
            lc['max_step_deg'] = float(st.max())
            lc['steps_over_5deg'] = int((st > STEP_MAX_DEG + 1e-6).sum())
    lc['kp_range'] = j.kp_rng
    lc['kd_range'] = j.kd_rng
    cmp = np.array(j.cmp[:]) if j.cmp else np.zeros((0, 2))
    cmp = cmp[(cmp[:, 0] >= t0) & (cmp[:, 0] <= t_end)] if len(cmp) else cmp
    if len(cmp):
        dd = np.degrees(cmp[:, 1])
        # /lowcmd сравнивается с ближайшей из 5 последних /lowcmd_bare (порядок доставки в стенд и задержка блока);
        # не совпала ни с одной — блок изменил цель (клип шага/предела, коллизия, удержание)
        lc['vs_bare'] = {'bare_msgs': int(j.bare_n), 'compared': int(len(dd)), 'differs_gt_0p01deg': int((dd > 0.01).sum()),
                         'differs_gt_1deg': int((dd > 1.0).sum()), 'max_diff_deg': float(dd.max()),
                         'p99_diff_deg': float(np.percentile(dd, 99)),
                         'differs_at_rel_s': [round(float(x - t0), 3) for x in cmp[dd > 0.01, 0][:60]],
                         'differs_deg': [round(float(x), 2) for x in dd[dd > 0.01][:60]]}
    ok_rate = lc.get('min_hz_5s_mean', 0) and lc['min_hz_5s_mean'] >= RATE_MIN_HZ and lc['max_gap_ms_sim'] <= GAP_MAX_S * 1000
    lc['t2_rate_ok'] = bool(ok_rate)
    res['lowcmd'] = lc

    # ---------------- граф: запрещённое
    g = res.get('graph', {})
    gts = g.get('/odom/ground_truth', {}).get('sub', [])
    res['forbidden'] = {
        'gt_subscribers_other_than_stand': gts,
        'lowcmd_publishers': g.get('/lowcmd', {}).get('pub', []),
        'lowcmd_bare_pub': g.get('/lowcmd_bare', {}).get('pub', []),
        'lowcmd_bare_sub': g.get('/lowcmd_bare', {}).get('sub', []),
        'cmd_vel_publishers_other': g.get('/cmd_vel', {}).get('pub', []),
    }

    with open(os.path.join(a.out, 'gt.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_rel', 'x', 'y', 'z', 'roll_deg', 'pitch_deg', 'yaw_deg', 'along', 'lateral'])
        for i in range(len(t)):
            w.writerow([f'{t[i] - t0:.3f}', f'{gt[i, 1]:.4f}', f'{gt[i, 2]:.4f}', f'{z[i]:.4f}',
                        f'{math.degrees(roll[i]):.2f}', f'{math.degrees(pitch[i]):.2f}', f'{math.degrees(yaw[i]):.2f}',
                        f'{along[i]:.4f}', f'{lateral[i]:.4f}'])
    np.savetxt(os.path.join(a.out, 'lowcmd_times.csv'), np.c_[ct - t0, cw - cw[0] if len(cw) else cw],
               delimiter=',', header='t_sim_rel,t_wall', fmt='%.4f')
    msg = (f"ЭПИЗОД: {'УСПЕХ' if success else 'НЕУСПЕХ'}  t3м={t3 if t3 is None else round(t3, 2)} с  "
           f"путь90={res['path_90s_m']:.2f} м  падение={res['fall']}  roll/pitch max={res['roll_max_deg']:.1f}/"
           f"{res['pitch_max_deg']:.1f}°  zmin={res['z_min_m']:.3f}  /lowcmd min1s={lc.get('min_hz_1s')} "
           f"maxgap={lc.get('max_gap_ms_sim', float('nan')):.1f} мс NaN={lc['nan_inf_msgs_fields']}  GT-подписчики={gts}")
    finish(0 if success else 1, msg)


if __name__ == '__main__':
    main()
