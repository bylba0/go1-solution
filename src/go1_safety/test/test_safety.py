"""Unit-тесты T2 Safety Block (без ROS). Запуск: pytest src/go1_safety/test -q
Модель берётся из GO1_MODEL_DIR (в образе /opt/go1/models/unitree_go1) или из models/ репозитория."""
import csv
import itertools
import math
import os
import pathlib

import mujoco
import numpy as np
import pytest

from go1_safety.collision import BLOCKED, DONE, CollisionChecker
from go1_safety.core import (HOLD_KD, HOLD_KP, HOME_Q, MAX_STEP_RAD, MODE_SERVO, Q_MAX, Q_MAX_TZ, Q_MIN, Q_MIN_TZ,
                             SafetyCore)

REPO_MODELS = pathlib.Path(__file__).resolve().parents[3] / 'models' / 'unitree_go1'
MODEL_DIR = pathlib.Path(os.environ.get('GO1_MODEL_DIR', REPO_MODELS))
GO1 = str(MODEL_DIR / 'go1.xml')
DEG = math.pi / 180
JUDGE_LIMIT = 0.001          # ТЗ: проникновение строго > 1 мм — нарушение


@pytest.fixture(scope='module')
def checker():
    return CollisionChecker(GO1)


@pytest.fixture(scope='module')
def colliding_poses(checker):
    """Реальные позы go1 с самоколлизией > 3 мм (детерминированный поиск)."""
    rng = np.random.default_rng(7)
    found = []
    for _ in range(20000):
        q = rng.uniform(Q_MIN, Q_MAX)
        if checker.penetration(q) > 0.003:
            found.append(q)
            if len(found) >= 12:
                break
    assert len(found) >= 12, 'не нашли поз с самоколлизией'
    return found


def make_cmd(q12, kp=40.0, kd=2.0, dq=0.0, tau=0.0):
    q = np.zeros(20)
    q[:12] = q12
    f = {'q': q, 'dq': np.zeros(20), 'kp': np.zeros(20), 'kd': np.zeros(20), 'tau': np.zeros(20)}
    f['dq'][:12] = dq
    f['kp'][:12] = kp
    f['kd'][:12] = kd
    f['tau'][:12] = tau
    return f


def run(core, t0, t1, dt=0.005, cmd=None, cmd_every=1):
    """Гоняем такты с шагом dt; cmd подаём каждые cmd_every тактов. Возвращает список выходов."""
    outs, t, k = [], t0, 0
    while t < t1 - 1e-12:
        if cmd is not None and k % cmd_every == 0:
            core.on_command(cmd, t)
        o = core.step(t)
        if o is not None:
            outs.append(o)
        t += dt
        k += 1
    return outs


def check_stream(outs, checker):
    """Критерии Judge для последовательности выходов: пределы, шаг ≤ 5°, коллизии с шагом 0.1°, конечность."""
    prev = None
    worst = 0.0
    for o in outs:
        for a in (o.q, o.dq, o.kp, o.kd, o.tau):
            assert np.all(np.isfinite(a))
        assert np.all(o.q >= Q_MIN_TZ) and np.all(o.q <= Q_MAX_TZ)
        if prev is not None:
            assert np.max(np.abs(o.q - prev)) <= 5 * DEG + 1e-12
            if not np.array_equal(o.q, prev):
                worst = max(worst, checker.sampled_max_penetration(prev, o.q, step_rad=0.1 * DEG))
        prev = o.q
    assert worst <= JUDGE_LIMIT / 2, f'проникновение {worst * 1000:.3f} мм'
    return worst


# ---------------------------------------------------------------- модель и пары
def test_collision_pairs_match_csv():
    scene = mujoco.MjModel.from_xml_path(str(MODEL_DIR / 'scene.xml'))
    m = mujoco.MjModel.from_xml_path(GO1)
    off = scene.ngeom - m.ngeom                         # id в CSV — от scene.xml (geom 0 = пол)
    rows = list(csv.DictReader(open(MODEL_DIR / 't2_collision_pairs.csv')))
    checked = {(int(r['geom1_id']), int(r['geom2_id'])) for r in rows if r['diagnostic_policy'] == 'checked'}
    assert len(rows) == 1540 and len(checked) == 655

    def allowed(g1, g2):
        w1, w2 = m.body_weldid[m.geom_bodyid[g1]], m.body_weldid[m.geom_bodyid[g2]]
        if w1 == w2:
            return False
        p1, p2 = m.body_weldid[m.body_parentid[w1]], m.body_weldid[m.body_parentid[w2]]
        if w1 and w2 and (p1 == w2 or p2 == w1):
            return False
        return bool((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1]))

    mine = {(a + off, b + off) for a, b in itertools.combinations(range(m.ngeom), 2) if allowed(a, b)}
    assert mine == checked


def test_home_is_safe(checker):
    assert checker.penetration(HOME_Q) == 0.0


def test_min_dist_matches_original_model(checker, colliding_poses):
    """Наше расстояние (margin 5 см) совпадает с проникновением по исходной модели (margin 1 мм, как у Judge)."""
    m = mujoco.MjModel.from_xml_path(GO1)
    d = mujoco.MjData(m)
    for q in colliding_poses[:5]:
        d.qpos[checker.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        ref = max(0.0, -float(np.min(d.contact.dist[:d.ncon])))
        assert abs(ref - checker.penetration(q)) < 1e-9


# ---------------------------------------------------------------- гарантия проверки пути
def test_advance_guarantee_dense(checker, colliding_poses):
    """На пройденном куске пути проникновение ≤ 0.5 мм в любой точке (плотная выборка 0.01°)."""
    rng = np.random.default_rng(3)
    blocked = 0
    for target in colliding_poses[:8]:
        start = HOME_Q + rng.uniform(-0.05, 0.05, 12)
        q, status = start, None
        for _ in range(500):                       # как в узле: такты по ≤ 4.5°
            delta = target - q
            frac = min(1.0, MAX_STEP_RAD / float(np.max(np.abs(delta))))
            cand = q + delta * frac
            q_new, status = checker.advance(q, cand)
            assert checker.sampled_max_penetration(q, q_new, step_rad=0.01 * DEG) <= checker.threshold + 1e-9
            q = q_new
            if status != 'partial' and (status == BLOCKED or np.array_equal(q, target)):
                break
        blocked += status == BLOCKED
        assert checker.penetration(q) <= checker.stop_threshold + 1e-9
    assert blocked == 8                              # к позе с коллизией > 3 мм дойти нельзя


def test_advance_free_space_is_cheap(checker):
    q0 = HOME_Q.copy()
    q1 = HOME_Q + np.array([0.0, 0.07, -0.07] * 4)
    checker.evals = 0
    q, status = checker.advance(q0, q1)
    assert status == DONE and np.array_equal(q, q1)
    assert checker.evals <= 10


# ---------------------------------------------------------------- SafetyCore
@pytest.fixture
def core(checker):
    c = SafetyCore(checker)
    c.latch_measured(HOME_Q)
    return c


def test_startup_holds_latched_pose(core):
    o = core.step(0.0)
    assert np.allclose(o.q, HOME_Q) and o.holding
    assert np.all(o.kp == HOLD_KP) and np.all(o.kd == HOLD_KD) and np.all(o.dq == 0) and np.all(o.tau == 0)


def test_no_output_before_any_pose(checker):
    c = SafetyCore(checker)
    assert c.step(0.0) is None


def test_limits_clipped(core, checker):
    bad = HOME_Q.copy()
    bad[0], bad[1], bad[2] = 2.0, 6.0, 0.5           # hip, thigh, calf вне диапазонов
    outs = run(core, 0.0, 2.0, cmd=make_cmd(bad))
    check_stream(outs, checker)
    assert core.stats.clipped_q > 0
    assert np.all(outs[-1].q <= Q_MAX + 1e-12) and np.all(outs[-1].q >= Q_MIN - 1e-12)


def test_jump_30deg_ramped(core, checker):
    tgt = HOME_Q.copy()
    tgt[1] += 30 * DEG
    tgt[4] += 30 * DEG
    outs = run(core, 0.0, 0.5, cmd=make_cmd(tgt))
    check_stream(outs, checker)
    assert np.allclose(outs[-1].q, tgt)              # дошли (не «вечное удержание»)
    steps = [np.max(np.abs(b.q - a.q)) for a, b in zip(outs, outs[1:])]
    assert max(steps) <= MAX_STEP_RAD + 1e-12


def test_safe_trajectory_tracked(core, checker):
    outs, t = [], 0.0
    for k in range(400):                              # 2 с, синус по бёдрам и голеням
        q = HOME_Q + np.array([0.0, 0.25, -0.3] * 4) * math.sin(2 * math.pi * 0.5 * t)
        core.on_command(make_cmd(q), t)
        o = core.step(t)
        outs.append(o)
        assert np.max(np.abs(o.q - q)) < 1e-9        # медленная безопасная траектория отрабатывается точно
        t += 0.005
    check_stream(outs, checker)


def test_self_collision_target_not_reached_and_recovers(core, checker, colliding_poses):
    outs = run(core, 0.0, 1.5, cmd=make_cmd(colliding_poses[0]))
    check_stream(outs, checker)
    assert core.blocked
    assert checker.penetration(outs[-1].q) <= checker.stop_threshold
    outs2 = run(core, 1.5, 3.0, cmd=make_cmd(HOME_Q))   # безопасная команда после опасной — отрабатывается
    check_stream(outs + outs2, checker)
    assert np.allclose(outs2[-1].q, HOME_Q)


@pytest.mark.parametrize('field', ['q', 'dq', 'kp', 'kd', 'tau'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
@pytest.mark.parametrize('slot', [0, 7, 11, 15])
def test_nonfinite_rejected(core, field, bad, slot):
    good = make_cmd(HOME_Q + np.array([0.0, 0.1, -0.1] * 4))
    run(core, 0.0, 0.2, cmd=good)
    before = core.goal.copy()
    f = {k: v.copy() for k, v in make_cmd(HOME_Q + 0.2).items()}
    f[field][slot] = bad
    assert core.on_command(f, 0.2) is False
    o = core.step(0.205)
    assert np.array_equal(core.goal, before)
    for a in (o.q, o.dq, o.kp, o.kd, o.tau):
        assert np.all(np.isfinite(a))


def test_wrong_shape_rejected(core):
    f = make_cmd(HOME_Q)
    f['q'] = f['q'][:12]
    assert core.on_command(f, 0.0) is False


def test_input_loss_hold(core, checker):
    tgt = HOME_Q + np.array([0.0, 0.2, -0.2] * 4)
    run(core, 0.0, 1.0, cmd=make_cmd(tgt, kp=60.0, kd=3.0, dq=0.5, tau=1.0))
    o = core.step(1.0)
    assert not o.holding and np.all(o.tau == 1.0)
    outs = run(core, 1.0, 1.6)                       # вход пропал
    late = [o for o in outs if o.holding]
    assert late, 'нет удержания'
    first_hold_t = 1.0 + 0.005 * next(i for i, o in enumerate(outs) if o.holding)
    assert first_hold_t - 1.0 <= 0.5                 # ТЗ: не позднее 0.5 с
    for o in late:
        assert np.allclose(o.q, tgt) and np.all(o.dq == 0) and np.all(o.tau == 0)
        assert np.all(o.kp == 60.0) and np.all(o.kd == 3.0)
    run(core, 1.6, 1.7, cmd=make_cmd(HOME_Q))        # вход вернулся — снова отрабатываем
    assert not core.step(1.7).holding


def test_hold_with_zero_kp_uses_default_gains(core):
    run(core, 0.0, 0.2, cmd=make_cmd(HOME_Q, kp=0.0, kd=0.0))
    o = run(core, 0.2, 1.0)[-1]
    assert o.holding and np.all(o.kp == HOLD_KP) and np.all(o.kd == HOLD_KD)


def test_gains_and_ff_clipped(core):
    run(core, 0.0, 0.1, cmd=make_cmd(HOME_Q, kp=1e6, kd=-5.0, dq=1e3, tau=1e3))
    o = core.step(0.1)
    assert np.all(o.kp <= 100) and np.all(o.kd >= 0) and np.all(np.abs(o.dq) <= 20)
    assert np.all(np.abs(o.tau) <= np.array([23.7, 23.7, 35.55] * 4) + 1e-9)


def test_mode_constant():
    assert MODE_SERVO == 10
