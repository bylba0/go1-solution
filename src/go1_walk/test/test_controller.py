import csv
import itertools
import math
import os

import numpy as np
import pytest

from go1_walk.controller import WalkController

POLICY = os.path.join(os.path.dirname(__file__), '..', 'policy', 'policy.npz')


def quat_yaw(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], np.float32)


def state(ctrl, tick, yaw=0.0, q=None):
    return {'quat': quat_yaw(yaw), 'gyro': np.zeros(3, np.float32),
            'q': (ctrl.q0 if q is None else q).astype(np.float32), 'dq': np.zeros(12, np.float32), 'tick': tick}


def run(ctrl, ticks, cmd=(0.3, 0.0, 0.0), bad=()):
    out = []
    for tk in ticks:
        s = state(ctrl, tk)
        if tk in bad:
            s['quat'] = np.array([np.nan, 0, 0, 0], np.float32)
        c = ctrl.on_lowstate(s, cmd)
        if c is not None:
            out.append((tk, c['q'].copy()))
    return out


def test_nan_state_is_skipped_without_poisoning():
    ticks = list(range(1000, 3000, 2))
    bad = {1040, 1500, 2000}                                  # 1040, 1500, 2000: policy ticks (every 20 ms from 1000)
    ref = run(WalkController(POLICY), [t for t in ticks if t not in bad])
    got = run(WalkController(POLICY), ticks, bad=bad)
    assert all(np.isfinite(q).all() for _, q in got)
    assert len(got) == len(ref) and all(a == b and np.array_equal(p, r) for (a, p), (b, r) in zip(got, ref))


def test_target_step_limited_with_actions_at_the_clip():
    ctrl = WalkController(POLICY)
    sign = [1.0]

    def bang(_obs):                                           # the largest possible action changes every policy step
        sign[0] = -sign[0]
        return np.full(12, sign[0] * 10 * ctrl.action_clip)
    ctrl.act = bang
    qs = [q.astype(np.float64) for _, q in run(ctrl, range(1000, 3000, 2))]
    steps = np.abs(np.diff(qs, axis=0)).max(axis=1)
    assert steps.max() <= ctrl.max_msg_step + 1e-6
    assert steps.max() >= 0.9 * ctrl.max_msg_step              # the limit is reached, i.e. it is what bounds the step
    assert all(((q >= ctrl.lo - 1e-6) & (q <= ctrl.hi + 1e-6)).all() for q in qs)


@pytest.mark.parametrize('yaw0, yaw, sign', [(0.0, 0.05, -1), (0.0, -0.05, 1),
                                             (math.pi - 0.02, -math.pi + 0.02, -1),   # +0.04 across +-pi
                                             (-math.pi + 0.02, math.pi - 0.02, 1)])   # -0.04 across +-pi
def test_heading_loop_sign_and_wrap(yaw0, yaw, sign):
    ctrl = WalkController(POLICY, heading_gain=1.0)
    ctrl.yaw0 = yaw0
    wz = ctrl.command(state(ctrl, 0, yaw=yaw), (0.3, 0.0, 0.0))[2]
    err = math.remainder(yaw - yaw0, 2 * math.pi)
    assert np.sign(wz) == sign and wz == pytest.approx(-err, abs=1e-6)       # small error: no clipping


def test_zero_command_stands():
    ctrl = WalkController(POLICY)
    ctrl.yaw0 = 0.0
    assert not ctrl.command(state(ctrl, 0, yaw=0.2), (0.0, 0.0, 0.0)).any()   # no heading correction, no gait
    assert ctrl.command(state(ctrl, 0, yaw=0.2), (0.3, 0.0, 0.0))[2] < 0


def test_target_box_is_collision_free_by_model_sampling():
    """Independent check of the target box stored in policy.npz: MuJoCo distances between the checked pairs of the
    organizers' list over all corners and random poses of the box; a wider box must show contacts (the check sees them)."""
    mujoco = pytest.importorskip('mujoco')
    model_dir = os.environ.get('GO1_MODEL_DIR')
    if not model_dir:
        pytest.skip('GO1_MODEL_DIR not set')
    m = mujoco.MjModel.from_xml_path(os.path.join(model_dir, 'scene.xml'))
    d = mujoco.MjData(m)
    rows = [r for r in csv.DictReader(open(os.path.join(model_dir, 't2_collision_pairs.csv')))
            if r['diagnostic_policy'] == 'checked']
    pairs = np.array([(int(r['geom1_id']), int(r['geom2_id'])) for r in rows])
    ft = np.zeros(6)

    def min_dist(q):
        d.qpos[:] = 0
        d.qpos[2], d.qpos[3] = 1.0, 1.0
        d.qpos[7:] = q
        mujoco.mj_forward(m, d)
        rb = m.geom_rbound
        c = np.linalg.norm(d.geom_xpos[pairs[:, 0]] - d.geom_xpos[pairs[:, 1]], axis=1) - rb[pairs[:, 0]] - rb[pairs[:, 1]]
        return min((mujoco.mj_geomDistance(m, d, int(g1), int(g2), 0.05, ft) for g1, g2 in pairs[c < 0.02]), default=0.05)

    p = np.load(POLICY)
    lo, hi = p['target_lo'], p['target_hi']
    rng = np.random.default_rng(0)
    poses = [lo + (hi - lo) * np.array(c) for c in itertools.product([0, 1], repeat=12)]
    poses += list(lo + (hi - lo) * rng.random((1000, 12)))
    assert min(min_dist(q) for q in poses) > 0.002               # >= 2 mm clearance everywhere in the box
    lo2, hi2 = lo.copy(), hi.copy()
    lo2[2::3], hi2[1::3] = -2.70, 2.5                            # calf and thigh widened towards the joint limits
    assert min(min_dist(lo2 + (hi2 - lo2) * rng.random(12)) for _ in range(500)) < 0


# Target box of the right legs (hip, thigh, calf), rad; left hips mirrored. Verified by the sampling test above.
HIP, THIGH, CALF = (-0.35, 0.275), (0.45, 1.25), (-2.25, -1.175)


def box_arrays(hip, thigh, calf):
    lo, hi = [], []
    for right in (True, False, True, False):
        h = hip if right else (-hip[1], -hip[0])
        lo += [h[0], thigh[0], calf[0]]
        hi += [h[1], thigh[1], calf[1]]
    return np.array(lo), np.array(hi)


def test_target_box_matches_verified_numbers():
    p = np.load(POLICY)
    lo, hi = box_arrays(HIP, THIGH, CALF)
    assert np.allclose(p['target_lo'], lo) and np.allclose(p['target_hi'], hi)


def test_training_box_matches_verified_numbers():
    """The training environment builds targets in the same box (source of the next exported policy)."""
    import ast
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'training', 'env.py')
    if not os.path.exists(path):
        pytest.skip('training sources not present')
    consts = {t.id: ast.literal_eval(n.value) for n in ast.parse(open(path).read()).body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ('HIP', 'THIGH', 'CALF')}
    assert (consts['HIP'], consts['THIGH'], consts['CALF']) == (HIP, THIGH, CALF)


def test_gain_override_reaches_the_command():
    p = np.load(POLICY)
    default = WalkController(POLICY).on_lowstate(state(WalkController(POLICY), 1000), (0.3, 0.0, 0.0))
    assert (default['kp'] == np.float32(p['kp'])).all() and (default['kd'] == np.float32(p['kd'])).all()
    ctrl = WalkController(POLICY, kp=45.0, kd=1.2)
    out = ctrl.on_lowstate(state(ctrl, 1000), (0.3, 0.0, 0.0))
    assert (out['kp'] == np.float32(45.0)).all() and (out['kd'] == np.float32(1.2)).all()


@pytest.mark.parametrize('yaw, wz', [(1.0, -0.3), (-1.0, 0.3)])
def test_heading_loop_is_limited(yaw, wz):
    ctrl = WalkController(POLICY)                              # gain 3: an error of 1 rad asks for 3 rad/s
    ctrl.yaw0 = 0.0
    assert ctrl.command(state(ctrl, 0, yaw=yaw), (0.3, 0.0, 0.0))[2] == pytest.approx(wz, abs=1e-12)


def test_normalized_observation_is_clipped():
    ctrl = WalkController(POLICY)
    edge = ctrl.obs_mean + 2 * ctrl.obs_clip * ctrl.obs_std
    far = ctrl.obs_mean + 1e3 * ctrl.obs_clip * ctrl.obs_std
    assert np.array_equal(ctrl.act(far), ctrl.act(edge))


def test_policy_schedule_catches_up_after_a_state_gap():
    """After 200 ms without /lowstate the next policy step is at the gap end and the 20 ms grid continues;
    there is no burst of policy steps to make up for the missed ones."""
    ctrl = WalkController(POLICY)
    calls = []
    step = ctrl.policy_step
    ctrl.policy_step = lambda ls, cmd, tick: (calls.append(int(ls['tick'])), step(ls, cmd, tick))
    ticks = list(range(1000, 1100, 2)) + list(range(1300, 1400, 2))
    run(ctrl, ticks)
    after = [t for t in calls if t >= 1300]
    assert after[:3] == [1300, 1320, 1340]
    assert all(b - a >= ctrl.policy_ms for a, b in zip(calls, calls[1:]))
