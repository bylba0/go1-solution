import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from training import env as E

CFG = json.load(open(Path(__file__).resolve().parents[1] / 'configs/walk.json'))['env']


def reference_motor_torque():
    """Simulator motor law: core.motor_torque from GO1_SIM_DIR if available, else its transcription
    (go1_ws/src/go1_unitree_bridge/go1_unitree_bridge/core.py, lines 41-54)."""
    sim = os.environ.get('GO1_SIM_DIR')
    path = Path(sim or '.') / 'go1_ws/src/go1_unitree_bridge/go1_unitree_bridge/core.py'
    if sim and path.is_file():
        spec = importlib.util.spec_from_file_location('sim_core', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return lambda q_des, q, dq, kp, kd: mod.motor_torque(np.full(12, 0x0A), q_des, np.zeros(12), np.zeros(12),
                                                            kp, kd, q, dq)

    def transcribed(q_des, q, dq, kp, kd):
        tau = 0.0 + np.where(np.abs(q_des) < 1e8, kp * (q_des - q), 0.0) + kd * (0.0 - dq)
        return np.clip(tau, -E.TORQUE_LIMIT, E.TORQUE_LIMIT)
    return transcribed


def test_pd_matches_simulator_law_bitwise():
    ref = reference_motor_torque()
    rng = np.random.default_rng(0)
    for _ in range(2000):
        q_des, q = rng.uniform(-3, 3, 12), rng.uniform(-3, 3, 12)
        dq = rng.normal(0, 10, 12)
        kp, kd = np.full(12, rng.uniform(0, 80)), np.full(12, rng.uniform(0, 5))
        assert np.array_equal(E.pd_torque(q_des, q, dq, kp, kd), ref(q_des, q, dq, kp, kd))


def test_box_is_inside_joint_limits_and_mirrored():
    from go1_safety.core import Q_MAX, Q_MIN
    lo, hi = E.target_box()
    assert np.all(lo > Q_MIN + 0.05) and np.all(hi < Q_MAX - 0.05)
    assert np.all(lo <= E.STAND_Q) and np.all(E.STAND_Q <= hi)
    assert np.allclose(lo[3], -hi[0]) and np.allclose(hi[3], -lo[0])  # FL hip mirrors FR hip


def test_target_box_has_clearance(model_dir):
    """Random poses, corners and edge points of the box keep >= 2 mm clearance over the checked pairs."""
    from go1_safety.collision import CollisionChecker
    # Safety Block checker on the floorless go1.xml (the 655 checked pairs of t2_collision_pairs.csv): min_dist(q) is
    # the signed distance in m (negative = penetration); pairs closer than its 5 cm margin are reported.
    ck = CollisionChecker(os.path.join(model_dir, 'go1.xml'))
    lo, hi = E.target_box()
    rng = np.random.default_rng(1)
    corners = np.array([[lo[i] if (c >> i) & 1 else hi[i] for i in range(12)] for c in range(4096)])
    edges = corners[rng.integers(0, 4096, 5000)]
    j = rng.integers(0, 12, 5000)
    edges[np.arange(5000), j] = lo[j] + rng.random(5000) * (hi[j] - lo[j])
    poses = np.vstack([lo + (hi - lo) * rng.random((20000, 12)), corners, edges])
    worst = min(ck.min_dist(q) for q in poses)
    assert worst >= 0.002, worst


def test_clip_target_keeps_box_and_message_step(model_dir):
    e = E.Go1Env(CFG, model_dir, 0)
    e.reset()
    rng = np.random.default_rng(2)
    prev = e.target.copy()
    for _ in range(200):
        new = e.clip_target(rng.uniform(-10, 10, 12))
        msgs = e.messages(new)
        steps = np.abs(np.diff(np.vstack([prev, msgs]), axis=0))
        assert steps.max() <= E.MAX_MSG_STEP + 1e-6
        assert np.all(new >= e.lo) and np.all(new <= e.hi)
        e.target = prev = new


def test_stand_is_stable_under_zero_action(model_dir):
    cfg = json.loads(json.dumps(CFG))
    cfg['randomize']['enable'] = False
    cfg['noise']['level'] = 0.0
    e = E.Go1Env(cfg, model_dir, 0)
    e.reset()
    for _ in range(150):
        *_, term, trunc, info = e.step(np.zeros(12))
        assert not term
        assert np.abs(e.data.qfrc_applied[6:]).max() < 0.5 * E.TORQUE_LIMIT.min()
    assert e.data.qpos[2] > 0.24


@pytest.mark.parametrize('delay_ms', [0, 4, 10])
def test_trajectory_matches_simulator_node(model_dir, delay_ms):
    """Same commands -> bitwise the same trajectory as the simulator node loop (go1_mujoco.py: reset, hold before
    the first command, last command before every step), with the law from the simulator's core.py."""
    import mujoco
    sim = os.environ.get('GO1_SIM_DIR')
    path = Path(sim or '.') / 'go1_ws/src/go1_unitree_bridge/go1_unitree_bridge/core.py'
    if not path.is_file():
        pytest.skip('GO1_SIM_DIR is not set')
    spec = importlib.util.spec_from_file_location('sim_core', path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    m = mujoco.MjModel.from_xml_path(os.path.join(model_dir, 'scene.xml'))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
    mujoco.mj_forward(m, d)
    m.actuator_gainprm[:, 0] = 0.0
    m.actuator_biasprm[:] = 0.0
    d.ctrl[:] = 0.0
    servo, z = np.full(12, 0x0A), np.zeros(12)
    cmd = None

    def node_step():
        q, dq = d.qpos[7:].copy(), d.qvel[6:].copy()
        if cmd is None:
            tau = core.motor_torque(servo, core.STAND_Q, z, z, np.full(12, 40.0), np.full(12, 2.0), q, dq)
        else:
            tau = core.motor_torque(servo, cmd, z, z, np.full(12, 35.0), np.full(12, 1.0), q, dq)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[6:] = tau
        mujoco.mj_step(m, d)

    cfg = json.loads(json.dumps(CFG))
    cfg.update(kp=35.0, kd=1.0, max_delay_ms=delay_ms)
    cfg['randomize']['enable'] = False
    e = E.Go1Env(cfg, model_dir, 0)
    for _ in range(int(cfg['settle_s'] * 1000) // E.SIM_MS):
        node_step()
    e.reset()
    e.delay_ms = delay_ms
    rng = np.random.default_rng(0)
    sched = {}
    for _ in range(150):
        t0 = e.t_ms
        e.step(rng.normal(0, 1.0, 12))
        sched.update({t0 + E.MSG_MS * j + delay_ms: msg.astype(np.float32) for j, msg in enumerate(e.last_msgs)})
        for i in range(E.POLICY_MS // E.SIM_MS):
            due = sorted(t for t in sched if t <= t0 + E.SIM_MS * i)
            if due:
                cmd = sched[due[-1]].astype(float)
                for t in due:
                    del sched[t]
            node_step()
        assert np.array_equal(e.data.qpos, d.qpos) and np.array_equal(e.data.qvel, d.qvel)


def test_packs_and_drops_keep_limits(model_dir):
    """Catch-up packs and dropped observations: time jumps, targets stay in the box and within the step limit."""
    cfg = json.load(open(Path(__file__).resolve().parents[1] / 'configs/walk_trot.json'))
    ec = cfg['env']
    ec['randomize'].update(enable=True, pack_prob=0.5, obs_drop_prob=0.5)
    e = E.Go1Env(ec, model_dir, 4)
    obs, _ = e.reset()
    scales = dict(gyro=0.25, gravity=1.0, q=1.0, dq=0.05, action=1.0, command=[2.0, 2.0, 0.25], phase=1.0)
    assert e.obs_scale(scales).shape == obs.shape
    rng = np.random.default_rng(5)
    prev, t_prev, jumps = e.target.copy(), e.t_ms, 0
    for _ in range(100):
        e.step(rng.normal(0, 1, 12))
        steps = np.abs(np.diff(np.vstack([prev, e.last_msgs]), axis=0))
        assert steps.max() <= E.MAX_MSG_STEP + 1e-6 and np.all(e.target >= e.lo) and np.all(e.target <= e.hi)
        jumps += e.t_ms - t_prev > E.POLICY_MS
        prev, t_prev = e.target.copy(), e.t_ms
    assert 20 < jumps < 80


TRAINING = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('config', sorted(str(p.relative_to(TRAINING)) for p in
                                          [*TRAINING.glob('configs/*.json'), *TRAINING.glob('logs/*/config.json')]))
def test_repository_configs_run(model_dir, config):
    """Every config in the repository (and of every logged run) builds the environment and the networks and steps."""
    from training.ppo import ActorCritic
    cfg = json.load(open(TRAINING / config))
    e = E.Go1Env(cfg['env'], model_dir, 0)
    obs, cobs = e.reset()
    pc = cfg['ppo']
    scale = e.obs_scale(pc['obs_scales']) if pc.get('obs_scales') else None
    ac = ActorCritic(e.obs_dim, e.critic_dim, 12, pc['actor_hidden'], pc['critic_hidden'], pc['init_std'],
                     obs_scale=scale, max_std=pc.get('max_std'))
    assert ac.std().max().item() <= (pc.get('max_std') or 1e9) + 1e-6
    for _ in range(5):
        obs, cobs, *_ = e.step(np.zeros(12))
    assert obs.shape == (e.obs_dim,) and cobs.shape == (e.critic_dim,)
