import json
from pathlib import Path

import numpy as np
import pytest
import torch

from go1_walk.controller import WalkController
from training import env as E
from training.export import export
from training.ppo import ActorCritic

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
CFG = json.load(open(CONFIGS / 'walk.json'))


def make_policy(tmp_path, model_dir, config='walk.json'):
    torch.manual_seed(0)
    ec = json.load(open(CONFIGS / config))['env']
    ec['randomize']['enable'] = False
    ec['noise']['level'] = 0.0
    e = E.Go1Env(ec, model_dir, 3)
    ac = ActorCritic(e.obs_dim, e.critic_dim, 12, [64, 32], [64, 32], 0.5)
    ac.obs_norm.update(torch.randn(500, e.obs_dim) * 2 + 0.3)
    path = tmp_path / 'policy.npz'
    export(ac.state_dict(), ec, path)
    return e, ac, path


def test_numpy_forward_matches_torch(tmp_path, model_dir):
    e, ac, path = make_policy(tmp_path, model_dir)
    c = WalkController(path)
    x = torch.randn(50, e.obs_dim) * 3
    with torch.no_grad():
        ref = ac.actor(ac.obs_norm(x)).numpy()
    got = np.stack([c.act(v) for v in x.double().numpy()])
    assert np.abs(got - ref).max() < 1e-5


@pytest.mark.parametrize('config', ['walk.json', 'walk_trot.json'])
def test_controller_reproduces_env_targets(tmp_path, model_dir, config):
    e, ac, path = make_policy(tmp_path, model_dir, config)
    c = WalkController(path, heading_hold=False)
    e.reset()
    e.command = np.array([0.3, 0.0, 0.1])
    e.frames = [e._frame()] * e.history
    e.next_cmd_ms = 10 ** 9
    tick0 = 1000
    for k in range(100):
        s = e.lowstate()
        s['tick'] = tick0 + e.t_ms
        with torch.no_grad():
            action = ac.actor(ac.obs_norm(torch.tensor(e.obs(), dtype=torch.float32))).double().numpy()
        out = [c.on_lowstate(dict(s, tick=s['tick'] + E.MSG_MS * j), tuple(e.command)) for j in range(4)]
        e.step(action)
        got = np.stack([o['q'] for o in out])
        assert np.abs(got - e.last_msgs).max() < 1e-5, k
        assert np.all(out[0]['kp'] == np.float32(e.kp_cmd)) and np.all(out[0]['kd'] == np.float32(e.kd_cmd))
        assert c.on_lowstate(dict(s, tick=s['tick'] + 16), tuple(e.command)) is None


def test_controller_steps_never_exceed_limit_with_irregular_ticks(tmp_path, model_dir):
    e, ac, path = make_policy(tmp_path, model_dir)
    c = WalkController(path)
    e.reset()
    s = e.lowstate()
    rng = np.random.default_rng(0)
    tick, prev = 0, E.STAND_Q.astype(np.float32)
    for _ in range(2000):
        tick += int(rng.choice([1, 2, 3, 4, 13, 200], p=[.2, .4, .2, .1, .08, .02]))
        s['gyro'] = rng.normal(0, 2, 3).astype(np.float32)
        out = c.on_lowstate(dict(s, tick=tick), (0.3, 0.0, 0.0))
        if out is not None:
            assert np.abs(out['q'].astype(np.float64) - prev).max() <= E.MAX_MSG_STEP + 1e-6
            assert np.all(out['q'] >= c.lo - 1e-6) and np.all(out['q'] <= c.hi + 1e-6)
            prev = out['q'].astype(np.float64)
