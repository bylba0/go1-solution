"""Checkpoint -> policy.npz for the numpy controller (weights, normalization, observation and action parameters).

python -m training.export CHECKPOINT OUT.npz
"""
import hashlib
import sys

import numpy as np
import torch

from training import env as E


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def export(state_dict, env_cfg, path):
    sd = {k: v.cpu().numpy() for k, v in state_dict.items()}
    layers = sorted({int(k.split('.')[1]) for k in sd if k.startswith('actor.')})
    out = {}
    for i, n in enumerate(layers):
        out[f'w{i}'] = sd[f'actor.{n}.weight'].astype(np.float64)
        out[f'b{i}'] = sd[f'actor.{n}.bias'].astype(np.float64)
    lo, hi = E.target_box()
    out.update(
        n_layers=np.array(len(layers)),
        obs_mean=sd['obs_norm.mean'].astype(np.float64),
        obs_std=np.sqrt(sd['obs_norm.var'].astype(np.float64) + 1e-8),
        obs_clip=np.array(10.0),
        history=np.array(env_cfg['history']),
        q0=E.STAND_Q, action_scale=np.tile(env_cfg['action_scale'], 4), action_clip=np.array(env_cfg['action_clip']),
        target_lo=lo, target_hi=hi, max_msg_step=np.array(E.MAX_MSG_STEP),
        kp=np.array(env_cfg['kp']), kd=np.array(env_cfg['kd']),
        policy_ms=np.array(E.POLICY_MS), msg_ms=np.array(E.MSG_MS),
    )
    gait = env_cfg.get('gait')
    if gait:
        out.update(gait_freq=np.array(gait['freq']), gait_calf=np.array(gait['calf']),
                   gait_thigh_max=np.array(gait['thigh_max']))
    np.savez(path, **out)
    with open(str(path) + '.sha256', 'w') as f:
        f.write(f'{sha256(path)}  {path.name if hasattr(path, "name") else path}\n')


if __name__ == '__main__':
    from pathlib import Path
    ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    export(ckpt['model'], ckpt['config']['env'], Path(sys.argv[2]))
