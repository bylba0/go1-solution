"""PPO training of the Go1 walking policy on CPU.

python -m training.train --config training/configs/walk.json --out training/runs/NAME --workers K [--seed S]
                         [--resume CHECKPOINT] [--iterations N] [--minutes M]
The model directory (with scene.xml) is taken from --model-dir or GO1_MODEL_DIR.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import copy
import csv
import json
import multiprocessing as mp
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from training.env import Go1Env
from training.export import export, sha256
from training.ppo import PPO, ActorCritic


def worker(conn, env_cfg, model_dir, seeds):
    envs = [Go1Env(env_cfg, model_dir, s) for s in seeds]
    stats = [dict(ret=0.0, len=0, vel_err=0.0) for _ in envs]
    first = [e.reset() for e in envs]
    conn.send((np.stack([o for o, _ in first]), np.stack([c for _, c in first])))
    while True:
        actions = conn.recv()
        if actions is None:
            return
        out = {k: [] for k in ('obs', 'cobs', 'rew', 'done', 'trunc_cobs', 'trunc')}
        episodes = []
        for e, a, s in zip(envs, actions, stats):
            obs, cobs, rew, term, trunc, info = e.step(a)
            s['ret'] += rew
            s['len'] += 1
            s['vel_err'] += info['vel_err']
            out['trunc_cobs'].append(cobs)
            out['trunc'].append(trunc and not term)
            if term or trunc:
                episodes.append(dict(ret=s['ret'], len=s['len'], vel_err=s['vel_err'] / s['len'],
                                     fell=info['fell'], self_collision=info['self_collision']))
                s.update(ret=0.0, len=0, vel_err=0.0)
                obs, cobs = e.reset()
            out['obs'].append(obs)
            out['cobs'].append(cobs)
            out['rew'].append(rew)
            out['done'].append(term or trunc)
        conn.send(({k: np.array(v) for k, v in out.items()}, episodes))


class VecEnv:
    def __init__(self, env_cfg, model_dir, workers, per_worker, seed):
        ctx = mp.get_context('spawn')
        self.conns, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            seeds = [seed * 100003 + w * per_worker + i for i in range(per_worker)]
            p = ctx.Process(target=worker, args=(b, env_cfg, model_dir, seeds), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        first = [c.recv() for c in self.conns]
        self.obs = np.concatenate([o for o, _ in first])
        self.cobs = np.concatenate([c for _, c in first])
        self.per_worker = per_worker

    def step(self, actions):
        for i, c in enumerate(self.conns):
            c.send(actions[i * self.per_worker:(i + 1) * self.per_worker])
        res = [c.recv() for c in self.conns]
        out = {k: np.concatenate([r[0][k] for r in res]) for k in res[0][0]}
        return out, [ep for r in res for ep in r[1]]

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=5)


def code_commit():
    """Commit of the training code: CODE_COMMIT if set (copies without git), else the repository HEAD."""
    if os.environ.get('CODE_COMMIT'):
        return os.environ['CODE_COMMIT']
    try:
        root = Path(__file__).resolve().parents[1]
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', 'training'], cwd=root, capture_output=True, text=True).stdout
        return head + ('-dirty' if dirty.strip() else '')
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume')
    ap.add_argument('--iterations', type=int, help='overrides ppo.iterations')
    ap.add_argument('--minutes', type=float, help='stop after this wall time')
    ap.add_argument('--model-dir', default=os.environ.get('GO1_MODEL_DIR'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device of the PPO update; rollouts always run on CPU')
    args = ap.parse_args()
    if not args.model_dir or not os.path.isfile(os.path.join(args.model_dir, 'scene.xml')):
        sys.exit('model directory with scene.xml is required: --model-dir or GO1_MODEL_DIR')

    cfg = json.load(open(args.config))
    pc, ec = cfg['ppo'], cfg['env']
    iterations = args.iterations or pc['iterations']
    out = Path(args.out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / 'config.json')
    json.dump({'seed': args.seed, 'workers': args.workers, 'resume': Path(args.resume).name if args.resume else None,
               'python': platform.python_version(), 'mujoco': mujoco.__version__, 'numpy': np.__version__,
               'torch': torch.__version__, 'system': platform.system(), 'machine': platform.machine(),
               'device': args.device, 'commit': code_commit()},
              open(out / 'run.json', 'w'), indent=1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.workers))
    dev = torch.device(args.device)
    probe = Go1Env(ec, args.model_dir)
    scale = probe.obs_scale(pc['obs_scales']) if pc.get('obs_scales') else None
    ac = ActorCritic(probe.obs_dim, probe.critic_dim, 12, pc['actor_hidden'], pc['critic_hidden'], pc['init_std'],
                     obs_scale=scale, max_std=pc.get('max_std')).to(dev)
    ppo = PPO(ac, pc)
    start_it, total_steps, elapsed0 = 0, 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        ac.load_state_dict(ck['model'])
        ppo.opt.load_state_dict(ck['optimizer'])
        ppo.lr = ck['lr']
        start_it, total_steps, elapsed0 = ck['iteration'], ck['steps'], ck['elapsed']
    rollout = copy.deepcopy(ac).cpu() if dev.type != 'cpu' else ac  # CPU copy for acting, synced after updates

    venv = VecEnv(ec, args.model_dir, args.workers, pc['envs_per_worker'], args.seed + start_it)
    n_env, T = args.workers * pc['envs_per_worker'], pc['steps_per_iter']
    gamma = pc['gamma']
    fields = ['iteration', 'steps', 'time_s', 'mean_return', 'episode_len', 'vel_err', 'fall_rate',
              'self_collision_rate', 'episodes', 'kl', 'lr', 'action_std']
    new_curve = not (out / 'curve.csv').exists()
    curve = open(out / 'curve.csv', 'a', newline='')
    writer = csv.DictWriter(curve, fieldnames=fields)
    if new_curve:
        writer.writeheader()
    t0 = time.time()
    obs, cobs = torch.tensor(venv.obs, dtype=torch.float32), torch.tensor(venv.cobs, dtype=torch.float32)

    def save(it, name):
        path = out / 'checkpoints' / name
        torch.save({'model': ac.state_dict(), 'optimizer': ppo.opt.state_dict(), 'lr': ppo.lr, 'iteration': it,
                    'steps': total_steps, 'elapsed': elapsed0 + time.time() - t0, 'config': cfg}, path)
        with open(out / 'checkpoints' / 'sha256.txt', 'a') as f:
            f.write(f'{sha256(path)}  {name}\n')

    it = start_it
    for it in range(start_it + 1, start_it + iterations + 1):
        buf = {k: [] for k in ('obs', 'cobs', 'act', 'logp', 'val', 'rew', 'done')}
        episodes = []
        with torch.no_grad():
            for _ in range(T):
                d = rollout.dist(obs)
                act = d.sample()
                buf['obs'].append(obs)
                buf['cobs'].append(cobs)
                buf['act'].append(act)
                buf['logp'].append(d.log_prob(act).sum(-1))
                buf['val'].append(rollout.value(cobs))
                res, eps = venv.step(act.numpy().astype(np.float64))
                episodes += eps
                rew = torch.tensor(res['rew'], dtype=torch.float32)
                trunc = torch.tensor(res['trunc'])
                if trunc.any():  # time limit: bootstrap from the value of the last state
                    tv = rollout.value(torch.tensor(res['trunc_cobs'][res['trunc']], dtype=torch.float32))
                    rew[trunc] += gamma * tv
                buf['rew'].append(rew)
                buf['done'].append(torch.tensor(res['done'], dtype=torch.float32))
                obs = torch.tensor(res['obs'], dtype=torch.float32)
                cobs = torch.tensor(res['cobs'], dtype=torch.float32)
            last_val = rollout.value(cobs)
        b = {k: torch.stack(v).to(dev) for k, v in buf.items()}
        adv, ret = ppo.gae(b['rew'], b['val'], b['done'], last_val.to(dev))
        st = ppo.update(b['obs'].flatten(0, 1), b['cobs'].flatten(0, 1), b['act'].flatten(0, 1),
                        b['logp'].flatten(0, 1), adv.flatten(), ret.flatten())
        # after the update, so that the update sees the normalization the rollout was collected with
        ac.obs_norm.update(b['obs'].flatten(0, 1))
        ac.critic_norm.update(b['cobs'].flatten(0, 1))
        if rollout is not ac:
            rollout.load_state_dict(ac.state_dict())
        total_steps += n_env * T
        row = {'iteration': it, 'steps': total_steps, 'time_s': round(elapsed0 + time.time() - t0, 1),
               'episodes': len(episodes), 'kl': round(st['kl'], 5), 'lr': f'{ppo.lr:.2e}',
               'action_std': round(ac.std().mean().item(), 4)}
        if episodes:
            mean = lambda k: float(np.mean([e[k] for e in episodes]))
            row.update(mean_return=round(mean('ret'), 4), episode_len=round(mean('len'), 1),
                       vel_err=round(mean('vel_err'), 4), fall_rate=round(mean('fell'), 4),
                       self_collision_rate=round(mean('self_collision'), 4))
        writer.writerow(row)
        curve.flush()
        if it % 10 == 0:
            print(' '.join(f'{k}={row.get(k, "")}' for k in fields), flush=True)
        if it % pc['checkpoint_every'] == 0:
            save(it, f'ckpt_{it:06d}.pt')
        if args.minutes and time.time() - t0 > args.minutes * 60:
            break
    save(it, 'final.pt')
    export(ac.state_dict(), ec, out / 'policy.npz')
    venv.close()
    print(f'done: {it} iterations, {total_steps} steps, policy {out / "policy.npz"}')


if __name__ == '__main__':
    main()
