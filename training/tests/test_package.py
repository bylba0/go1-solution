"""Evidence package: the checkpoint kept in training/logs/<run>/ (one *.pt) exports bitwise to the policy recorded next
to it, and the policy shipped in src/go1_walk/ is one of them."""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from training.export import export

ROOT = Path(__file__).resolve().parents[2]
RUNS = sorted({p.parent for p in (ROOT / 'training/logs').glob('*/*.pt')})


def recorded(path):
    return path.read_text().split()[0]


@pytest.mark.parametrize('run', RUNS, ids=[r.name for r in RUNS])
def test_checkpoint_exports_to_recorded_policy(tmp_path, run):
    (path,) = run.glob('*.pt')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    out = tmp_path / 'policy.npz'
    export(ck['model'], ck['config']['env'], out)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == recorded(run / 'policy.npz.sha256')
    assert ck['config'] == __import__('json').load(open(run / 'config.json'))


def test_shipped_policy_comes_from_a_logged_run(tmp_path):
    """The shipped policy.npz is byte for byte the export of the checkpoint of exactly one logged run."""
    shipped = ROOT / 'src/go1_walk/policy/policy.npz'
    digest = hashlib.sha256(shipped.read_bytes()).hexdigest()
    assert digest == recorded(shipped.with_name('policy.npz.sha256'))
    (run,) = [r for r in RUNS if recorded(r / 'policy.npz.sha256') == digest]
    (path,) = run.glob('*.pt')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    out = tmp_path / 'policy.npz'
    export(ck['model'], ck['config']['env'], out)
    assert out.read_bytes() == shipped.read_bytes()


@pytest.mark.parametrize('run', RUNS, ids=[r.name for r in RUNS])
def test_checkpoint_matches_curve_and_checksums(run):
    """The kept checkpoint is the one listed in checkpoints.sha256 and its iteration and steps are a row of curve.csv."""
    (path,) = run.glob('*.pt')
    listed = dict(line.split()[::-1] for line in (run / 'checkpoints.sha256').read_text().splitlines())
    assert listed[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    rows = {int(r['iteration']): r for r in csv.DictReader(open(run / 'curve.csv'))}
    assert int(rows[ck['iteration']]['steps']) == ck['steps']


def test_trial2_first_iterations_reproduce(tmp_path):
    """The current code reproduces the first iterations of the run that produced the shipped weights exactly (all
    columns of curve.csv except time). Bitwise equality holds on the platform and library versions of that run."""
    run = ROOT / 'training/logs/trial2_trot'
    meta = json.load(open(run / 'run.json'))
    if (platform.system(), platform.machine(), torch.__version__) != (meta['system'], meta['machine'], meta['torch']):
        pytest.skip('different platform or torch version than the logged run')
    model = os.environ.get('GO1_MODEL_DIR')
    if not model:
        pytest.skip('GO1_MODEL_DIR is not set')
    out = tmp_path / 'repro'
    subprocess.run([sys.executable, '-m', 'training.train', '--config', str(run / 'config.json'), '--out', str(out),
                    '--workers', str(meta['workers']), '--seed', str(meta['seed']), '--iterations', '3', '--device', 'cpu'],
                   cwd=ROOT, check=True, capture_output=True, env=dict(os.environ, OPENBLAS_NUM_THREADS='1'))
    got = list(csv.DictReader(open(out / 'curve.csv')))
    want = list(csv.DictReader(open(run / 'curve.csv')))[:len(got)]
    assert len(got) == 3
    assert [{k: v for k, v in r.items() if k != 'time_s'} for r in got] == \
        [{k: v for k, v in r.items() if k != 'time_s'} for r in want]


def test_episode_rules():
    """Success: 3 m along the heading at the end of the warm-up within 90 s; a fall needs > 0.5 s outside the limits."""
    from training.episode_rules import score
    t = np.arange(0, 91.5, 0.01)
    pos = np.c_[np.maximum(t - 1.5, 0) * 0.1, np.zeros_like(t), np.full_like(t, 0.27)]
    quat = np.tile([1.0, 0, 0, 0], (len(t), 1))
    r = score(t, pos, quat)
    assert r['success_b'] and r['fall_t'] is None and r['t_3m'] == pytest.approx(30.0, abs=0.02)
    low = pos.copy()
    low[(t > 40) & (t < 40.4), 2] = 0.1                              # 0.4 s low: not a fall
    assert score(t, low, quat)['success_b']
    low[(t > 50) & (t < 50.7), 2] = 0.1                              # 0.7 s low after 3 m: a fall
    r = score(t, low, quat)
    assert not r['success_b'] and r['success_a'] and r['fall_t'] == pytest.approx(50.0, abs=0.02)
    bench = ROOT / 'bench/t3/evaluate.py'
    if bench.is_file():                                              # same rules as the evaluation tool
        src = bench.read_text()
        for name in ('WARMUP, WINDOW, GOAL = 1.5, 90.0, 3.0', 'FALL_Z, FALL_ANGLE, FALL_HOLD = 0.16, math.radians(50.0), 0.5',
                     "'success_b': i3 is not None and fall is None"):
            assert name in src
