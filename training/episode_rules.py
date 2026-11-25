"""Episode scoring used for the evaluation reports in logs/*/evaluation.json (rules of the T3 task, section 5.3).

An episode is scored from the true trunk state sampled over time: warm-up 1.5 s, then a 90 s window. Success means
reaching 3 m forward along the heading at the end of the warm-up, measured from the position there, within the window.
A fall is any of: trunk height < 0.16 m, |roll| > 50 deg, |pitch| > 50 deg, held continuously for more than 0.5 s.
`success_a`: 3 m reached before the first fall; `success_b` (reported as success): 3 m reached and no fall in the episode.
"""
import math

import numpy as np

WARMUP, WINDOW, GOAL = 1.5, 90.0, 3.0                          # s, s, m
FALL_Z, FALL_ANGLE, FALL_HOLD = 0.16, math.radians(50.0), 0.5  # m, rad, s


def rpy(quat):
    """Roll, pitch, yaw (ZYX) of (w, x, y, z) quaternions, N x 4."""
    w, x, y, z = np.asarray(quat, float).T
    return (np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def fall_onset(t, z, roll, pitch):
    """Start time of the first fall or None: any fall sign held continuously for more than FALL_HOLD (by samples)."""
    bad = (np.asarray(z) < FALL_Z) | (np.abs(roll) > FALL_ANGLE) | (np.abs(pitch) > FALL_ANGLE)
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2] - 1
    long = t[ends] - t[starts] > FALL_HOLD
    return float(t[starts[long][0]]) if long.any() else None


def score(t, pos, quat):
    """Episode metrics: t from reset (s), pos N x 3 (m), quat N x 4 (w, x, y, z) of the trunk in the world."""
    t, pos = np.asarray(t, float), np.asarray(pos, float)
    roll, pitch, yaw = rpy(quat)
    fall = fall_onset(t, pos[:, 2], roll, pitch)
    w = min(int(np.searchsorted(t, WARMUP - 1e-9)), len(t) - 1)          # end of warm-up: reference heading and position
    c, s = math.cos(yaw[w]), math.sin(yaw[w])
    d = pos[:, :2] - pos[w, :2]
    along, lateral = d @ [c, s], d @ [-s, c]
    heading = np.degrees((yaw - yaw[w] + np.pi) % (2 * np.pi) - np.pi)
    win = np.flatnonzero((t >= t[w]) & (t <= WARMUP + WINDOW + 1e-9))
    hit = win[along[win] >= GOAL]
    i3, last = (int(hit[0]) if len(hit) else None), int(win[-1])
    return {'success_a': bool(i3 is not None and (fall is None or t[i3] < fall)),
            'success_b': i3 is not None and fall is None,
            'fall_t': fall, 't_3m': None if i3 is None else float(t[i3] - t[w]),
            'path': float(along[last]), 'lateral_end': float(lateral[last]), 'heading_end_deg': float(heading[last]),
            'z_min': float(pos[:, 2].min()), 'roll_max_deg': float(np.degrees(np.abs(roll).max())),
            'pitch_max_deg': float(np.degrees(np.abs(pitch).max()))}
