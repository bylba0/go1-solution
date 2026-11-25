"""Walking policy executor (numpy only).

Time comes from LowState.tick (ms of simulation time): a policy step every `policy_ms`, between policy steps the
target is linearly interpolated and a new target is emitted every `msg_ms`. Targets are clipped to the
self-collision-free box and each emitted target differs from the previously emitted one by at most `max_msg_step`
per joint: targets are emitted one per call, in order, and a policy step interpolates from the last emitted target.
Joint order: FR, FL, RR, RL legs, each hip, thigh, calf.
"""
import math

import numpy as np


def gravity_from_quat(quat):
    """Unit gravity direction in the body frame from a (w, x, y, z) world orientation."""
    w, x, y, z = quat
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])


def gait_reference(t_ms, cmd, gait):
    """Open-loop trot offsets from the stand pose (same as in training); zero for a zero command or no gait."""
    if not gait or np.linalg.norm(cmd) < 0.01:
        return np.zeros(12)
    phase = 2 * math.pi * gait['freq'] * t_ms / 1000 + np.array([0.0, math.pi, math.pi, 0.0])
    amp = np.clip(cmd[0] / (2 * 0.28 * gait['freq']), -gait['thigh_max'], gait['thigh_max'])
    off = np.zeros((4, 3))
    off[:, 1] = amp * np.cos(phase)
    off[:, 2] = -gait['calf'] * np.maximum(np.sin(phase), 0.0)
    return off.ravel()


def yaw_from_quat(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class WalkController:
    def __init__(self, policy_npz, heading_hold=True, heading_gain=3.0, heading_max=0.3, kp=None, kd=None):
        p = np.load(policy_npz)
        n = int(p['n_layers'])
        self.layers = [(p[f'w{i}'], p[f'b{i}']) for i in range(n)]
        self.obs_mean, self.obs_std, self.obs_clip = p['obs_mean'], p['obs_std'], float(p['obs_clip'])
        self.history = int(p['history'])
        self.q0, self.scale, self.action_clip = p['q0'], p['action_scale'], float(p['action_clip'])
        self.lo, self.hi, self.max_msg_step = p['target_lo'], p['target_hi'], float(p['max_msg_step'])
        # gains of the trained policy unless overridden (training randomizes them by +-10 %)
        self.kp = np.full(12, float(p['kp'] if kp is None else kp), np.float32)
        self.kd = np.full(12, float(p['kd'] if kd is None else kd), np.float32)
        self.policy_ms, self.msg_ms = int(p['policy_ms']), int(p['msg_ms'])
        self.gait = ({'freq': float(p['gait_freq']), 'calf': float(p['gait_calf']),
                      'thigh_max': float(p['gait_thigh_max'])} if 'gait_freq' in p else None)
        self.heading_hold, self.heading_gain, self.heading_max = heading_hold, heading_gain, heading_max
        self.reset()

    def reset(self):
        self.sent = self.q0.copy()  # last emitted target (before float32 rounding)
        self.last_action = np.zeros(12)
        self.frames = None
        self.next_policy = None
        self.t0 = None  # tick of the first policy step: gait phase origin
        self.msgs = []  # (tick, target) not yet emitted
        self.yaw0 = None

    def act(self, obs):
        """Deterministic policy action for a raw actor observation."""
        x = np.clip((obs - self.obs_mean) / self.obs_std, -self.obs_clip, self.obs_clip)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
        return x

    def command(self, lowstate, cmd_vel):
        if cmd_vel is None:
            return np.zeros(3)
        cmd = np.array(cmd_vel, dtype=np.float64)
        if np.linalg.norm(cmd) < 0.01:  # zero command: stand (no heading correction, which would start the gait)
            return np.zeros(3)
        if self.heading_hold:
            err = math.remainder(yaw_from_quat(lowstate['quat']) - self.yaw0, 2 * math.pi)
            cmd[2] += float(np.clip(-self.heading_gain * err, -self.heading_max, self.heading_max))
        return cmd

    def observe(self, lowstate, cmd, t_ms):
        quat, gyro, q, dq = (np.asarray(lowstate[k], np.float64) for k in ('quat', 'gyro', 'q', 'dq'))
        frame = [gyro, gravity_from_quat(quat), q - self.q0, dq, self.last_action, cmd]
        if self.gait:
            ph = 2 * math.pi * self.gait['freq'] * t_ms / 1000
            frame.append([math.sin(ph), math.cos(ph)])
        frame = np.concatenate(frame)
        self.frames = [frame] * self.history if self.frames is None else self.frames[1:] + [frame]
        return np.concatenate(self.frames[::-1])

    def policy_step(self, lowstate, cmd, tick):
        t_ms = tick - self.t0
        action = np.clip(self.act(self.observe(lowstate, cmd, t_ms)), -self.action_clip, self.action_clip)
        ref = gait_reference(t_ms + self.policy_ms, cmd, self.gait)
        raw = np.clip(self.q0 + ref + self.scale * action, self.lo, self.hi)
        lim = 4 * self.max_msg_step
        new = self.sent + np.clip(raw - self.sent, -lim, lim)
        targets = self.sent + (new - self.sent) * np.array([[0.25], [0.5], [0.75], [1.0]])
        targets[-1] = new
        self.msgs = [(tick + self.msg_ms * j, targets[j]) for j in range(4)]
        self.last_action = action

    def on_lowstate(self, lowstate, cmd_vel, t_sim=None):
        if not all(np.isfinite(lowstate[k]).all() for k in ('quat', 'gyro', 'q', 'dq')):
            return None  # non-finite state: skipped, the internal state stays valid and the last target is held
        tick = int(lowstate['tick'])
        if self.yaw0 is None:
            self.yaw0 = yaw_from_quat(lowstate['quat'])
            self.next_policy = self.t0 = tick
        if tick >= self.next_policy:
            self.policy_step(lowstate, self.command(lowstate, cmd_vel), self.next_policy)
            while self.next_policy <= tick:
                self.next_policy += self.policy_ms
        if not self.msgs or self.msgs[0][0] > tick:
            return None
        self.sent = self.msgs.pop(0)[1]
        zero = np.zeros(12, np.float32)
        return {'q': self.sent.astype(np.float32), 'dq': zero, 'kp': self.kp.copy(), 'kd': self.kd.copy(),
                'tau': zero.copy()}
