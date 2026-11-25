"""Go1 velocity-tracking environment on MuJoCo with the simulator's joint control law.

Physics matches the simulator node: same scene, 2 ms Euler step, start from the `home` keyframe under the
node's stand hold, and before every physics step tau = clip(kp*(q_des - q) + kd*(0 - dq), +-limit) applied via
qfrc_applied. The policy runs at 50 Hz; its target reaches the joints as 200 Hz messages linearly interpolated
between policy steps, rounded to float32 as in LowCmd, with an optional per-episode transport delay.
Joint order everywhere: FR, FL, RR, RL legs, each hip, thigh, calf (LowCmd order, also qpos[7:] of the model).
"""
import math
import os

import mujoco
import numpy as np

STAND_Q = np.array([0.0, 0.8536, -1.7072] * 4)  # simulator stand pose
TORQUE_LIMIT = np.array([23.7, 23.7, 35.55] * 4)
HOLD_KP, HOLD_KD = 40.0, 2.0  # simulator hold gains before the first command
SIM_MS, POLICY_MS, MSG_MS = 2, 20, 5
MAX_MSG_STEP = math.radians(4.3)  # per joint between consecutive 200 Hz messages

# Self-collision-free box of joint targets, given for right legs; left legs mirror the hip interval.
# Checked to keep >= 2 mm clearance over all model collision pairs (see tests/test_env.py).
HIP = (-0.35, 0.275)
THIGH = (0.45, 1.25)
CALF = (-2.25, -1.175)


def target_box():
    lo, hi = np.zeros(12), np.zeros(12)
    for leg, right in enumerate((True, False, True, False)):
        h = HIP if right else (-HIP[1], -HIP[0])
        lo[3 * leg:3 * leg + 3] = h[0], THIGH[0], CALF[0]
        hi[3 * leg:3 * leg + 3] = h[1], THIGH[1], CALF[1]
    return lo, hi


def pd_torque(q_des, q, dq, kp, kd):
    """Simulator motor law for servo mode with tau_ff = 0 and dq_des = 0, same operation order."""
    return np.clip(0.0 + kp * (q_des - q) + kd * (0.0 - dq), -TORQUE_LIMIT, TORQUE_LIMIT)


def gait_reference(t_ms, cmd, gait):
    """Open-loop trot offsets from the stand pose at time t_ms: diagonal pairs FR+RL and FL+RR in antiphase, thigh
    swing from the forward command (0.28 m of foot travel per rad), calf lifts the foot in swing; zero when the
    command is zero or `gait` is None."""
    if not gait or np.linalg.norm(cmd) < 0.01:
        return np.zeros(12)
    phase = 2 * math.pi * gait['freq'] * t_ms / 1000 + np.array([0.0, math.pi, math.pi, 0.0])
    amp = np.clip(cmd[0] / (2 * 0.28 * gait['freq']), -gait['thigh_max'], gait['thigh_max'])
    off = np.zeros((4, 3))
    off[:, 1] = amp * np.cos(phase)
    off[:, 2] = -gait['calf'] * np.maximum(np.sin(phase), 0.0)
    return off.ravel()


def gravity_from_quat(quat):
    """Unit gravity direction in the body frame from a (w, x, y, z) world orientation."""
    w, x, y, z = quat
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])


def roll_pitch(quat):
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def f32(x):
    return np.asarray(x, np.float32).astype(np.float64)


class Go1Env:
    """One environment; `cfg` is the "env" section of the training config."""

    def __init__(self, cfg, model_dir, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(model_dir, 'scene.xml'))
        m = self.model
        self.gain, self.bias = m.actuator_gainprm.copy(), m.actuator_biasprm.copy()
        m.actuator_gainprm[:, 0] = 0.0  # torques come only from qfrc_applied, as in the simulator node
        m.actuator_biasprm[:] = 0.0
        self.data = mujoco.MjData(m)
        names = [m.joint(i).name for i in range(1, m.njnt)]
        assert names == [f'{l}_{j}_joint' for l in ('FR', 'FL', 'RR', 'RL') for j in ('hip', 'thigh', 'calf')], names
        self.home = m.key('home').id
        self.floor = m.geom('floor').id
        self.feet = np.array([m.geom(n).id for n in ('FR', 'FL', 'RR', 'RL')])
        self.foot_sites = np.array([m.site(n).id for n in ('FR', 'FL', 'RR', 'RL')])
        self.trunk = m.body('trunk').id
        self.nominal_mass = m.body_mass[self.trunk]
        self.nominal_foot_friction = m.geom_friction[self.feet, 0].copy()
        self.lo, self.hi = target_box()
        self.q0 = STAND_Q
        self.scale = np.tile(cfg['action_scale'], 4)
        self.kp_cmd, self.kd_cmd = cfg['kp'], cfg['kd']
        self.history = cfg['history']
        self.gait = cfg.get('gait')
        self.frame = 45 + (2 if self.gait else 0)
        self.obs_dim = self.frame * self.history
        self.critic_dim = self.frame + 3 + 1 + 4 + 4
        self.max_steps = int(cfg['episode_s'] * 1000) // POLICY_MS
        # Start state: `home` keyframe settled under the simulator hold before the first command; the simulator
        # node runs mj_forward with the position actuators still enabled, then disables them.
        gain, bias = m.actuator_gainprm.copy(), m.actuator_biasprm.copy()
        m.actuator_gainprm[:], m.actuator_biasprm[:] = self.gain, self.bias
        mujoco.mj_resetDataKeyframe(m, self.data, self.home)
        mujoco.mj_forward(m, self.data)
        m.actuator_gainprm[:], m.actuator_biasprm[:] = gain, bias
        self.data.ctrl[:] = 0.0
        for _ in range(int(cfg['settle_s'] * 1000) // SIM_MS):
            self._physics(STAND_Q, HOLD_KP, HOLD_KD)
        self.start = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(m, self.data, self.start, mujoco.mjtState.mjSTATE_INTEGRATION)

    def obs_scale(self, s):
        """Per-element actor input scales from the config section `obs_scales`."""
        frame = [s['gyro']] * 3 + [s['gravity']] * 3 + [s['q']] * 12 + [s['dq']] * 12 + [s['action']] * 12 + s['command']
        return np.tile(frame + ([s['phase']] * 2 if self.gait else []), self.history)

    # ---------------------------------------------------------------- episode
    def reset(self):
        c, r, m, d = self.cfg, self.rng, self.model, self.data
        rnd = c['randomize']
        self.friction = r.uniform(*rnd['foot_friction']) if rnd['enable'] else self.nominal_foot_friction[0]
        m.geom_friction[self.feet, 0] = self.friction
        self.payload = r.uniform(*rnd['trunk_mass_add']) if rnd['enable'] else 0.0
        m.body_mass[self.trunk] = self.nominal_mass + self.payload
        self.kp_scale = r.uniform(1 - rnd['gain_scale'], 1 + rnd['gain_scale']) if rnd['enable'] else 1.0
        self.kd_scale = r.uniform(1 - rnd['gain_scale'], 1 + rnd['gain_scale']) if rnd['enable'] else 1.0
        self.delay_ms = SIM_MS * int(r.integers(0, c['max_delay_ms'] // SIM_MS + 1))
        mujoco.mj_setState(m, d, self.start, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m, d)
        self.t_ms = 0
        self.steps = 0
        self.target = STAND_Q.copy()  # last sent target
        self.applied = None  # until the first message arrives the simulator keeps its own stand hold
        self.pending = []
        self.last_action = np.zeros(12)
        self.air_time = np.zeros(4)
        self.swing_peak = np.zeros(4)
        self.last_contact = np.ones(4, bool)
        self.foot_pos = d.site_xpos[self.foot_sites].copy()
        self.warmup_ms = 1500 if r.random() < c['warmup_prob'] else 0
        self.next_cmd_ms = 0
        self.command = np.zeros(3)
        self._update_command()
        self.next_push_ms = int(r.uniform(*rnd['push_interval_s']) * 1000)
        frame = self._frame()
        self.frames = [frame] * self.history
        return self.obs(), self.critic_obs()

    def _update_command(self):
        c, r = self.cfg['command'], self.rng
        if self.t_ms < self.warmup_ms:
            self.command = np.zeros(3)
            self.next_cmd_ms = self.warmup_ms
            return
        if self.t_ms < self.next_cmd_ms:
            return
        if r.random() < c['zero_prob']:
            cmd = np.zeros(3)
        else:
            vx = r.uniform(*c['vx_main']) if r.random() < c['vx_main_prob'] else r.uniform(*c['vx'])
            vy = r.uniform(-c['vy'], c['vy']) if r.random() < c['vy_prob'] else 0.0
            wz = r.uniform(-c['wz'], c['wz']) if r.random() < c['wz_prob'] else 0.0
            cmd = np.array([vx, vy, wz])
        self.command = cmd
        self.next_cmd_ms = self.t_ms + int(r.uniform(*c['resample_s']) * 1000)

    # ---------------------------------------------------------------- physics
    def _physics(self, q_des, kp, kd):
        d = self.data
        d.qfrc_applied[6:] = pd_torque(q_des, d.qpos[7:], d.qvel[6:], kp, kd)
        mujoco.mj_step(self.model, d)

    def _pack(self, ms):
        """Simulator catching up: `ms` of physics with the last received target and no command reception, then the
        newest message that has arrived meanwhile is taken (subscription depth 1)."""
        kp, kd = f32(self.kp_cmd) * self.kp_scale, f32(self.kd_cmd) * self.kd_scale
        for _ in range(ms // SIM_MS):
            if self.applied is None:
                self._physics(STAND_Q, HOLD_KP, HOLD_KD)
            else:
                self._physics(self.applied, kp, kd)
        self.t_ms += ms
        while self.pending and self.pending[0][0] <= self.t_ms:
            self.applied = self.pending.pop(0)[1]

    def clip_target(self, action):
        """Action -> next policy target: box, then rate limit so each of 4 messages moves <= MAX_MSG_STEP."""
        ref = gait_reference(self.t_ms + POLICY_MS, self.command, self.gait)
        raw = np.clip(self.q0 + ref + self.scale * action, self.lo, self.hi)
        lim = 4 * MAX_MSG_STEP
        return self.target + np.clip(raw - self.target, -lim, lim)

    def messages(self, new):
        """The four 200 Hz targets between the last policy target and the new one."""
        return f32(self.target + (new - self.target) * np.array([[0.25], [0.5], [0.75], [1.0]]))

    def step(self, action):
        c, d, r = self.cfg, self.data, self.rng
        action = np.clip(action, -c['action_clip'], c['action_clip'])
        new = self.clip_target(action)
        self.last_msgs = self.messages(new)
        for j, msg in enumerate(self.last_msgs):
            self.pending.append((self.t_ms + MSG_MS * j + self.delay_ms, msg))
        self.target = new
        kp, kd = f32(self.kp_cmd) * self.kp_scale, f32(self.kd_cmd) * self.kd_scale
        tau_sum = np.zeros(12)
        energy = 0.0
        for k in range(POLICY_MS // SIM_MS):
            t = self.t_ms + SIM_MS * k
            while self.pending and self.pending[0][0] <= t:
                self.applied = self.pending.pop(0)[1]
            if self.applied is None:
                self._physics(STAND_Q, HOLD_KP, HOLD_KD)
            else:
                self._physics(self.applied, kp, kd)
            tau = d.qfrc_applied[6:]
            tau_sum += np.abs(tau)
            energy += np.sum(np.abs(tau * d.qvel[6:]))
        self.t_ms += POLICY_MS
        self.steps += 1
        rnd = c['randomize']
        # random numbers are drawn only for enabled options, so a config without them reproduces older runs exactly
        if rnd['enable'] and rnd.get('pack_prob', 0.0) > 0 and r.random() < rnd['pack_prob']:
            self._pack(int(r.integers(1, rnd.get('pack_ms', 200) // POLICY_MS + 1)) * POLICY_MS)
        if rnd['enable'] and rnd['push_vel'] > 0 and self.t_ms >= self.next_push_ms:
            ang = r.uniform(0, 2 * math.pi)
            d.qvel[0:2] += r.uniform(0, c['randomize']['push_vel']) * np.array([math.cos(ang), math.sin(ang)])
            self.next_push_ms = self.t_ms + int(r.uniform(*c['randomize']['push_interval_s']) * 1000)

        # contacts of the last physics step
        n = d.ncon
        geom = d.contact.geom[:n]
        body = self.model.geom_bodyid[geom] if n else np.zeros((0, 2), int)
        self_collision = bool(n) and bool(np.any((body[:, 0] > 0) & (body[:, 1] > 0) & (d.contact.dist[:n] < 0)))
        on_floor = geom[(geom == self.floor).any(1)].ravel() if n else np.zeros(0, int)
        contact = np.isin(self.feet, on_floor)

        roll, pitch = roll_pitch(d.qpos[3:7])
        lim = math.radians(c['max_tilt_deg'])
        fell = d.qpos[2] < c['min_height'] or abs(roll) > lim or abs(pitch) > lim
        terminated = fell or self_collision
        truncated = self.steps >= self.max_steps

        rew, info = self._reward(action, tau_sum / (POLICY_MS // SIM_MS), energy, contact, terminated)
        self.last_action = action
        self._update_command()
        p_drop = c['randomize'].get('obs_drop_prob', 0.0) if c['randomize']['enable'] else 0.0
        drop = p_drop > 0 and r.random() < p_drop
        self.frames = self.frames[1:] + [self.frames[-1] if drop else self._frame()]
        info.update(fell=fell, self_collision=self_collision)
        return self.obs(), self.critic_obs(), rew, terminated, truncated, info

    # ---------------------------------------------------------------- observations
    def lowstate(self):
        """State as the simulator publishes it (float32)."""
        d = self.data
        return {'tick': int(d.time * 1000.0), 'quat': d.qpos[3:7].astype(np.float32),
                'gyro': d.qvel[3:6].astype(np.float32), 'q': d.qpos[7:].astype(np.float32),
                'dq': d.qvel[6:].astype(np.float32)}

    def _frame(self, noise=True):
        s = self.lowstate()
        quat, gyro, q, dq = (s[k].astype(np.float64) for k in ('quat', 'gyro', 'q', 'dq'))
        grav = gravity_from_quat(quat)
        lvl = self.cfg['noise']['level'] if noise else 0.0
        if lvl > 0:
            ns, u = self.cfg['noise'], lambda k, n: lvl * ns[k] * self.rng.uniform(-1, 1, n)
            gyro, grav, q, dq = gyro + u('gyro', 3), grav + u('gravity', 3), q + u('q', 12), dq + u('dq', 12)
        frame = [gyro, grav, q - self.q0, dq, self.last_action, self.command]
        if self.gait:
            ph = 2 * math.pi * self.gait['freq'] * self.t_ms / 1000
            frame.append([math.sin(ph), math.cos(ph)])
        return np.concatenate(frame)

    def obs(self):
        """Actor observation: last `history` frames, newest first."""
        return np.concatenate(self.frames[::-1])

    def critic_obs(self):
        d = self.data
        vel = self.body_linvel()
        n = d.ncon
        geom = d.contact.geom[:n]
        on_floor = geom[(geom == self.floor).any(1)].ravel() if n else np.zeros(0, int)
        contact = np.isin(self.feet, on_floor).astype(float)
        extra = [self.friction, self.payload, self.kp_scale, self.kd_scale]
        return np.concatenate([self._frame(noise=False), vel, [d.qpos[2]], contact, extra])

    def body_linvel(self):
        d = self.data
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, d.qpos[3:7])
        return rot.reshape(3, 3).T @ d.qvel[0:3]

    # ---------------------------------------------------------------- reward
    def _reward(self, action, tau, energy, contact, terminated):
        d, w, cmd = self.data, self.cfg['reward'], self.command
        vel = self.body_linvel()
        gyro = d.qvel[3:6]
        grav = gravity_from_quat(d.qpos[3:7])
        q = d.qpos[7:]
        moving = np.linalg.norm(cmd) > 0.01
        sigma = w['tracking_sigma']
        dt = POLICY_MS / 1000

        foot = d.site_xpos[self.foot_sites]
        foot_vel = (foot - self.foot_pos) / dt
        self.foot_pos = foot.copy()
        contact_filt = contact | self.last_contact
        first_contact = (self.air_time > 0) & contact_filt
        self.air_time += dt
        self.swing_peak = np.maximum(self.swing_peak, foot[:, 2])

        terms = {
            'tracking_lin_vel': math.exp(-np.sum((cmd[:2] - vel[:2]) ** 2) / sigma),
            'tracking_ang_vel': math.exp(-(cmd[2] - gyro[2]) ** 2 / sigma),
            'lin_vel_z': d.qvel[2] ** 2,
            'ang_vel_xy': np.sum(gyro[:2] ** 2),
            'orientation': np.sum(grav[:2] ** 2),
            'torques': math.sqrt(np.sum(tau ** 2)) + np.sum(tau),
            'energy': energy / (POLICY_MS // SIM_MS),
            'action_rate': np.sum((action - self.last_action) ** 2),
            'pose': math.exp(-np.sum((q - self.q0) ** 2 * np.array([1.0, 1.0, 0.1] * 4))),
            'stand_still': np.sum(np.abs(q - self.q0)) * (not moving),
            'feet_slip': np.sum(np.sum(foot_vel[:, :2] ** 2, 1) * contact) * moving,
            'feet_air_time': np.sum((self.air_time - 0.1) * first_contact) * moving,
            'feet_height': np.sum((self.swing_peak / w['max_foot_height'] - 1) ** 2 * first_contact) * moving,
            'termination': float(terminated),
        }
        self.air_time *= ~contact
        self.swing_peak *= ~contact
        self.last_contact = contact
        total = sum(w['scales'][k] * v for k, v in terms.items()) * dt
        return max(total, 0.0), {'vel_err': float(np.linalg.norm(cmd[:2] - vel[:2]))}
