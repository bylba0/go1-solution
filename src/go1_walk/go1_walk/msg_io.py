"""Parts of the walking node that do not need ROS: raw CDR field access, LowState/LowCmd conversion.

LowState and LowCmd have only fixed-size fields, so every field sits at the same byte offset in every serialized
message. The offsets are found in a serialized template whose fields hold distinct marker values.
"""
import numpy as np

MODE_SERVO = 0x0A
CMD_FIELDS = ('q', 'dq', 'kp', 'kd', 'tau')
# name: (dtype, count) of the LowState fields used by the controller
STATE_SPEC = {'quat': ('f4', 4), 'gyro': ('f4', 3), 'acc': ('f4', 3), 'q': ('f4', 12), 'dq': ('f4', 12),
              'tau_est': ('f4', 12), 'foot_force': ('i2', 4), 'tick': ('u4', 1)}
CMD_SPEC = {f: ('f4', 12) for f in CMD_FIELDS}


def markers(spec):
    """Distinct marker values for every element of every field: {name: array}."""
    out, k = {}, 0
    for name, (dtype, n) in spec.items():
        base = {'f4': 1048577.0, 'i2': 0x3A01, 'u4': 0x5A3C96E1}[dtype]
        out[name] = (base + np.arange(k, k + n)).astype(dtype)
        k += n
    return out


def locate(template, marks):
    """{name: (dtype, element indices)} of the marker values in `template`; ValueError if a marker is not found
    exactly once or is not aligned to its size."""
    layout = {}
    for name, values in marks.items():
        size = values.dtype.itemsize
        idx = []
        for v in values:
            pat = v.tobytes()
            if template.count(pat) != 1:
                raise ValueError(f'{name}: marker found {template.count(pat)} times')
            off = template.index(pat)
            if off % size:
                raise ValueError(f'{name}: unaligned offset {off}')
            idx.append(off // size)
        layout[name] = (values.dtype, np.array(idx))
    return layout


def read(raw, layout):
    """Field values from serialized bytes; single-element fields as scalars."""
    out = {}
    for name, (dt, idx) in layout.items():
        v = np.frombuffer(raw, dt, len(raw) // dt.itemsize)[idx]
        out[name] = v if len(idx) > 1 else v[0]
    return out


def write(buf, layout, values):
    """Write fields into a serialized message (bytearray) in place."""
    for name, val in values.items():
        dt, idx = layout[name]
        np.frombuffer(buf, dt, len(buf) // dt.itemsize)[idx] = val


def state_from_msg(m):
    imu, ms = m.imu_state, m.motor_state[:12]
    f32 = np.float32
    return {'quat': np.array(imu.quaternion, f32), 'gyro': np.array(imu.gyroscope, f32),
            'acc': np.array(imu.accelerometer, f32), 'q': np.array([s.q for s in ms], f32),
            'dq': np.array([s.dq for s in ms], f32), 'tau_est': np.array([s.tau_est for s in ms], f32),
            'foot_force': np.array(m.foot_force, np.int16), 'tick': np.uint32(m.tick)}


def state_to_msg(m, s):
    m.imu_state.quaternion, m.imu_state.gyroscope, m.imu_state.accelerometer = s['quat'], s['gyro'], s['acc']
    for i in range(12):
        ms = m.motor_state[i]
        ms.q, ms.dq, ms.tau_est = float(s['q'][i]), float(s['dq'][i]), float(s['tau_est'][i])
    m.foot_force = s['foot_force']
    m.tick = int(np.ravel(s['tick'])[0])
    return m


def cmd_to_msg(m, cmd):
    """Controller command into LowCmd: motors 0..11 in servo mode, 12..19 zero."""
    mc = m.motor_cmd
    for i in range(12):
        c = mc[i]
        c.mode = MODE_SERVO
        c.q, c.dq, c.kp, c.kd, c.tau = (float(cmd[f][i]) for f in CMD_FIELDS)
    for c in mc[12:]:
        c.mode = 0
        c.q = c.dq = c.kp = c.kd = c.tau = 0.0
    return m


def cmd_from_msg(m):
    mc = m.motor_cmd
    out = {f: np.array([getattr(mc[i], f) for i in range(len(mc))], np.float32) for f in CMD_FIELDS}
    out['mode'] = np.array([mc[i].mode for i in range(len(mc))])
    return out


def same(a, b):
    """Field-by-field exact equality of two dicts of arrays/scalars (keys of `a`)."""
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)
