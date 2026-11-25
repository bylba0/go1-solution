import os
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from go1_walk.controller import WalkController
from go1_walk.msg_io import (CMD_FIELDS, CMD_SPEC, MODE_SERVO, STATE_SPEC, cmd_from_msg, cmd_to_msg, locate, markers,
                             read, same, state_from_msg, state_to_msg, write)

POLICY = os.path.join(os.path.dirname(__file__), '..', 'policy', 'policy.npz')


def fake_state():
    return NS(imu_state=NS(quaternion=np.zeros(4, np.float32), gyroscope=np.zeros(3, np.float32),
                           accelerometer=np.zeros(3, np.float32)),
              motor_state=[NS(q=0.0, dq=0.0, tau_est=0.0) for _ in range(20)],
              foot_force=np.zeros(4, np.int16), tick=0)


def fake_cmd():
    return NS(motor_cmd=[NS(mode=7, q=9.0, dq=9.0, kp=9.0, kd=9.0, tau=9.0) for _ in range(20)])


def serialize_state(m):
    """CDR-like little-endian layout of the used LowState fields with padding and unrelated fields in between."""
    imu = m.imu_state
    out = bytearray(b'\x00\x01\x00\x00') + struct.pack('<2B2xI', 0xFE, 0xEF, 77)
    out += struct.pack('<4f3f3f3f', *imu.quaternion, *imu.gyroscope, *imu.accelerometer, 0.1, 0.2, 0.3)
    for s in m.motor_state:
        out += struct.pack('<B3x7fbx2xI', 10, s.q, s.dq, 0.5, s.tau_est, 0, 0, 0, 30, 0)
    out += struct.pack('<4h4hI', *m.foot_force, 1, 2, 3, 4, m.tick)
    return bytes(out)


def serialize_cmd(m):
    out = bytearray(b'\x00\x01\x00\x00') + struct.pack('<4B', 0xFE, 0xEF, 0, 0)
    for c in m.motor_cmd:
        out += struct.pack('<B3x5f3I', c.mode, c.q, c.dq, c.tau, c.kp, c.kd, 0, 0, 0)
    return bytes(out)


def test_markers_distinct_and_typed():
    for spec in (STATE_SPEC, CMD_SPEC):
        marks = markers(spec)
        assert {k: (v.dtype.str[1:], len(v)) for k, v in marks.items()} == spec
        raw = [x.tobytes() for v in marks.values() for x in v]
        assert len(set(raw)) == len(raw)


def test_state_raw_read_matches_full():
    marks = markers(STATE_SPEC)
    layout = locate(serialize_state(state_to_msg(fake_state(), marks)), marks)
    rng = np.random.default_rng(0)
    s = {k: rng.normal(size=len(v)).astype(v.dtype) if v.dtype.kind == 'f' else
         rng.integers(0, 3000, len(v)).astype(v.dtype) for k, v in marks.items()}
    s['tick'] = np.uint32(123456)
    raw = serialize_state(state_to_msg(fake_state(), s))
    got = read(raw, layout)
    assert same(state_from_msg(state_to_msg(fake_state(), s)), got)
    assert isinstance(got['tick'], np.uint32) and int(got['tick']) == 123456
    assert got['q'].shape == (12,) and got['foot_force'].dtype == np.int16


def test_locate_rejects_missing_duplicate_unaligned():
    marks = {'a': np.array([5.5], np.float32)}
    pat = marks['a'][0].tobytes()
    with pytest.raises(ValueError):
        locate(b'\x00' * 16, marks)
    with pytest.raises(ValueError):
        locate(b'\x00' * 4 + pat + pat, marks)
    with pytest.raises(ValueError):
        locate(b'\x00' * 2 + pat + b'\x00' * 2, marks)
    assert locate(b'\x00' * 4 + pat, marks)['a'][1].tolist() == [1]


def test_cmd_to_msg_servo_and_zero_tail():
    cmd = {f: np.arange(12, dtype=np.float32) + k for k, f in enumerate(CMD_FIELDS)}
    m = cmd_to_msg(fake_cmd(), cmd)
    got = cmd_from_msg(m)
    assert got['mode'].tolist() == [MODE_SERVO] * 12 + [0] * 8
    for f in CMD_FIELDS:
        assert got[f][:12].tolist() == cmd[f].tolist() and not got[f][12:].any()


def test_raw_cmd_write_matches_full_serialization():
    marks = markers(CMD_SPEC)
    tmpl = serialize_cmd(cmd_to_msg(fake_cmd(), marks))
    layout, buf = locate(tmpl, marks), bytearray(tmpl)
    ctrl = WalkController(POLICY)
    tick, sent = 1000, 0
    for _ in range(40):
        state = {'quat': np.array([1, 0, 0, 0], np.float32), 'gyro': np.zeros(3, np.float32),
                 'q': ctrl.q0.astype(np.float32), 'dq': np.zeros(12, np.float32), 'tick': tick}
        cmd = ctrl.on_lowstate(state, (0.3, 0.0, 0.0))
        tick += 2
        if cmd is None:
            continue
        sent += 1
        write(buf, layout, {f: cmd[f] for f in CMD_FIELDS})
        assert bytes(buf) == serialize_cmd(cmd_to_msg(fake_cmd(), cmd))
    assert sent >= 15  # a target every 5 ms of ticks


def test_same():
    assert same({'a': np.ones(3), 't': 5}, {'a': np.ones(3), 't': 5, 'x': 1})
    assert not same({'a': np.ones(3)}, {'a': np.array([1.0, 1.0, 2.0])})
