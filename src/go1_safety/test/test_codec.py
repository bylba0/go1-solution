"""Кодек LowCmd против штатной (де)сериализации rclpy. Нужны rclpy и unitree_go (в образе); на хосте без них — skip."""
import numpy as np
import pytest

pytest.importorskip('rclpy')
unitree_go = pytest.importorskip('unitree_go.msg')
from rclpy.serialization import deserialize_message, serialize_message  # noqa: E402

from go1_safety.lowcmd_codec import FIELDS, LowCmdCodec  # noqa: E402

LowCmd = unitree_go.LowCmd


@pytest.fixture(scope='module')
def codec():
    return LowCmdCodec()


def test_decode_matches_rclpy(codec):
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = LowCmd()
        vals = {f: rng.uniform(-50, 50, 20).astype(np.float32) for f in FIELDS}
        for i, c in enumerate(m.motor_cmd):
            c.mode = int(rng.integers(0, 256))
            for f in FIELDS:
                setattr(c, f, float(vals[f][i]))
        m.head = [1, 2]
        m.crc = 12345
        got = codec.decode(serialize_message(m))
        for f in FIELDS:
            assert np.array_equal(got[f].astype(np.float32), vals[f]), f


def test_decode_keeps_nonfinite(codec):
    m = LowCmd()
    m.motor_cmd[3].kp = float('nan')
    m.motor_cmd[15].tau = float('-inf')
    got = codec.decode(serialize_message(m))
    assert np.isnan(got['kp'][3]) and np.isneginf(got['tau'][15])


def test_encode_matches_rclpy(codec):
    rng = np.random.default_rng(1)
    q, dq, kp, kd, tau = (rng.uniform(-3, 3, 12) for _ in range(5))
    m = deserialize_message(codec.encode(q, dq, kp, kd, tau), LowCmd)
    for i in range(20):
        c = m.motor_cmd[i]
        if i < 12:
            assert c.mode == 0x0A
            for f, v in (('q', q), ('dq', dq), ('kp', kp), ('kd', kd), ('tau', tau)):
                assert getattr(c, f) == pytest.approx(float(np.float32(v[i])), abs=0)
        else:
            assert c.mode == 0 and c.q == 0 and c.kp == 0 and c.kd == 0 and c.tau == 0 and c.dq == 0
    assert list(m.head) == [0xFE, 0xEF] and m.level_flag == 0xFF


def test_wrong_size_falls_back(codec):
    raw = serialize_message(LowCmd())
    assert codec.decode(raw[:-4]) is None
    assert codec.decode_slow(raw) is not None
