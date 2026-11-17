"""Fuzz-тест кодека LowCmd против стандартной (де)сериализации rclpy: 10 000 случайных сообщений, все 20 моторов,
NaN/±Inf/±0/субнормальные/предельные float32. Нужны rclpy и unitree_go (в образе); на хосте без них — skip.
Число сообщений: GO1_CODEC_FUZZ_N (по умолчанию 10000)."""
import os
import struct

import numpy as np
import pytest

pytest.importorskip('rclpy')
unitree_go = pytest.importorskip('unitree_go.msg')
from rclpy.serialization import deserialize_message, serialize_message  # noqa: E402

from go1_safety.lowcmd_codec import FIELDS, LowCmdCodec  # noqa: E402

LowCmd = unitree_go.LowCmd
N = int(os.environ.get('GO1_CODEC_FUZZ_N', '10000'))
F32_MAX_EXACT = float(np.finfo(np.float32).max)
# Python-обёртка rosidl отклоняет |x| > 3.402823466e+38 (округлённая граница), то есть ровно ±FLT_MAX; ±Inf и NaN пропускает.
# Поэтому в сравнении с rclpy берём соседнее значение, а сам FLT_MAX проверяем записью битов в байты (test_flt_max_bits).
F32_MAX = float(np.nextafter(np.float32(F32_MAX_EXACT), np.float32(0)))
F32_TINY = float(np.finfo(np.float32).tiny)
SPECIAL = [0.0, -0.0, float('inf'), float('-inf'), float('nan'), F32_MAX, -F32_MAX, F32_TINY, -F32_TINY,
           1.4e-45, -1.4e-45, 1.0, -1.0, 2.146e9, 16000.0]           # + PosStopF / VelStopF Unitree


def rand_f32(rng, n):
    """Смесь: специальные значения, случайные битовые шаблоны float32 (без NaN-нагрузок), обычный диапазон."""
    kind = rng.integers(0, 3, n)
    out = np.empty(n, dtype=np.float32)
    special = np.array(SPECIAL, dtype=np.float32)
    out[kind == 0] = special[rng.integers(0, len(special), int((kind == 0).sum()))]
    bits = rng.integers(0, 2 ** 32, int((kind == 1).sum()), dtype=np.uint64).astype(np.uint32)
    vals = bits.view(np.float32)
    vals[np.isnan(vals)] = np.float32('nan')                         # NaN — канонический (нагрузка не переносится через float)
    vals[np.abs(vals) == np.float32(F32_MAX_EXACT)] *= np.float32(F32_MAX / F32_MAX_EXACT)   # см. F32_MAX выше
    out[kind == 1] = vals
    out[kind == 2] = rng.uniform(-100, 100, int((kind == 2).sum())).astype(np.float32)
    return out


def same_f32(a, b):
    """Побитовое равенство float32; NaN == NaN (нагрузку не сравниваем)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (a.view(np.uint32) == b.view(np.uint32))))


@pytest.fixture(scope='module')
def codec():
    return LowCmdCodec()


def test_fuzz_decode_vs_rclpy(codec):
    """rclpy.serialize_message(случайный LowCmd) → наш decode == поля сообщения (все 20 моторов, побитово)."""
    rng = np.random.default_rng(20261117)
    bad = 0
    for k in range(N):
        m = LowCmd()
        m.head = [int(x) for x in rng.integers(0, 256, 2)]
        m.level_flag = int(rng.integers(0, 256))
        m.frame_reserve = int(rng.integers(0, 256))
        m.sn = [int(x) for x in rng.integers(0, 2 ** 32, 2, dtype=np.uint64)]
        m.version = [int(x) for x in rng.integers(0, 2 ** 32, 2, dtype=np.uint64)]
        m.bandwidth = int(rng.integers(0, 2 ** 16))
        m.crc = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
        m.wireless_remote = [int(x) for x in rng.integers(0, 256, 40)]
        vals = {f: rand_f32(rng, 20) for f in FIELDS}
        for i, c in enumerate(m.motor_cmd):
            c.mode = int(rng.integers(0, 256))
            c.reserve = [int(x) for x in rng.integers(0, 2 ** 32, 3, dtype=np.uint64)]
            for f in FIELDS:
                setattr(c, f, float(vals[f][i]))
        raw = serialize_message(m)
        got = codec.decode(raw)
        assert got is not None, f'#{k}: decode вернул None'
        ref = deserialize_message(raw, LowCmd)
        for f in FIELDS:
            if not (same_f32(got[f], vals[f]) and same_f32(got[f], [getattr(c, f) for c in ref.motor_cmd])):
                bad += 1
                break
    assert bad == 0, f'расхождений декодера: {bad} из {N}'
    print(f'\ndecode: {N} сообщений × 20 моторов × 5 полей — совпало побитово с rclpy')


def test_fuzz_encode_vs_rclpy(codec):
    """Наш encode(...) == rclpy.serialize_message(того же LowCmd) байт в байт; и обратно deserialize даёт те же поля."""
    rng = np.random.default_rng(17112026)
    bad = 0
    for k in range(N):
        vals = {f: rand_f32(rng, 12) for f in FIELDS}
        ours = codec.encode(vals['q'], vals['dq'], vals['kp'], vals['kd'], vals['tau'])
        m = LowCmd()
        m.head = [0xFE, 0xEF]
        m.level_flag = 0xFF
        for i in range(12):
            c = m.motor_cmd[i]
            c.mode = 0x0A
            for f in FIELDS:
                setattr(c, f, float(vals[f][i]))
        ref = serialize_message(m)
        if ours != ref:
            # единственное допустимое различие — нагрузка NaN; проверяем, что отличаются только байты NaN-значений
            a = np.frombuffer(ours, dtype='<f4', count=len(ours) // 4)
            b = np.frombuffer(ref, dtype='<f4', count=len(ref) // 4)
            diff = np.nonzero(a.view(np.uint32) != b.view(np.uint32))[0]
            if not (len(ours) == len(ref) and np.all(np.isnan(a[diff]) & np.isnan(b[diff]))):
                bad += 1
                continue
        back = deserialize_message(ours, LowCmd)
        for i in range(20):
            c = back.motor_cmd[i]
            if i < 12:
                ok = c.mode == 0x0A and all(same_f32([getattr(c, f)], [vals[f][i]]) for f in FIELDS)
            else:
                ok = c.mode == 0 and all(getattr(c, f) == 0.0 for f in FIELDS)
            if not ok:
                bad += 1
                break
    assert bad == 0, f'расхождений кодера: {bad} из {N}'
    print(f'encode: {N} сообщений — байты совпали с rclpy.serialize_message, обратное чтение совпало')


def test_struct_layout_sanity(codec):
    """Размер и смещения стабильны: сообщение фиксированной длины, поля float32 выровнены по 4."""
    assert codec.size == len(serialize_message(LowCmd()))
    for f in FIELDS:
        assert len(set(codec.idx[f].tolist())) == 20
    assert struct.unpack('<2B', codec.template[:2]) in ((0, 1), (0, 3))


def test_flt_max_bits(codec):
    """Ровно ±FLT_MAX: rclpy-Python не может ни присвоить, ни даже десериализовать такое сообщение (assert в setter),
    наш raw-декодер читает его корректно. Эталон — struct.unpack прямо из байтов."""
    raw = bytearray(serialize_message(LowCmd()))
    a = np.frombuffer(raw, dtype='<f4', count=len(raw) // 4)
    for f in FIELDS:
        a[codec.idx[f]] = np.float32(F32_MAX_EXACT)
    a[codec.idx['tau'][19]] = -np.float32(F32_MAX_EXACT)
    got = codec.decode(bytes(raw))
    for f in FIELDS:
        ref = [struct.unpack_from('<f', raw, 4 * int(j))[0] for j in codec.idx[f]]
        assert same_f32(got[f], ref)
    assert got['tau'][19] == -F32_MAX_EXACT and got['q'][0] == F32_MAX_EXACT
    with pytest.raises(Exception):
        deserialize_message(bytes(raw), LowCmd)          # подтверждаем ограничение rclpy (обычная подписка упала бы)
