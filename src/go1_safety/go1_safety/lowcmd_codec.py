"""Быстрый кодек unitree_go/msg/LowCmd без Python-объектов на 20 моторов.

LowCmd — сообщение фиксированной длины (только массивы фиксированного размера), поэтому смещения полей в CDR-байтах
постоянны. Смещения не задаются вручную: шаблон сериализуется с уникальными метками в каждом поле q/dq/kp/kd/tau
каждого мотора, метки ищутся в байтах. Чтение входа — numpy.frombuffer по индексам; запись выхода — заполнение
float32 в готовом шаблоне и publish(bytes). Это в разы дешевле, чем (де)сериализация 20 вложенных MotorCmd в rclpy.
Любое несоответствие (другая длина, не little-endian CDR) → None, и узел использует обычную десериализацию.
"""
import struct

import numpy as np
from rclpy.serialization import deserialize_message, serialize_message
from unitree_go.msg import LowCmd

FIELDS = ('q', 'dq', 'kp', 'kd', 'tau')
N_SLOTS = 20
CDR_LE = (b'\x00\x01', b'\x00\x03')               # заголовок инкапсуляции: CDR / PL_CDR little-endian


class LowCmdCodec:
    def __init__(self, n_used=12, mode_used=0x0A):
        probe = LowCmd()
        assert len(probe.motor_cmd) == N_SLOTS
        sentinel = {}
        for i, c in enumerate(probe.motor_cmd):
            for fi, f in enumerate(FIELDS):
                v = 1000.0 + i * 10 + fi + 0.5          # точно представимо в float32, все различны
                setattr(c, f, v)
                sentinel[(f, i)] = v
        raw = serialize_message(probe)
        self.size = len(raw)
        assert raw[:2] in CDR_LE, raw[:2]
        idx = {f: np.zeros(N_SLOTS, dtype=np.int64) for f in FIELDS}
        for (f, i), v in sentinel.items():
            pat = struct.pack('<f', v)
            pos = raw.find(pat)
            assert pos >= 0 and raw.find(pat, pos + 1) < 0 and pos % 4 == 0, (f, i, pos)
            idx[f][i] = pos // 4
        self.idx = idx
        # mode — uint8 перед q того же мотора (проверяем по факту, а не по правилам выравнивания)
        probe2 = LowCmd()
        for i, c in enumerate(probe2.motor_cmd):
            c.mode = 0x40 + i
        raw2 = serialize_message(probe2)
        self.mode_pos = np.array([raw2.find(bytes([0x40 + i]), 4) for i in range(N_SLOTS)])
        assert all(p > 0 for p in self.mode_pos) and len(set(self.mode_pos.tolist())) == N_SLOTS
        # шаблон выхода: моторы 0..n_used-1 — mode 0x0A, остальные нули; head/level_flag как у Unitree
        tmpl = LowCmd()
        tmpl.head = [0xFE, 0xEF]
        tmpl.level_flag = 0xFF
        for i in range(n_used):
            tmpl.motor_cmd[i].mode = mode_used
        self.template = bytearray(serialize_message(tmpl))
        self.n_used = n_used
        self._out_idx = {f: idx[f][:n_used] for f in FIELDS}

    def decode(self, raw):
        """bytes → dict {поле: np.ndarray(20, float64)} или None (тогда — обычная десериализация)."""
        if len(raw) != self.size or bytes(raw[:2]) not in CDR_LE:
            return None
        a = np.frombuffer(raw, dtype='<f4', count=self.size // 4)
        return {f: a[self.idx[f]].astype(np.float64) for f in FIELDS}

    @staticmethod
    def decode_slow(raw):
        """Запасной путь через rclpy (любая валидная сериализация)."""
        m = deserialize_message(raw, LowCmd)
        cmds = m.motor_cmd
        if len(cmds) != N_SLOTS:
            return None
        return {f: np.array([getattr(c, f) for c in cmds], dtype=np.float64) for f in FIELDS}

    def encode(self, q, dq, kp, kd, tau):
        """12 значений каждого поля → bytes LowCmd (моторы 12..19 — нули, mode 0x0A у 0..11)."""
        buf = bytearray(self.template)
        a = np.frombuffer(buf, dtype='<f4', count=self.size // 4)
        for f, v in (('q', q), ('dq', dq), ('kp', kp), ('kd', kd), ('tau', tau)):
            a[self._out_idx[f]] = v
        return bytes(buf)
