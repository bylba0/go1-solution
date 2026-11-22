"""T4, режим lidar: страховка на случай пропажи /scan (без ROS).

Узел в режиме lidar ведёт параллельно два оценщика с одним курсом (мировой yaw IMU): ядро lidar (lidar_mode) и наш
legs (core.OdometryEstimator). Этот класс выбирает, чью позицию xy публиковать, и сшивает их без скачков:
  * lidar — пока сканы приходят: выход = поза lidar + поправка, которая затухает экспоненциально (blend_tau);
  * legs  — если /scan не приходил дольше scan_timeout с sim (или поза рабочего потока старше этого): выход =
    последний выход + ПРИРАЩЕНИЯ legs с момента переключения (поза не прыгает);
  * возврат к lidar — когда сканы снова идут без перерывов дольше scan_timeout в течение resume_delay с: поправка =
    (выход − поза lidar) в момент возврата, дальше она затухает с постоянной blend_tau (как BLEND_TAU в ядре);
    если поправка больше max_correction — она НЕ затухает (выход = наша поза на возврате + приращения lidar).
max_correction: ядро после паузы сопоставляет скан с опорным по своему прогнозу по ногам и исправить может только
расхождение в пределах PAIR_DIST = 0,3 м; при большем расхождении его абсолютная поза сомнительна (bag B, пауза 40 с:
ядро ошиблось на 0,57 м, наш legs — на 0,33 м), а ввод поправки в 0,4 м за ~1 с даёт RPE участка 40 %.
resume_delay даёт ядру время сопоставить первые сканы после паузы и самому ввести свою поправку (BLEND_TAU = 0,2 с),
чтобы эта поправка не попала в выход быстрым рывком.
Штамп назад (сброс sim) — сброс состояния: снова lidar, без поправки.
"""
import math

import numpy as np

SOURCES = ('lidar', 'legs')


class LidarFallback:
    def __init__(self, scan_timeout=1.0, blend_tau=1.0, resume_delay=0.5, max_correction=0.25):
        if not (scan_timeout > 0 and blend_tau > 0 and resume_delay >= 0 and max_correction >= 0):
            raise ValueError('scan_timeout > 0, blend_tau > 0, resume_delay ≥ 0, max_correction ≥ 0')
        self.scan_timeout = float(scan_timeout)
        self.blend_tau = float(blend_tau)
        self.resume_delay = float(resume_delay)
        self.max_correction = float(max_correction)
        self.switches = 0          # число переключений источника (оба направления), не сбрасывается штампом назад
        self.resets = 0            # сбросы по штампу назад
        self.reset()

    def reset(self):
        self.source = 'lidar'
        self.out = None            # последний выход xy
        self.t = None              # время последнего update
        self.t_start = None        # первое update — отсчёт «нет сканов», если их не было вовсе
        self.off = np.zeros(2)     # поправка выхода lidar
        self.t_off = 0.0
        self.keep_off = False      # поправка больше max_correction — не затухает
        self.anchor = None         # (выход, legs xy) в момент перехода на legs
        self.t_resume = None       # с какого момента сканы снова свежие (в режиме legs)

    def no_scan_s(self, t, t_scan):
        """Сколько с sim нет сканов (с первого update, если сканов не было вовсе)."""
        ref = t_scan if t_scan is not None else self.t_start
        return 0.0 if ref is None else max(0.0, t - ref)

    def _fresh(self, t, t_scan, lidar_t):
        if t_scan is None or t - t_scan > self.scan_timeout:
            return False
        return lidar_t is None or t - lidar_t <= self.scan_timeout

    def update(self, t, lidar_xy, legs_xy, t_scan, lidar_t=None):
        """t — время выхода (sim); lidar_xy — поза ядра lidar (или None); legs_xy — поза legs (или None);
        t_scan — время прихода последнего скана (None — не было); lidar_t — штамп позы lidar (возраст рабочего потока).
        Возвращает xy выхода (2,) или None, пока выхода нет."""
        if not math.isfinite(t):
            return self.out
        if self.t is not None and t < self.t - 1e-9:
            self.reset()
            self.resets += 1
        if t_scan is not None and t_scan > t:      # скан «из будущего» (штамп раньше часов не бывает) — считаем t
            t_scan = t
        if self.t_start is None:
            self.t_start = t
        self.t = t
        lidar_xy = None if lidar_xy is None or not np.all(np.isfinite(lidar_xy)) else np.array(lidar_xy, float)[:2]
        legs_xy = None if legs_xy is None or not np.all(np.isfinite(legs_xy)) else np.array(legs_xy, float)[:2]
        fresh = lidar_xy is not None and self._fresh(t, t_scan if t_scan is not None else self.t_start, lidar_t)

        if self.source == 'lidar':
            if lidar_xy is not None:
                self.out = lidar_xy + self.correction(t)
            elif self.out is None and legs_xy is not None and t - self.t_start > self.scan_timeout:
                self.out = legs_xy.copy()       # ядро lidar так и не дало позы — выход сразу по legs
            if not fresh and legs_xy is not None and self.out is not None:
                self._to_legs(legs_xy)          # с этого тика — приращения legs от текущего выхода
            return None if self.out is None else self.out.copy()

        # source == 'legs'
        if fresh:
            self.t_resume = t if self.t_resume is None else self.t_resume
        else:
            self.t_resume = None
        if legs_xy is not None:
            out0, legs0 = self.anchor
            self.out = out0 + (legs_xy - legs0)
        if fresh and t - self.t_resume >= self.resume_delay:
            self.source = 'lidar'
            self.switches += 1
            self.off, self.t_off = self.out - lidar_xy, t
            self.keep_off = float(np.linalg.norm(self.off)) > self.max_correction
            self.t_resume = None
        return self.out.copy()

    def correction(self, t):
        """Поправка к позе lidar в момент t."""
        if self.keep_off:
            return self.off.copy()
        return self.off * math.exp(-max(t - self.t_off, 0.0) / self.blend_tau)

    def _to_legs(self, legs_xy):
        self.source = 'legs'
        self.switches += 1
        self.anchor = (self.out.copy(), legs_xy.copy())
        self.t_resume = None
