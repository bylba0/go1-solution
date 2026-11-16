"""Проверка самоколлизий Go1 для T2 (без ROS).

Модель: go1.xml (MuJoCo Menagerie, MuJoCo 3.3.7) БЕЗ пола. Стандартные правила фильтрации MuJoCo на этой модели
(маски contype/conaffinity, фильтр родитель/потомок, одно сваренное тело) дают ровно те 655 пар, что помечены
`checked` в t2_collision_pairs.csv (проверено тестом test_collision_pairs_match_csv).

Гарантия на ВСЁМ отрезке пути, а не только в точках выборки:
  в нашей копии модели geom_margin поднят до `margin` (5 см), поэтому mj_collision сообщает расстояние d для всех
  пар ближе 5 см (дальше — d ≥ margin). Для каждой геометрии g и сустава j заранее посчитана константа R[g, j] —
  верхняя граница расстояния от оси сустава до любой точки геометрии (сумма длин звеньев по цепочке + rbound).
  При изменении суставов на Δq любая точка геометрии g смещается не больше чем на Σ_j R[g, j]·|Δq_j|, а расстояние
  между двумя геометриями меняется не больше чем на сумму таких смещений. Значит, из точки с расстоянием d можно
  пройти по прямой долю пути ds, при которой d − L·ds ≥ −threshold, и проникновение нигде на этом куске не превысит
  threshold. Вдали от препятствий это 1–2 вычисления на такт, вблизи шаг автоматически мельчает.
"""
import math

import mujoco
import numpy as np

JOINT_NAMES = [f'{leg}_{kind}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for kind in ('hip', 'thigh', 'calf')]

DONE, PARTIAL, BLOCKED = 'done', 'partial', 'blocked'


class CollisionChecker:
    def __init__(self, model_path, threshold=0.0005, stop_threshold=0.00025, margin=0.05, max_evals=400):
        """threshold — допустимое проникновение на пути (м); stop_threshold — точки остановки держим с запасом:
        проникновение в них ≤ stop_threshold, чтобы из них всегда можно было отъехать."""
        assert 0.0 <= stop_threshold < threshold
        self.threshold = float(threshold)
        self.stop_threshold = float(stop_threshold)
        self.margin = float(margin)
        self.max_evals = int(max_evals)
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.m.geom_margin[:] = self.margin          # только в нашей копии: расстояния до 5 см видны в контактах
        self.d = mujoco.MjData(self.m)
        key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, key)
        self.qadr = np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                              for n in JOINT_NAMES])
        self.collidable = np.array([g for g in range(self.m.ngeom)
                                    if self.m.geom_contype[g] or self.m.geom_conaffinity[g]])
        self.R = self._lipschitz_radii()             # (n_collidable, 12)
        self.evals = 0                               # счётчик вычислений (для статистики CPU)

    # ------------------------------------------------------------------ геометрия
    def _lipschitz_radii(self):
        m = self.m
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        R = np.zeros((len(self.collidable), 12))
        for gi, g in enumerate(self.collidable):
            gb = int(m.geom_bodyid[g])
            chain = []                                # тела от геометрии вверх до корня
            b = gb
            while b != 0:
                chain.append(b)
                b = int(m.body_parentid[b])
            for k, j in enumerate(jids):
                jb = int(m.jnt_bodyid[j])
                if jb not in chain:
                    continue                          # сустав не влияет на эту геометрию
                # якорь сустава → начало тела jb → ... → начало тела gb → центр геометрии, + радиус геометрии
                bound = float(np.linalg.norm(m.jnt_pos[j]))
                for body in chain[:chain.index(jb)]:   # тела ниже jb до gb включительно
                    bound += float(np.linalg.norm(m.body_pos[body]))
                bound += float(np.linalg.norm(m.geom_pos[g])) + float(m.geom_rbound[g])
                R[gi, k] = bound
        return R

    def lipschitz(self, direction):
        """Граница скорости изменения расстояния между любыми двумя геометриями на единицу параметра пути
        q(s) = qa + s·direction, s ∈ [0, 1]."""
        g = self.R @ np.abs(np.asarray(direction, dtype=float))
        if g.size < 2:
            return float(g.sum())
        top2 = np.partition(g, -2)[-2:]
        return float(top2.sum())

    # ------------------------------------------------------------------ вычисления
    def min_dist(self, q):
        """Минимальное знаковое расстояние по 655 парам (м); отрицательное — проникновение. Не больше margin."""
        self.evals += 1
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        n = self.d.ncon
        if n == 0:
            return self.margin
        return float(min(self.margin, np.min(self.d.contact.dist[:n])))

    def penetration(self, q):
        """Проникновение (м, ≥ 0) — как у Judge: максимум по проверяемым парам."""
        return max(0.0, -self.min_dist(q))

    def is_safe(self, q, threshold=None):
        thr = self.threshold if threshold is None else threshold
        return self.min_dist(q) > -thr

    def advance(self, qa, qb):
        """Идём по прямой из qa в qb, пока это доказуемо безопасно.

        Возвращает (q, status): DONE — дошли до qb; PARTIAL — исчерпан бюджет вычислений, продолжим в следующем такте;
        BLOCKED — дальше нельзя (следующая точка дала бы проникновение > stop_threshold), q — последняя безопасная.
        На всём пройденном куске проникновение ≤ threshold (граница L), в точке остановки ≤ stop_threshold.
        """
        qa = np.asarray(qa, dtype=float)
        qb = np.asarray(qb, dtype=float)
        delta = qb - qa
        span = float(np.max(np.abs(delta))) if delta.size else 0.0
        if span < 1e-12:
            return qb.copy(), DONE
        L = self.lipschitz(delta)
        s, q = 0.0, qa.copy()
        d = self.min_dist(q)
        if d < -self.threshold:
            return q, BLOCKED                         # старт уже небезопасен — не двигаемся
        min_ds = 1e-6 / span                          # < 0.00006° по максимальному суставу — считаем, что упёрлись
        for _ in range(self.max_evals):
            ds = (d + self.threshold) / L if L > 0 else 1.0
            if ds < min_ds:
                return q, BLOCKED
            s_next = min(1.0, s + ds)
            q_next = qa + s_next * delta
            d_next = self.min_dist(q_next)
            if d_next < -self.stop_threshold:
                return q, BLOCKED                     # точку с проникновением > stop_threshold не принимаем
            s, q, d = s_next, q_next, d_next
            if s >= 1.0:
                return qb.copy(), DONE
        return q, PARTIAL

    def sampled_max_penetration(self, qa, qb, step_rad=math.radians(0.1), model_margin=None):
        """Эталонная проверка как у Judge: точки через ≤ step_rad по максимальному суставу, включая концы.
        Возвращает максимальное проникновение (м). Для тестов и judge_sim."""
        qa = np.asarray(qa, dtype=float)
        qb = np.asarray(qb, dtype=float)
        n = max(1, int(math.ceil(float(np.max(np.abs(qb - qa))) / step_rad)))
        worst = 0.0
        for k in range(n + 1):
            worst = max(worst, self.penetration(qa + (qb - qa) * (k / n)))
        return worst
