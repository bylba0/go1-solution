# Источник модели Unitree Go1 (MJCF)

- Откуда: https://github.com/Urika000/go1-simulator-public, каталог `models/mujoco/unitree_go1/`,
  коммит `96768c1483a0d2a1e471e43c7d6da9ea29cb3522` (публичная симуляция Hackathon Go1).
- Происхождение по ТЗ_T2: MuJoCo Menagerie, revision `0059d4335f8156206f63a35662313385f7ad6d74`, MuJoCo 3.3.7.
- `t2_collision_pairs.csv` — перечень 1540 пар геометрий; проверяются 655 (`diagnostic_policy == checked`).
- Лицензия модели — BSD-3-Clause (`LICENSE`).
- Контрольные суммы — `SHA256SUMS`; сборка образа проверяет их (`sha256sum -c`), при расхождении сборка падает.
