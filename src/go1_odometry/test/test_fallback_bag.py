"""Страховка режима lidar на записанном bag B (t4_B_turns_both) с вырезанными сканами — офлайн, без ROS-графа.
Каталог bag — GO1_T4_BAGS; без него, без rosbag2_py или без tools/dev_t4_offline_eval.py (в образе) — пропуск.
Истина /odom/ground_truth читается здесь и в dev-скрипте ТОЛЬКО для метрик; узел её не видит.

Окна (с от первого штампа bag; движение на B — с 11 по 62 с): 15 с — 25…40, 40 с — 15…55; и вовсе без /scan.
Пороги:
  * без пропаданий — как ядро lidar (переключений нет), RPE ≤ 1,5 %;
  * 15 с — RPE ≤ 8 %;
  * 40 с и без сканов — не хуже нашего legs на том же bag + 0,5 п. п.: почти всё движение идёт по legs, поэтому
    лучше legs быть не может; допуск — на первую секунду до переключения (там выход — ядро) и на то, что участки RPE
    по 1 м по-разному делятся на lidar и legs (на B один участок ≈ 1/6 суммы, 0,5 п. п. — заметно меньше);
  * скачки за тик 50 Гц (робот при 0,2 м/с проходит 4 мм за тик):
    - добавка сшивки — сдвиг выхода, не объяснимый движением ни lidar, ни legs: ≤ 5,5 мм. По построению на переходе
      lidar→legs и legs→lidar она 0, дальше — только ввод поправки: ≤ max_correction / blend_tau = 0,25 м/с → 5 мм
      за тик (+10 % запаса);
    - около переключений (±1 с) |Δвыход − Δистина| ≤ 12 мм = 0,02 с × (0,36 м/с — пиковая скорость корпуса на bag при
      команде 0,2 м/с — + 0,25 м/с ввода поправки). Сам legs без переключений даёт за тик до 9–11 мм (дрожь опоры).
    Скачок на расхождение оценщиков (0,1–0,6 м на возврате) дал бы десятки сантиметров за тик.
"""
import os
import pathlib
import sys

import pytest

BAGS = pathlib.Path(os.environ.get('GO1_T4_BAGS', '/nonexistent'))
_UP = pathlib.Path(__file__).resolve().parents
TOOLS = (_UP[3] if len(_UP) > 3 else _UP[-1]) / 'tools'    # в образе пакет смонтирован в /src — tools нет
ADD_MAX = 0.0055
NEAR_MAX = 0.012


@pytest.fixture(scope='module')
def bag_b():
    p = BAGS / 't4_B_turns_both'
    if not p.is_dir():
        pytest.skip(f'нет bag {p} (задайте GO1_T4_BAGS)')
    pytest.importorskip('rosbag2_py')
    if not (TOOLS / 'dev_t4_offline_eval.py').is_file():
        pytest.skip('нет tools/dev_t4_offline_eval.py')
    sys.path.insert(0, str(TOOLS))
    import dev_t4_offline_eval as ev
    d = ev.load_inputs(p, scans=True)
    legs = ev.evaluate(ev.replay(d, {}), d['gt'])['rpe_mean_pct']
    return ev, d, legs


def _fb(bag_b, windows=(), no_scan=False):
    ev, d, _ = bag_b
    info = {}
    est = ev.replay_fallback(ev.drop_scans(d, windows, no_scan), info=info)
    r = ev.evaluate(est, d['gt'])
    jump = ev.switch_jumps(est, d['gt'], info)
    return r, info, jump


def test_no_dropout_is_lidar(bag_b):
    r, info, _ = _fb(bag_b)
    assert info['switches'] == 0 and info['legs_frac'] == 0.0
    assert r['rpe_mean_pct'] <= 1.5, r
    assert r['nan'] == 0 and r['odom_hz'] > 49.9


def test_dropout_15s(bag_b):
    r, info, jump = _fb(bag_b, [(25.0, 15.0)])
    assert info['switches'] == 2
    assert r['rpe_mean_pct'] <= 8.0, r
    assert jump[0] <= ADD_MAX and jump[1] <= NEAR_MAX, jump


def test_dropout_40s(bag_b):
    r, info, jump = _fb(bag_b, [(15.0, 40.0)])
    assert info['switches'] == 2
    assert r['rpe_mean_pct'] <= bag_b[2] + 0.5, (r, bag_b[2])
    assert jump[0] <= ADD_MAX and jump[1] <= NEAR_MAX, jump


def test_no_scans_at_all_not_worse_than_legs(bag_b):
    r, info, jump = _fb(bag_b, no_scan=True)
    assert info['switches'] == 1 and info['legs_frac'] > 0.95
    assert r['rpe_mean_pct'] <= bag_b[2] + 0.5, (r, bag_b[2])
    assert jump[0] <= ADD_MAX and jump[1] <= NEAR_MAX, jump
