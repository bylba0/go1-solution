#!/usr/bin/env python3
"""Проверка solution_manifest.json по протоколу ping (ТЗ_T1, 3.3). Запускается при сборке образа: ошибка → сборка падает.

  validate_manifest.py <manifest.json> <каталог launch-файлов go1_bringup>

Требования:
  * ровно 4 поля: protocol_version, supported_tasks, self_trained_rl, preferred_speed_mps (без повторов ключей);
  * protocol_version — строка "1.0"; self_trained_rl — bool;
  * supported_tasks — список уникальных строк из T2..T5;
  * preferred_speed_mps — число в [0.1, 0.6], если заявлен T3, иначе null;
  * у каждого заявленного задания есть launch-файл <каталог>/t<N>.launch.py (его запускает /entrypoint.sh TN).
Только стандартная библиотека.
"""
import json
import os
import sys

FIELDS = {'protocol_version', 'supported_tasks', 'self_trained_rl', 'preferred_speed_mps'}
TASKS = ('T2', 'T3', 'T4', 'T5')


def validate(manifest_path, launch_dir):
    errors = []
    with open(manifest_path, encoding='utf-8') as f:
        pairs = json.load(f, object_pairs_hook=list)       # список пар: видим и повторяющиеся ключи
    if not isinstance(pairs, list) or not all(isinstance(p, tuple) or isinstance(p, list) for p in pairs):
        return ['манифест должен быть JSON-объектом']
    keys = [k for k, _ in pairs]
    m = dict(pairs)
    if len(keys) != 4 or set(keys) != FIELDS:
        errors.append(f'нужны ровно 4 поля {sorted(FIELDS)}, получено {keys}')
    if m.get('protocol_version') != '1.0' or type(m.get('protocol_version')) is not str:
        errors.append('protocol_version должен быть строкой "1.0"')
    if type(m.get('self_trained_rl')) is not bool:
        errors.append('self_trained_rl должен быть true/false')
    tasks = m.get('supported_tasks')
    if type(tasks) is not list or not all(type(t) is str and t in TASKS for t in tasks) or len(set(tasks)) != len(tasks):
        errors.append(f'supported_tasks — список уникальных значений из {list(TASKS)}, получено {tasks!r}')
        tasks = [t for t in tasks or [] if type(t) is str] if type(tasks) is list else []
    v = m.get('preferred_speed_mps')
    if 'T3' in tasks:
        if type(v) not in (int, float) or not (0.1 <= v <= 0.6):   # type(): bool не считается числом
            errors.append(f'preferred_speed_mps при T3 — число в [0.1, 0.6], получено {v!r}')
    elif v is not None:
        errors.append(f'preferred_speed_mps без T3 должен быть null, получено {v!r}')
    for t in tasks:
        path = os.path.join(launch_dir, f't{t[1:]}.launch.py')
        if t in TASKS and not os.path.isfile(path):
            errors.append(f'для заявленного {t} нет launch-файла {path}')
    return errors


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        errors = validate(sys.argv[1], sys.argv[2])
    except (OSError, ValueError) as e:
        errors = [f'не удалось прочитать манифест: {e}']
    if errors:
        for e in errors:
            print(f'solution_manifest.json: {e}', file=sys.stderr)
        return 1
    print(f'solution_manifest.json: OK ({sys.argv[1]})', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
