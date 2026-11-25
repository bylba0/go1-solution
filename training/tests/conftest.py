import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/go1_safety'), str(ROOT / 'src/go1_walk')]


@pytest.fixture(scope='session')
def model_dir():
    d = os.environ.get('GO1_MODEL_DIR')
    if not d or not os.path.isfile(os.path.join(d, 'scene.xml')):
        pytest.skip('GO1_MODEL_DIR with scene.xml is not set')
    return d
