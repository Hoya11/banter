import shutil
import subprocess
from pathlib import Path

import pytest


def test_inline_web_state_machine() -> None:
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required for the vanilla web state-machine tests')

    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, '--test', 'web/tests/index-state.test.mjs'],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, (
        'web state-machine tests failed\n'
        f'--- stdout ---\n{result.stdout}\n'
        f'--- stderr ---\n{result.stderr}'
    )
