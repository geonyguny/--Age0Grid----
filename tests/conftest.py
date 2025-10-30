# tests/conftest.py
import pytest
from project.runner.cli import _cvar_fallback  # ensure symbol importable

@pytest.fixture
def eval_entrypoint():
    """
    간단 엔트리포인트: (args:list[str]) -> dict
    project.runner.cli 를 통해 실행하고 JSON을 리턴하는 형태로 구현되어 있다면,
    여기서는 테스트에서 필요로 할 최소 형태만 제공.
    """
    import subprocess, sys, json, tempfile, os

    def _run(args):
        cmd = [sys.executable, "-m", "project.runner.cli"] + list(args)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out = r.stdout.strip()
        return json.loads(out) if out else {}
    return _run
