# tests/conftest.py
import pytest
from project.runner.cli import _cvar_fallback  # ensure symbol importable

@pytest.fixture
def eval_entrypoint():
    """
    통합 엔트리포인트:
    - 키워드 인자: eval_entrypoint(method="hjb", market_mode="bootstrap", ...)
    - dict 1개 위치 인자: eval_entrypoint({"method":"hjb", ...})
    - argv(list/tuple) 1개 위치 인자: eval_entrypoint(["--method","hjb", ...])
    """
    from project.runner.cli import eval_entrypoint as factory
    inner = factory()

    def _run(*args, **kwargs):
        # 1) dict를 첫 인자로 받으면 kwargs로 병합
        if args and isinstance(args[0], dict):
            kwargs = {**args[0], **kwargs}
            args = args[1:]

        # 2) argv(list/tuple)로 들어오면 서브프로세스로 모듈 실행
        if args and isinstance(args[0], (list, tuple)):
            import subprocess, sys, json
            argv = list(args[0])
            cmd = [sys.executable, "-m", "project.runner.cli", *argv]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            out = r.stdout.strip()
            return json.loads(out) if out else {}

        # 3) 기본: 키워드 인자를 그대로 내부 eval_entrypoint에 위임
        return inner(**kwargs)

    return _run
