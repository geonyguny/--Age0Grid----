# project/run_experiment.py
import datetime
import os
import csv
import sys
import io

def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

def append_metrics_csv(path: str, payload: dict):
    # metrics & args 뽑기
    m   = payload.get('metrics') or {}
    arg = payload.get('args') or {}

    # 한 행(row) 구성: 기존 + 소비/연금 컬럼
    row = {
        'ts': now_iso(),
        'asset': payload.get('asset'),
        'method': payload.get('method'),
        'lambda': payload.get('lambda_term'),
        'F_target': payload.get('F_target'),
        'alpha': payload.get('alpha'),

        'ES95': m.get('ES95'),
        'EW': m.get('EW'),
        'EL': m.get('EL'),
        'Ruin': m.get('Ruin'),
        'mean_WT': m.get('mean_WT'),

        'HedgeHit': m.get('HedgeHit'),
        'HedgeKMean': m.get('HedgeKMean'),
        'HedgeActiveW': m.get('HedgeActiveW'),

        'fee_annual': payload.get('fee_annual'),
        'w_max': payload.get('w_max'),
        'horizon_years': payload.get('horizon_years'),
        'seeds': arg.get('seeds'),
        'n_paths': arg.get('n_paths'),
        'mortality_on': (arg.get('mortality') == 'on') if isinstance(arg.get('mortality'), str) else False,
        'market_mode': arg.get('market_mode'),

        # --- 소비 지표(신규) ---
        'p10_c_last': m.get('p10_c_last'),
        'p50_c_last': m.get('p50_c_last'),
        'p90_c_last': m.get('p90_c_last'),
        'C_ES95_avg': m.get('C_ES95_avg'),

        # --- 연금 오버레이(신규) ---
        'ann_on': arg.get('ann_on'),
        'ann_alpha': arg.get('ann_alpha'),
        'ann_L': arg.get('ann_L'),
        'ann_d': arg.get('ann_d'),
        'ann_index': arg.get('ann_index'),
        'y_ann': m.get('y_ann'),
        'a_factor': m.get('a_factor'),
        'P': m.get('P'),
    }

    # 헤더 결정: 파일이 있으면 기존 헤더 재사용(형식 충돌 방지), 없으면 새 헤더로 생성
    fieldnames = list(row.keys())
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not write_header:
        try:
            with open(path, 'r', encoding='utf-8') as rf:
                r = csv.reader(rf)
                old_header = next(r)
                if old_header:
                    fieldnames = old_header
        except Exception:
            fieldnames = list(row.keys())

    safe_row = {k: row.get(k) for k in fieldnames}

    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(safe_row)

def _make_console_unicode_safe():
    """
    Windows CP949 콘솔 등에서 argparse --help 출력 시
    U+2212(−) 같은 문자로 인코딩 오류가 나는 것을 방지.
    - 가능하면 현재 인코딩 유지 + errors='replace'
    - 불가 시 UTF-8로 재래핑
    - pytest 캡처(StringIO) 환경은 건드리지 않음
    """
    # stdout
    try:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        # reconfigure가 있으면 그대로 사용
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding=enc, errors="replace")  # type: ignore[attr-defined]
        elif hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=enc, errors="replace")  # type: ignore[assignment]
    except Exception:
        # 최후 수단: UTF-8
        try:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")  # type: ignore[assignment]
        except Exception:
            pass

    # stderr
    try:
        enc = getattr(sys.stderr, "encoding", None) or "utf-8"
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding=enc, errors="replace")  # type: ignore[attr-defined]
        elif hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=enc, errors="replace")  # type: ignore[assignment]
    except Exception:
        try:
            if hasattr(sys.stderr, "buffer"):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")  # type: ignore[assignment]
        except Exception:
            pass

def main():
    # 콘솔 인코딩 보호막 설치 (help 출력 포함 모든 경로에서 안전)
    _make_console_unicode_safe()

    # 위임 엔트리: 기존 스크립트 호출 호환 유지
    # - argparse가 --help로 정상 종료(SystemExit(0))해도 그대로 통과
    from .runner.cli import main as _main
    _main()

if __name__ == "__main__":
    main()
