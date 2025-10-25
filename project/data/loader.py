# project/data/loader.py
from __future__ import annotations

import os
import json
import hashlib
import re
from typing import Dict, Optional
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# Optional schema validator (존재하면 사용, 없으면 no-op)
# ─────────────────────────────────────────────────────────
try:
    from project.data.schema_checks import assert_market_csv_valid  # type: ignore
except Exception:  # pragma: no cover
    def assert_market_csv_valid(path: str, required_cols=None, date_col: str = "date") -> None:
        # 최소한의 검증만 로컬에서 수행 (없으면 조용히 통과)
        import pandas as _pd
        _df = _pd.read_csv(path, nrows=1)
        for c in (required_cols or []):
            if c not in _df.columns:
                raise ValueError(f"[schema] missing column '{c}' in {os.path.basename(path)}")
        if date_col not in _df.columns:
            raise ValueError(f"[schema] missing date column '{date_col}' in {os.path.basename(path)}")

# ─────────────────────────────────────────────────────────
# Regex
# ─────────────────────────────────────────────────────────
_WINDOW_RE = re.compile(r"^(?:\d{4}-\d{2})?:(?:\d{4}-\d{2})?$")

__all__ = ["load_market_csv"]

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def _hash_key(path: str, asset: str, use_real_rf: str, window: Optional[str]) -> str:
    """파일 메타(경로/mtime/size)+파라미터(asset/use_real_rf/window)에 종속된 캐시 키 생성."""
    st = os.stat(path)
    key = {
        "path": os.path.abspath(path),
        "mtime": int(st.st_mtime),
        "size": int(st.st_size),
        "asset": str(asset).upper().strip(),
        "use_real_rf": str(use_real_rf).lower().strip(),
        "window": (window or "").strip(),
    }
    return hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()


def _slice_window(df: pd.DataFrame, window: Optional[str]) -> pd.DataFrame:
    """'YYYY-MM:YYYY-MM' / 'YYYY-MM:' / ':YYYY-MM' 포맷으로 기간 슬라이스."""
    if not window:
        return df
    s = str(window).strip()
    if not _WINDOW_RE.match(s):
        raise ValueError(f"--data_window 형식 오류: '{window}' (예: 1999-01:2024-12, '1999-01:', ':2024-12')")
    a, b = s.split(":")
    # 날짜는 아래에서 YYYY-MM 문자열로 표준화하므로 문자열 비교로도 안전
    if a:
        df = df[df["date"] >= a]
    if b:
        df = df[df["date"] <= b]
    return df.reset_index(drop=True)


def _to_monthly_rate_like(x: np.ndarray) -> np.ndarray:
    """
    지수→월간률 변환; 이미 월간률이면 그대로 반환.
    (휴리스틱) 값이 매우 크거나 중앙 절대값이 큰 경우 지수로 간주.
    NaN/Inf는 0으로 치환.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    with np.errstate(all="ignore"):
        is_index_like = (np.nanmax(arr) > 5.0) or (np.nanmedian(np.abs(arr)) > 0.2)
    if is_index_like and arr.size >= 2:
        r = np.empty_like(arr, dtype=float)
        with np.errstate(all="ignore"):
            r[1:] = arr[1:] / arr[:-1] - 1.0
        r[0] = r[1] if arr.size > 1 and np.isfinite(r[1]) else 0.0
        return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _as_float_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """열을 float ndarray로 강제 변환(+NaN/Inf→NaN 유지). 없으면 전부 NaN."""
    if col not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, copy=False)


def _usd_to_krw_ret(r_usd: Optional[np.ndarray], fx_ret: Optional[np.ndarray]) -> np.ndarray:
    """
    USD 수익률과 USDKRW 월수익률로 KRW 수익률 구성:
      r_krw[t] = (1+r_usd[t]) * (1+fx_ret[t]) - 1
    길이가 다르면 공통 최소 길이에 맞춰 절단.
    """
    if r_usd is None or fx_ret is None:
        return np.array([], dtype=float)
    r_usd = np.asarray(r_usd, dtype=float)
    fx_ret = np.asarray(fx_ret, dtype=float)
    if r_usd.size == 0 or fx_ret.size == 0:
        return np.array([], dtype=float)
    T = min(r_usd.size, fx_ret.size)
    out = (1.0 + r_usd[:T]) * (1.0 + fx_ret[:T]) - 1.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _parse_on_flag(v: str) -> bool:
    return str(v).strip().lower() in {"on", "true", "1", "yes", "y"}


def _read_csv_forgiving(p: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(p, encoding=enc, engine="python")
        except Exception:
            continue
    return pd.read_csv(p)


# ─────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────
def load_market_csv(
    path: str,
    asset: str = "KR",
    use_real_rf: str = "on",
    data_window: Optional[str] = None,
    cache: bool = True,
) -> Dict[str, np.ndarray]:
    """
    CSV schema v1 (월간):
      권장 표준 컬럼: date, ret_kr_eq, cpi, rf_nom
      허용 별칭    : cpi_kr→cpi, rf_kr_nom→rf_nom, ret_fx_usdkrw→ret_fx
      선택 컬럼    : ret_us_eq_usd, ret_gold_usd, usdkrw, ret_us_eq_krw, ret_gold_krw, rf_real

    반환 키:
      dates(str[]), ret_asset, ret_kr_eq, ret_us_eq_krw, ret_gold_krw,
      rf_nom, rf_real, cpi, cpi_rate, ret_fx, ret_fx_usdkrw

    주의:
      - 수익률은 0.01 = +1%
      - CPI는 지수/률 모두 허용(자동 판별)
      - KRW 환산: (1+r_usd)*(1+fx) - 1
      - use_real_rf: "on"이면 rf_real ≈ rf_nom - cpi_rate (CSV에 rf_real 있으면 우선)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"market_csv not found: {path}")

    # 0) 스키마 검증(표준 컬럼 기준; 별칭 허용은 schema_checks에서 처리)
    assert_market_csv_valid(path, required_cols=["ret_kr_eq", "cpi", "rf_nom"], date_col="date")

    # 1) 캐시 준비
    cache_dir = os.path.join(os.path.dirname(path), "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _hash_key(path, asset, use_real_rf, data_window)
    cache_npz = os.path.join(cache_dir, f"{cache_key}.npz")

    if cache and os.path.exists(cache_npz):
        z = np.load(cache_npz, allow_pickle=True)
        out = {k: z[k] for k in z.files}
        if "dates" in out and out["dates"].dtype.kind == "O":
            out["dates"] = out["dates"].astype(str)
        return out  # type: ignore[return-value]

    # 2) CSV 로드
    df = _read_csv_forgiving(path)
    df.columns = [str(c).strip() for c in df.columns]

    # 3) 별칭 → 표준 컬럼 정규화 (표준이 이미 있으면 보존)
    alias_renames = {
        "cpi_kr": "cpi",
        "rf_kr_nom": "rf_nom",
        "ret_fx_usdkrw": "ret_fx",
    }
    for src, dst in alias_renames.items():
        if dst not in df.columns and src in df.columns:
            df = df.rename(columns={src: dst})

    # 4) 날짜 표준화/정렬 (YYYY-MM 문자열)
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    except Exception:
        df["date"] = df["date"].astype(str).str.slice(0, 7)
    df = df.sort_values("date").reset_index(drop=True)

    # 5) 기간 슬라이스
    df = _slice_window(df, data_window)
    if len(df) < 24:
        raise ValueError(f"데이터 구간이 짧습니다(>=24 필요). window={data_window}, len={len(df)}")

    # 6) FX 월수익률 (우선순위: ret_fx ▶ ret_fx_usdkrw 별칭 ▶ usdkrw 레벨 변환)
    fx_ret: Optional[np.ndarray] = None
    if "ret_fx" in df.columns:
        fx_ret = _as_float_col(df, "ret_fx")
    elif "ret_fx_usdkrw" in df.columns:
        fx_ret = _as_float_col(df, "ret_fx_usdkrw")
    elif "usdkrw" in df.columns:
        usdkrw = _as_float_col(df, "usdkrw")
        fx_ret = _to_monthly_rate_like(usdkrw)

    # 7) risky legs
    ret_kr_eq = _as_float_col(df, "ret_kr_eq")

    # US: KRW 열 우선, 없으면 USD+FX로 생성
    ret_us_eq_krw = _as_float_col(df, "ret_us_eq_krw")
    if np.all(np.isnan(ret_us_eq_krw)):
        ret_us_eq_usd = _as_float_col(df, "ret_us_eq_usd")
        conv = _usd_to_krw_ret(ret_us_eq_usd, fx_ret) if fx_ret is not None else np.array([], dtype=float)
        if conv.size:
            ret_us_eq_krw = conv
        else:
            ret_us_eq_krw = np.full(len(df), np.nan, dtype=float)

    # GOLD: KRW 열 우선, 없으면 USD+FX로 생성
    ret_gold_krw = _as_float_col(df, "ret_gold_krw")
    if np.all(np.isnan(ret_gold_krw)):
        ret_gold_usd = _as_float_col(df, "ret_gold_usd")
        conv = _usd_to_krw_ret(ret_gold_usd, fx_ret) if fx_ret is not None else np.array([], dtype=float)
        if conv.size:
            ret_gold_krw = conv
        else:
            ret_gold_krw = np.full(len(df), np.nan, dtype=float)

    # 8) CPI & RF
    cpi_col = _as_float_col(df, "cpi")                  # 지수/률 혼재 허용
    cpi_rate = _to_monthly_rate_like(cpi_col)           # CPI 월간률
    rf_nom = _as_float_col(df, "rf_nom")

    use_real_flag = _parse_on_flag(use_real_rf)
    if "rf_real" in df.columns:
        rf_real_src = _as_float_col(df, "rf_real")
        rf_real = rf_real_src if use_real_flag else rf_nom
    else:
        approx_real = np.nan_to_num(rf_nom, nan=0.0) - np.nan_to_num(cpi_rate, nan=0.0)
        rf_real = approx_real if use_real_flag else rf_nom

    # 9) 자산 선택 (레거시 호환용 ret_asset)
    asset_u = str(asset).upper().strip()
    if asset_u == "KR":
        ret_asset = ret_kr_eq
    elif asset_u == "US":
        if np.all(np.isnan(ret_us_eq_krw)):
            raise ValueError("US 수익률 산출 불가: ret_us_eq_usd/ret_us_eq_krw/(ret_fx|usdkrw) 중 최소 조합 필요")
        ret_asset = ret_us_eq_krw
    elif asset_u in {"GOLD", "XAU"}:
        if np.all(np.isnan(ret_gold_krw)):
            raise ValueError("Gold 수익률 산출 불가: ret_gold_usd/ret_gold_krw/(ret_fx|usdkrw) 중 최소 조합 필요")
        ret_asset = ret_gold_krw
    else:
        raise ValueError(f"알 수 없는 asset: {asset} (KR|US|GOLD)")

    # 10) 출력 (모든 수치 NaN→0 치환은 ‘최종’ 단계에서만 수행)
    def _nz(a: np.ndarray) -> np.ndarray:
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=False)

    out: Dict[str, np.ndarray] = {
        "dates": df["date"].to_numpy(dtype=str, copy=False),
        "ret_asset": _nz(ret_asset),
        "ret_kr_eq": _nz(ret_kr_eq),
        "ret_us_eq_krw": _nz(ret_us_eq_krw),
        "ret_gold_krw": _nz(ret_gold_krw),
        "rf_nom": _nz(rf_nom),
        "rf_real": _nz(rf_real),
        "cpi": _nz(cpi_col),          # 원본 CPI (지수/률 혼재 가능)
        "cpi_rate": _nz(cpi_rate),    # CPI 월간률
        "ret_fx": _nz(fx_ret) if fx_ret is not None else np.full(len(df), np.nan, dtype=float),
        "ret_fx_usdkrw": _nz(fx_ret) if fx_ret is not None else np.full(len(df), np.nan, dtype=float),
    }

    # 11) 캐시 저장
    if cache:
        np.savez_compressed(cache_npz, **out)

    return out
