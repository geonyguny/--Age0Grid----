# project/data/loader.py
from __future__ import annotations
import os, json, hashlib, re
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# NEW: schema validator
from project.data.schema_checks import assert_market_csv_valid

REQUIRED_MIN = ["date", "ret_kr_eq", "cpi_kr", "rf_kr_nom"]
_WINDOW_RE = re.compile(r"^(?:\d{4}-\d{2})?:(?:\d{4}-\d{2})?$")

# --------------------------
# Helpers
# --------------------------
def _hash_key(path: str, asset: str, use_real_rf: str, window: Optional[str]) -> str:
    """파일 변경(경로/mtime/size) + 파라미터(asset/use_real_rf/window)에 종속된 캐시 키."""
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
    """'YYYY-MM:YYYY-MM' / 'YYYY-MM:' / ':YYYY-MM' 포맷 슬라이싱."""
    if not window:
        return df
    s = str(window).strip()
    if not _WINDOW_RE.match(s):
        raise ValueError(f"--data_window 형식 오류: '{window}' (예: 1999-01:2024-12, '1999-01:', ':2024-12')")
    a, b = s.split(":")
    if a:
        df = df[df["date"] >= a]
    if b:
        df = df[df["date"] <= b]
    return df

def _to_monthly_rate_like(x: np.ndarray) -> np.ndarray:
    """
    지수→월간률 변환; 이미 월간률(대략 절대값 중앙값이 작음)이면 그대로.
    결과/NaN/Inf는 0으로 치환.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    # 지수/율 휴리스틱: 값이 크거나 변동이 크면 지수로 간주
    is_index_like = (np.nanmax(arr) > 5.0) or (np.nanmedian(np.abs(arr)) > 0.2)
    if is_index_like and arr.size >= 2:
        r = np.empty_like(arr, dtype=float)
        with np.errstate(all="ignore"):
            r[1:] = arr[1:] / arr[:-1] - 1.0
        r[0] = r[1] if arr.size > 1 and np.isfinite(r[1]) else 0.0
        return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _as_float_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """열을 float np.ndarray로 강제 변환(+NaN/Inf→0). 없으면 전부 NaN 반환."""
    if col not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, copy=False)
    return v

def _usd_to_krw_ret(r_usd: np.ndarray, fx_ret: Optional[np.ndarray]) -> np.ndarray:
    """
    USD 수익률과 USDKRW 월수익률로 KRW 수익률 구성:
      r_krw[t] = (1+r_usd[t]) * (1+fx_ret[t]) - 1
    """
    if r_usd is None or fx_ret is None:
        return np.full_like(_as_np_len(r_usd, fx_ret), np.nan, dtype=float)
    if r_usd.size != fx_ret.size:
        T = min(r_usd.size, fx_ret.size)
        r_usd = r_usd[:T]; fx_ret = fx_ret[:T]
    out = np.empty_like(r_usd, dtype=float)
    out[:] = np.nan
    out[1:] = (1.0 + r_usd[1:]) * (1.0 + fx_ret[1:]) - 1.0
    return out

def _as_np_len(*xs) -> int:
    for x in xs:
        if isinstance(x, np.ndarray):
            return x.size
    return 0

# --------------------------
# Loader
# --------------------------
def load_market_csv(
    path: str,
    asset: str,
    use_real_rf: str = "on",
    data_window: Optional[str] = None,
    cache: bool = True,
) -> Dict[str, np.ndarray]:
    """
    CSV 스키마 v1 (월간):
      필수: date, ret_kr_eq, cpi_kr, rf_kr_nom
      선택: ret_us_eq_usd, ret_gold_usd, usdkrw, ret_us_eq_krw, ret_gold_krw, rf_kr_real

    반환 키:
      dates(str[]), ret_asset, ret_kr_eq, ret_us_eq_krw, ret_gold_krw,
      rf_nom, rf_real, cpi, cpi_rate, ret_fx, ret_fx_usdkrw

    주의:
      - 수익률은 0.01 = +1%
      - CPI는 지수/률 모두 허용(자동 판별)
      - KRW 환산: (1+r_usd)*(1+fx) - 1
      - use_real_rf: "on"이면 rf_real ≈ rf_nom - cpi_rate, "off"이면 rf_real = rf_nom 유지
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"market_csv not found: {path}")

    # 0) 스키마 사전 검증(파일 단위) — 에러 시 명확한 메시지로 실패
    #    기본 required_cols: ['risky_nom','tbill_nom','cpi']지만
    #    본 파이프라인의 v1 필수는 REQUIRED_MIN로도 재체크한다.
    assert_market_csv_valid(path, required_cols=["ret_kr_eq", "cpi_kr", "rf_kr_nom"], date_col="date")

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

    # --- read & normalize columns ---
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # 필수 헤더 확인(로더 층에서도 재확인)
    for c in REQUIRED_MIN:
        if c not in df.columns:
            raise ValueError(f"CSV 누락컬럼: '{c}' (필수: {REQUIRED_MIN})")

    # 날짜 표준화/정렬
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    except Exception:
        # 일부 CSV가 이미 'YYYY-MM' 문자열일 수 있음
        df["date"] = df["date"].astype(str).str.slice(0, 7)
    df = df.sort_values("date").reset_index(drop=True)

    # 기간 슬라이스
    df = _slice_window(df, data_window)
    if len(df) < 24:
        raise ValueError(f"데이터 구간이 짧습니다(>=24 필요). window={data_window}, len={len(df)}")

    # --- FX 월수익률 (usdkrw) ---
    if "usdkrw" in df.columns:
        usdkrw = _as_float_col(df, "usdkrw")
        fx_ret = _to_monthly_rate_like(usdkrw)
    else:
        usdkrw = None
        fx_ret = None

    # --- risky legs ---
    ret_kr_eq = _as_float_col(df, "ret_kr_eq")
    # KRW 열이 있으면 우선 사용, 없으면 USD열+fx로 생성
    ret_us_eq_krw = _as_float_col(df, "ret_us_eq_krw")
    if np.all(np.isnan(ret_us_eq_krw)):
        ret_us_eq_usd = _as_float_col(df, "ret_us_eq_usd")
        if ret_us_eq_usd.size and fx_ret is not None:
            ret_us_eq_krw = _usd_to_krw_ret(ret_us_eq_usd, fx_ret)

    ret_gold_krw = _as_float_col(df, "ret_gold_krw")
    if np.all(np.isnan(ret_gold_krw)):
        ret_gold_usd = _as_float_col(df, "ret_gold_usd")
        if ret_gold_usd.size and fx_ret is not None:
            ret_gold_krw = _usd_to_krw_ret(ret_gold_usd, fx_ret)

    # --- CPI & RF ---
    cpi_col  = _as_float_col(df, "cpi_kr")
    cpi_rate = _to_monthly_rate_like(cpi_col)  # CPI 월간률
    rf_nom   = _as_float_col(df, "rf_kr_nom")

    # use_real_rf 스위치: on → 실질 근사, off → 명목 그대로
    use_real_flag = str(use_real_rf).lower().strip() in ("on", "true", "1", "yes", "y")
    if "rf_kr_real" in df.columns:
        rf_real_src = _as_float_col(df, "rf_kr_real")
        rf_real = rf_real_src if use_real_flag else rf_nom
    else:
        # 실질 근사: rf_real ≈ rf_nom - cpi_rate (log 근사 X, 단순 차감)
        approx_real = np.nan_to_num(rf_nom, nan=0.0) - np.nan_to_num(cpi_rate, nan=0.0)
        rf_real = approx_real if use_real_flag else rf_nom

    # --- 자산 선택 (레거시 호환용 ret_asset) ---
    asset_u = str(asset).upper().strip()
    if asset_u == "KR":
        ret_asset = ret_kr_eq
    elif asset_u == "US":
        if np.all(np.isnan(ret_us_eq_krw)):
            raise ValueError("US 수익률 산출 불가: ret_us_eq_usd/ret_us_eq_krw/usdkrw 중 최소 조합 필요")
        ret_asset = ret_us_eq_krw
    elif asset_u in ("GOLD", "XAU"):
        if np.all(np.isnan(ret_gold_krw)):
            raise ValueError("Gold 수익률 산출 불가: ret_gold_usd/ret_gold_krw/usdkrw 중 최소 조합 필요")
        ret_asset = ret_gold_krw
    else:
        raise ValueError(f"알 수 없는 asset: {asset} (KR|US|GOLD)")

    # --- 출력 사전(+ FX 리턴 포함) ---
    out: Dict[str, np.ndarray] = {
        "dates": df["date"].to_numpy(dtype=str, copy=False),
        "ret_asset": np.nan_to_num(ret_asset, nan=0.0).astype(float),
        "ret_kr_eq": np.nan_to_num(ret_kr_eq, nan=0.0).astype(float),
        "ret_us_eq_krw": np.nan_to_num(ret_us_eq_krw, nan=0.0).astype(float),
        "ret_gold_krw": np.nan_to_num(ret_gold_krw, nan=0.0).astype(float),
        "rf_nom": np.nan_to_num(rf_nom, nan=0.0).astype(float),
        "rf_real": np.nan_to_num(rf_real, nan=0.0).astype(float),
        "cpi": np.nan_to_num(cpi_col, nan=0.0).astype(float),       # 원본 CPI 열(지수/률 혼재 가능)
        "cpi_rate": np.nan_to_num(cpi_rate, nan=0.0).astype(float), # CPI 월간률
        "ret_fx": (np.nan_to_num(fx_ret, nan=0.0).astype(float) if fx_ret is not None
                   else np.full(len(df), np.nan, dtype=float)),
        "ret_fx_usdkrw": (np.nan_to_num(fx_ret, nan=0.0).astype(float) if fx_ret is not None
                          else np.full(len(df), np.nan, dtype=float)),
    }

    if cache:
        np.savez_compressed(cache_npz, **out)
    return out
