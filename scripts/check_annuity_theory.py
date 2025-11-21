import pandas as pd
import numpy as np
from pathlib import Path


def load_kidi_qx(csv_path: str = "kidi_qx.csv") -> pd.DataFrame:
    """
    KIDI qx 파일을 읽어서 DataFrame 반환.
    컬럼: age, male, female
    """
    df = pd.read_csv(csv_path)
    # 정렬 및 기본 체크
    df = df.sort_values("age").reset_index(drop=True)
    required_cols = {"age", "male", "female"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"필수 컬럼 {required_cols} 이(가) 없습니다. 실제 컬럼: {df.columns.tolist()}")
    return df


def build_survival(df: pd.DataFrame,
                   sex: str = "M",
                   age0: int = 55,
                   horizon_years: int = 45):
    """
    주어진 sex/age0/horizon_years 에 대해
    - qx  : q_x, q_{x+1}, ..., q_{x+T-1}
    - px  : 1 - qx
    - tpx : t=0..T 에 대한 생존확률 {}_t p_x (t=0일 때 1.0)
    를 반환한다.
    """
    sex = sex.upper()
    col = "male" if sex == "M" else "female"

    # age0 ~ age0 + horizon_years 구간 추출
    df_sub = df[(df["age"] >= age0) & (df["age"] <= age0 + horizon_years)].copy()
    df_sub = df_sub.dropna(subset=[col])

    qx = df_sub[col].to_numpy()
    # horizon_years 보다 길면 잘라줌 (보통 같거나 더 길 것)
    if len(qx) > horizon_years:
        qx = qx[:horizon_years]

    px = 1.0 - qx  # 각 연령구간의 생존확률

    # tpx: t=0..T 에 대해 {}_t p_x (T = len(px))
    T = len(px)
    tpx = np.empty(T + 1)
    tpx[0] = 1.0
    for t in range(1, T + 1):
        tpx[t] = tpx[t - 1] * px[t - 1]

    ages = np.arange(age0, age0 + T + 1, 1)  # t=0일 때 age0, t=T일 때 age0+T
    return {
        "age0": age0,
        "T": T,
        "ages": ages,
        "qx": qx,
        "px": px,
        "tpx": tpx,  # 길이 T+1
    }


def annuity_value(df: pd.DataFrame,
                  sex: str = "M",
                  age0: int = 55,
                  horizon_years: int = 45,
                  r: float = 0.03,
                  fee: float = 0.004,
                  method: str = "payment",
                  when: str = "immediate") -> float:
    """
    생명표 + (r, fee)를 이용해 종신연금 현재가치(이론값)를 계산한다.

    Parameters
    ----------
    sex : 'M' or 'F'
    age0 : 연금개시 연령 (예: 55)
    horizon_years : 시뮬레이션 구간 (예: 45년 => 100세까지)
    r : 실질 할인율(연 투자수익률)
    fee : 연간 사업비 (예: 0.004 = 40bp)
    method:
        - 'payment' : 매년 지급되는 연금액에 (1 - fee)를 곱해 순연금액으로 보는 방식
        - 'discount': 할인율을 r + fee 로 올려서, 사업비를 수익률 차감으로 보는 방식
    when:
        - 'immediate' : 연말 지급형(annuity-immediate)  → t=1..T 에 대해 {}_{t-1}p_x * v^t
        - 'due'       : 선급 지급형(annuity-due)        → t=0..T-1 에 대해 {}_t p_x * v^t

    Returns
    -------
    float
        1원을 일시납했을 때, 매년 1원씩(또는 (1-fee)원씩) 받는 종신연금의 현재가치
    """
    surv = build_survival(df, sex=sex, age0=age0, horizon_years=horizon_years)
    T = surv["T"]
    tpx = surv["tpx"]  # 길이 T+1

    if method not in ("payment", "discount"):
        raise ValueError("method 는 'payment' 또는 'discount' 중 하나여야 합니다.")

    if when not in ("immediate", "due"):
        raise ValueError("when 은 'immediate' 또는 'due' 중 하나여야 합니다.")

    # 사업비 반영 방식
    if method == "discount":
        r_eff = r + fee
        pay_factor = 1.0
    else:  # 'payment'
        r_eff = r
        pay_factor = 1.0 - fee

    v = 1.0 / (1.0 + r_eff)

    pv = 0.0
    if when == "immediate":
        # t=1..T, 생존확률은 {}_{t-1}p_x
        for t in range(1, T + 1):
            pv += tpx[t - 1] * (v ** t)
    else:  # 'due'
        # t=0..T-1, 생존확률은 {}_t p_x, 선급 지급
        for t in range(0, T):
            pv += tpx[t] * (v ** t)

    return pay_factor * pv


def main():
    # 예시: kidi_qx.csv 는 이 스크립트와 같은 폴더나 상위폴더에 둔다
    csv_path = Path(__file__).resolve().parent.parent / "kidi_qx.csv"
    df = load_kidi_qx(csv_path)

    # 설정 값 (필요시 수정)
    age0 = 55
    horizon_years = 45
    r = 0.03     # 실질 할인율 가정(예시)
    fee = 0.004  # 연 40bp

    for sex in ("M", "F"):
        for when in ("immediate", "due"):
            v_pay = annuity_value(
                df, sex=sex, age0=age0,
                horizon_years=horizon_years,
                r=r, fee=fee,
                method="payment",   # 사업비를 연금액 축소로 반영
                when=when
            )
            v_disc = annuity_value(
                df, sex=sex, age0=age0,
                horizon_years=horizon_years,
                r=r, fee=fee,
                method="discount",  # 사업비를 할인율 상향으로 반영
                when=when
            )
            print(
                f"sex={sex}, when={when}: "
                f"payment-method={v_pay:.4f}, discount-method={v_disc:.4f}"
            )


if __name__ == "__main__":
    main()
