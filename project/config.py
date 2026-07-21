# project/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, Dict, Sequence, Union

# === CVaR calibration defaults (used by CLI defaults) ===
CVAR_TARGET_DEFAULT: Optional[float] = None   # e.g., 0.45
CVAR_TOL_DEFAULT: float = 0.01
LAMBDA_MIN_DEFAULT: float = 0.0
LAMBDA_MAX_DEFAULT: float = 5.0
LAMBDA_MAX_ITER: int = 14

# === Hedge defaults (MVP-level toggle) ===
HEDGE_ON_DEFAULT: bool = False
HEDGE_MODE_DEFAULT: Literal["mu", "sigma", "downside"] = "sigma"
HEDGE_COST_DEFAULT: float = 0.005                   # annual premium / haircut
HEDGE_SIGMA_K_DEFAULT: float = 0.20                 # fraction reduction on σ

# 타입 헬퍼
FloatGrid = Union[Sequence[float], Tuple[float, ...]]


@dataclass
class SimConfig:
    # --- Core asset/market settings ---
    asset: Literal["KR", "US", "Gold"] = "KR"
    mu_annual: float = 0.06
    sigma_annual: float = 0.20
    rf_annual: float = 0.02

    # --- Horizon / time resolution ---
    horizon_years: int = 35
    steps_per_year: int = 12

    # --- Controls / constraints ---
    w_max: float = 0.70
    # 과거(phi_adval)와 최신(fee_annual) 동시 지원
    fee_annual: float = 0.004       # ad-valorem fee (annual)
    phi_adval: Optional[float] = None  # 구버전 alias; 주어지면 fee_annual 대신 사용
    allow_short: bool = False
    allow_leverage: bool = False

    # --- Floor (level) ---
    floor_on: bool = False
    f_min_real: float = 0.0  # level floor per period in wealth units

    # --- Objective (CVaR@alpha with RU-dual at terminal) ---
    alpha: float = 0.95
    lambda_term: float = 0.0
    F_target: float = 0.0  # loss 정의용 목표 최종자산 (wealth 모드일 땐 참고값)

    # --- Baseline (rule-based) options ---
    baseline: Optional[Literal["4pct", "cpb", "vpw", "kgr"]] = None
    p_annual: float = 0.04          # constant-%-of-balance (annual)
    g_real_annual: float = 0.02     # VPW growth assumption (real)
    w_fixed: Optional[float] = None # fixed risky weight for rules

    # --- RL placeholders ---
    rl_lr: float = 3e-4
    rl_gamma: float = 0.996
    rl_epochs: int = 60
    rl_steps_per_epoch: int = 2048
    rl_hidden: int = 64
    rl_q_cap: float = 0.0

    # --- HJB grids (wealth / action / eta) ---
    hjb_W_min: float = 0.0
    hjb_W_max: float = 5.0
    hjb_W_grid: int = 121                         # 논문 기본 해상도(201도 가능)
    # 2026-07 추가: 저자산 구간 격자 집중 옵션(이산화 아티팩트 방지).
    # W_focus 이하에 격자점의 focus_frac 비율을 몰아주고, 그 위쪽(W_focus~W_max)은
    # 성기게 배치한다. 실제 인출 경로가 거의 벗어나지 않는 범위(대략 초기자산의
    # 1.5~2.5배)로 W_focus를 잡으면, 기존 W_max=5~10 균일격자에서 발생했던
    # "저자산 구간에서 위험자산비중이 0으로 잘못 수렴"하는 문제가 사라진다.
    hjb_W_focus: Optional[float] = 2.0
    hjb_W_focus_frac: float = 0.75
    # w-grid: 자동 생성(0~w_max, 균등분할). 직접 지정 시 리스트/튜플 모두 허용.
    hjb_w_grid: Optional[FloatGrid] = None
    hjb_w_grid_n: int = 8                         # 0~w_max를 n등분 (기본 8점)
    # η-grid 자동 생성을 위해 기본은 비움 → __post_init__에서 0..F_target 구간 linspace
    hjb_eta_grid: Tuple[float, ...] = field(default_factory=tuple)
    hjb_eta_n: int = 81                           # η 격자 점수(기본 81; dev 41~61 권장)

    # MC samples for expectation inside HJB
    hjb_Nshock: int = 1024
    hjb_q_max_mult: float = 1.0   # 소비율 격자 상한 배율(1.0=기존 4%룰 상한과 동일, 하위호환)

    # --- 국민연금(소득대체율 ρ) 외생소득 [2026-07 신규, 2026-07 개정] ---
    # pension_rho: 실질 소득대체율 ρ (0이면 국민연금 미반영).
    #   확정 정책실험 구간: {0.20, 0.30, 0.40} (실질 기준, 명목 소득대체율 43%와는 다른 개념)
    #   근거: 한정림·이항석(2013, 한국데이터분석학회지) 국민연금 실질 소득대체율 추정치
    #   (약 21~23%), 한국일보(2025.4) 보도 기준 현재 체감 실질 대체율(약 30%).
    # pension_income_mult: 은퇴 전 연소득 / W0 배율(X).
    #   [개정] 은퇴직전 소득 기준값을 가계 평균소득이 아니라 "국민연금 A값"
    #   (전체 가입자 평균소득월액, 2025년 309만원/월=3,708만원/년)으로 확정.
    #   X = 2025년 가계금융복지조사 가구 평균 금융자산(1억 3,690만원, 부동산 제외)
    #       / A값 연환산(3,708만원) ≈ 3.69
    #   (A값은 국민연금 급여 산정에 실제 쓰이는 "제도상 대표소득"이라, 가계 전체
    #   평균소득보다 ρ의 정의와 정합적임)
    # pension_claim_age: 국민연금 수급개시 연령(65세로 고정)
    # 주의: 본 모형은 전 과정 세전(before-tax) 금액 기준이며, 연금소득세 등 세금
    #   효과는 고려하지 않는다(논문 서론/3장에 명시 필요, 향후연구 과제).
    pension_rho: float = 0.0
    pension_income_mult: float = 3.692
    pension_claim_age: float = 65.0
    regret_c_ref_rate: Optional[float] = None  # 후회회피 기준소비율(연). None=기본 4%룰
    # 확률왜곡(Prelec 1998, 식34/35): φ(p)=exp[-η(-ln p)^α]. 둘 다 1.0이면 왜곡 없음(기존과 동일)
    prob_weight_alpha: float = 1.0
    prob_weight_eta: float = 1.0
    # 학습 중 무작위 연금매입 이벤트 (RL이 "연금매입으로 인한 소득변화"를 실제로 경험하게 함)
    train_random_annuity: str = "off"
    train_annuity_prob: float = 0.5
    train_annuity_theta_max: float = 0.8
    train_annuity_age_min: float = 55.0
    train_annuity_age_max: float = 65.0
    train_annuity_load: float = 0.08
    # [2026-07 신규] 사회안전망(기초생활보장): floor_on/f_min_real(인출전략 제약,
    # 기존 의도)과는 별개의 독립적 메커니즘. 본인 자산과 무관하게 정부가 부족분을
    # 외생적으로 채워주는 방식(계정 비차감).
    social_floor_on: str = "off"
    social_floor_min: float = 0.0
    social_floor_asset_test: float = 0.0   # 자산조사 기준(W0 정규화 단위, 기본 0=완전소진 시만)
    social_floor_income_test: float = 0.0  # 소득조사 기준(사적연금 y_ann, 기본 0=조금이라도 있으면 실격)

    # --- Eval / bookkeeping ---
    # seeds는 리스트/튜플 모두 허용 → 내부적으로 튜플로 고정
    seeds: Union[Tuple[int, ...], Sequence[int]] = (0, 1, 2, 3, 4)
    n_paths_eval: int = 500
    tag: str = "dev"

    # --- Hedge options (MVP toggle) ---
    hedge_on: bool = HEDGE_ON_DEFAULT
    hedge_mode: Literal["mu", "sigma", "downside"] = HEDGE_MODE_DEFAULT
    hedge_cost: float = HEDGE_COST_DEFAULT        # = hedge_premium_annual
    hedge_sigma_k: float = HEDGE_SIGMA_K_DEFAULT  # σ 감소 비율(또는 downside 강도)
    hedge_tx: float = 0.0                         # 거래비용(월 환산은 env에서)

    # --- DEV-ONLY knobs (turn off for paper runs) ---
    w_min_dev: float = 0.0            # 개발용 최소 risky 비중(본실험은 0.0 권장)
    dev_cvar_stage: bool = False      # 개발용 stage-wise tail penalty
    dev_cvar_kappa: float = 10.0      # stage penalty intensity(DEV에서만 영향)
    dev_w2_penalty: float = 0.02      # 개발용 w^2 penalty
    dev_split_w_grid: bool = False    # λ에 따라 w-grid 분리(개발 가시화)

    # --- Market (iid / bootstrap CSV) ---
    market_mode: Literal["iid", "bootstrap"] = "iid"
    market_csv: Optional[str] = None
    bootstrap_block: int = 24
    use_real_rf: Literal["on", "off"] = "on"

    # --- Mortality / bequest (옵션) ---
    mortality: Literal["on", "off"] = "off"
    mort_table: Optional[str] = None
    age0: int = 55
    sex: Literal["M", "F"] = "M"
    bequest_kappa: float = 0.0
    bequest_gamma: float = 1.0

    # --- Annuity overlay (옵션: env에서 사용) ---
    ann_on: Literal["on", "off"] = "off"
    ann_alpha: float = 0.0
    ann_L: float = 0.0
    ann_d: int = 0
    ann_index: Literal["real", "nominal"] = "real"
    y_ann: float = 0.0  # 외부 고정지급(있다면)

    # --- XAI / IO / misc ---
    xai_on: Literal["on", "off"] = "on"
    quiet: Literal["on", "off"] = "on"
    bands: Literal["on", "off"] = "on"
    outputs: str = "./outputs"
    data_profile: Optional[Literal["dev", "full"]] = None
    data_window: Optional[str] = None

    # --- Allocation & FX hedge (멀티에셋일 때 사용 가능) ---
    alpha_mix: Optional[str] = None
    alpha_kr: Optional[float] = None
    alpha_us: Optional[float] = None
    alpha_au: Optional[float] = None
    h_FX: Optional[float] = None
    fx_hedge_cost: Optional[float] = None

    # --- Lite overrides (optional) ---
    q_floor: Optional[float] = None
    beta: Optional[float] = None

    # --- Stage-wise CVaR (optional) ---
    cvar_stage: Literal["on", "off"] = "off"
    alpha_stage: float = 0.95
    lambda_stage: float = 0.0
    cstar_mode: Literal["fixed", "annuity", "vpw"] = "annuity"
    cstar_m: float = 0.04 / 12

    def __post_init__(self):
        # seeds: 튜플로 고정
        if not isinstance(self.seeds, tuple):
            try:
                self.seeds = tuple(int(s) for s in self.seeds)  # type: ignore
            except Exception:
                self.seeds = (0,)

        # η-grid 자동 세팅: 미지정이면 [0, F_target] 구간을 hjb_eta_n 점으로 생성
        if not self.hjb_eta_grid or len(self.hjb_eta_grid) <= 3:
            try:
                import numpy as _np
                F = float(self.F_target or 1.0)
                n = int(self.hjb_eta_n or 41)
                self.hjb_eta_grid = tuple(float(x) for x in _np.linspace(0.0, F, n))
            except Exception:
                n = max(int(self.hjb_eta_n or 41), 1)
                F = float(self.F_target or 1.0)
                step = F / (n - 1 if n > 1 else 1)
                self.hjb_eta_grid = tuple(0.0 + i * step for i in range(n))

        # w-grid 자동 세팅: 미지정(None/빈)일 때 0~w_max 균등분할(hjb_w_grid_n 점)
        if not self.hjb_w_grid:
            n = max(int(self.hjb_w_grid_n or 8), 2)
            try:
                import numpy as _np
                self.hjb_w_grid = tuple(float(x) for x in _np.linspace(0.0, float(self.w_max), n))
            except Exception:
                step = float(self.w_max) / (n - 1)
                self.hjb_w_grid = tuple(0.0 + i * step for i in range(n))
        else:
            # 사용자가 리스트/튜플로 준 그리드를 정규화(0~w_max 범위, 오름차순, 중복제거)
            try:
                vals = [float(x) for x in self.hjb_w_grid]  # type: ignore[arg-type]
                vals = [min(max(0.0, x), float(self.w_max)) for x in vals]
                vals = sorted(set(vals))
                if len(vals) < 2:
                    # 안전장치: 최소 2점은 필요
                    vals = [0.0, float(self.w_max)]
                self.hjb_w_grid = tuple(vals)
            except Exception:
                # 실패 시 균등분할로 대체
                n = max(int(self.hjb_w_grid_n or 8), 2)
                step = float(self.w_max) / (n - 1)
                self.hjb_w_grid = tuple(0.0 + i * step for i in range(n))

    def monthly(self) -> Dict[str, float]:
        """월간 단위로 변환(수수료는 ad-valorem 기준)."""
        mu_m = (1.0 + self.mu_annual) ** (1.0 / self.steps_per_year) - 1.0
        sigma_m = self.sigma_annual / (self.steps_per_year ** 0.5)
        rf_m = (1.0 + self.rf_annual) ** (1.0 / self.steps_per_year) - 1.0

        # fee: 최신 필드(fee_annual)를 우선, 없으면 phi_adval 사용
        fee_annual = float(self.fee_annual if self.phi_adval is None else self.phi_adval)
        # ad-valorem을 “월간 *비율*”로 변환 (복리기준 역산)
        phi_m = 1.0 - (1.0 - fee_annual) ** (1.0 / self.steps_per_year)

        p_m = 1.0 - (1.0 - self.p_annual) ** (1.0 / self.steps_per_year)
        g_m = (1.0 + self.g_real_annual) ** (1.0 / self.steps_per_year) - 1.0
        beta_m = self.rl_gamma  # 월간 할인인자(간편 재사용)

        return dict(mu_m=mu_m, sigma_m=sigma_m, rf_m=rf_m,
                    phi_m=phi_m, p_m=p_m, g_m=g_m, beta_m=beta_m)


ASSET_PRESETS = {
    # [2026-07] 논문 <표 1> 값으로 정합화 (출처: KOSPI/S&P500/LBMA, 2000-2024 실질수익률 기준)
    # 본 논문의 기본분석은 한국주식(KR) 단일 위험자산 + 안전자산(3년 국채) 2자산 구조를
    # 채택하며, US/Gold는 부록의 대안적 위험자산 가정 하 강건성 체크 용도로만 사용한다.
    "KR":   dict(mu_annual=0.055, sigma_annual=0.22),  # KOSPI, 2000-2024
    "US":   dict(mu_annual=0.065, sigma_annual=0.18),  # S&P500, 2000-2024
    "Gold": dict(mu_annual=0.02,  sigma_annual=0.15),  # LBMA, 2000-2024
    # [2026-07 신규] TDF(2015&2020, 은퇴임박형 빈티지) 실증 자산배분 비중을 반영한
    # 합성 위험자산(국내주식 14.6% : 해외주식 85.4%). 국내:해외 비중 근거는
    # 최재윤·송인욱·박영규(2024, 경영학연구), 자산군별 베타(민감도) 추정치
    # (국내주식 β=0.0534, 해외주식 β=0.3131). μ,σ는 위 KR/US 장기(2000-2024)
    # 파라미터를 국내:해외 비중으로 가중합성(상관계수 ρ=0.60 가정)한 값이다.
    # (q,w) 구조·HJB·RL 등 모형 구조는 변경 없이, 위험자산의 "질"만 개선한 것이며,
    # 안전자산(2%, 무위험)은 기존과 동일하게 유지한다. 하위자산 간 동적 재조정
    # (전술적 자산배분)은 고려하지 않고 정적 비중으로 고정하였다(생애주기에 따른
    # 위험 조정은 (q,w) 자체의 동적 최적화를 통해 이루어짐).
    "TDF":  dict(mu_annual=0.0635, sigma_annual=0.1749),
}
# 안전자산: 3년 만기 국채수익률 ≈ 연 2.0% (rf_annual 기본값과 일치, config 상단 참조)
