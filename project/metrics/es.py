import numpy as np

def es95_wealth(wealth_terminal: np.ndarray, alpha: float = 0.95) -> float:
    """
    ES95 (wealth): 하위 (1-alpha) 구간의 최종부 평균(꼬리 평균).
    값이 클수록 꼬리위험이 낮다는 뜻 → "좋음".
    """
    wt = np.asarray(wealth_terminal).reshape(-1)
    q = np.quantile(wt, 1 - alpha)  # 예: alpha=0.95 → 하위 5% 경계
    tail = wt[wt <= q]
    return float(tail.mean()) if tail.size > 0 else float(q)
