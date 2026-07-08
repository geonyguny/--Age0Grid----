# -*- coding: utf-8 -*-
"""
HJB 정책함수(Pi_w, Pi_q) 시각화 스크립트
사용법: 프로젝트 루트(project 폴더의 부모 디렉토리)에서
    python plot_hjb_policy.py
로 실행하면 outputs/hjb_policy_heatmap.png 가 생성됩니다.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 한글 폰트 설정 (Windows는 보통 '맑은 고딕'이 기본 내장) ──
for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if _font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        matplotlib.rcParams["font.family"] = _font
        break
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, ".")
from project.config import SimConfig
from project.hjb import HJBSolver

# ── 실제 CLI 실행과 동일한 파라미터로 맞춤 ──
cfg = SimConfig(asset="KR", horizon_years=35, w_max=0.70, alpha=0.95)
cfg.crra_gamma = 3.0  # getattr(cfg, "crra_gamma", 3.0) 기본값과 동일(명시적으로 지정)
cfg.hjb_W_max = 10.0      # 추가
cfg.hjb_W_grid = 242      # 추가

solver = HJBSolver(cfg)
res = solver.solve(seed=0)

Pi_w = res["Pi_w"]      # (T, |W_grid|)
Pi_q = res["Pi_q"]      # (T, |W_grid|)
W_grid = res["W_grid"]  # (|W_grid|,)

T = Pi_w.shape[0]
age0 = 55
ages = age0 + np.arange(T) / cfg.steps_per_year

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

im0 = axes[0].imshow(
    Pi_w.T, aspect="auto", origin="lower",
    extent=[ages[0], ages[-1], W_grid[0], W_grid[-1]],
    cmap="viridis", vmin=0.0, vmax=float(cfg.w_max),
)
axes[0].set_title("위험자산 비중 정책 Π_w(나이, 자산수준)")
axes[0].set_xlabel("나이(세)")
axes[0].set_ylabel("자산수준 (W / W0)")
cb0 = fig.colorbar(im0, ax=axes[0])
cb0.set_label("위험자산 비중 w")

im1 = axes[1].imshow(
    Pi_q.T * 12 * 100, aspect="auto", origin="lower",  # 월 인출률 -> 연환산 %
    extent=[ages[0], ages[-1], W_grid[0], W_grid[-1]],
    cmap="magma",
)
axes[1].set_title("인출률 정책 Π_q(나이, 자산수준) [연환산 %]")
axes[1].set_xlabel("나이(세)")
axes[1].set_ylabel("자산수준 (W / W0)")
cb1 = fig.colorbar(im1, ax=axes[1])
cb1.set_label("연환산 인출률 (%)")

fig.suptitle(
    f"HJB 이론적 최적정책 (CRRA γ={cfg.crra_gamma}, w_max={cfg.w_max}, "
    f"Gauss-Hermite 구적 기댓값)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

import os
os.makedirs("outputs", exist_ok=True)
out_path = "outputs/hjb_policy_heatmap.png"
fig.savefig(out_path, dpi=150)
print(f"저장 완료: {out_path}")

# ── 참고용 요약 수치도 같이 출력 ──
idx_w1 = int(np.argmin(np.abs(W_grid - 1.0)))
print(f"W≈1.0(초기자산)에서의 위험자산 비중: age55={Pi_w[0, idx_w1]:.2f}, "
      f"age70={Pi_w[int(15*cfg.steps_per_year), idx_w1]:.2f}, "
      f"age89(말기)={Pi_w[-1, idx_w1]:.2f}")
