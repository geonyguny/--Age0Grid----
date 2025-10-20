# scripts/fig_style_paper.py
# 논문 도판용 Matplotlib 스타일 프리셋
from __future__ import annotations

def apply_paper_style(font_priority: list[str] | None = None) -> None:
    """
    논문 도판에 적합한 전역 스타일 적용.
    - 기본 폰트: Times New Roman / Arial / Noto Sans CJK KR / NanumGothic / DejaVu Sans
    - 얇은 라인, 축/범례 글꼴 크기 통일, 흰 배경, 가는 그리드
    - savefig: dpi=300, bbox='tight'
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # 폰트 우선순위
    font_priority = font_priority or [
        "Times New Roman", "Arial", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"
    ]

    # 설치된 폰트 중 사용 가능한 첫 번째를 선택
    try:
        from matplotlib import font_manager
        available = set(f.name for f in font_manager.fontManager.ttflist)
        for f in font_priority:
            if f in available:
                mpl.rcParams["font.family"] = f
                break
        else:
            # 폴백
            mpl.rcParams["font.family"] = "DejaVu Sans"
    except Exception:
        mpl.rcParams["font.family"] = "DejaVu Sans"

    # 전역 스타일
    mpl.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,

        "lines.linewidth": 1.0,
        "lines.markersize": 3,

        "figure.figsize": (3.5, 2.8),   # 1단 폭 기준(~8.5cm)
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

def apply_report_style() -> None:
    """
    리포트/요약용(조금 더 큼, 가독성 우선). 필요시 선택 적용.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "lines.linewidth": 1.2,
    })
