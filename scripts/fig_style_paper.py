# scripts/fig_style_paper.py
# 논문/보고서 도판용 Matplotlib 스타일 프리셋 (refactored: 2025-11-07)
from __future__ import annotations

from typing import List, Optional

def _pick_first_available(candidates: List[str]) -> str:
    """설치된 폰트 중 candidates 순서대로 최초 매칭(정확/부분) 반환. 없으면 'DejaVu Sans'."""
    import matplotlib as mpl
    from matplotlib import font_manager
    try:
        installed = list(font_manager.fontManager.ttflist)
        names = [f.name for f in installed]
        name_set = set(names)

        # 1) 정확 매칭
        for c in candidates:
            if c in name_set:
                return c

        # 2) 부분 매칭(대소문자 무시)
        low = [n.lower() for n in names]
        for c in candidates:
            c_low = c.lower()
            for i, nlow in enumerate(low):
                if c_low in nlow:
                    return names[i]
    except Exception:
        pass
    return "DejaVu Sans"


def _apply_common_pdf_text_settings() -> None:
    """PDF/PS에서 텍스트가 벡터로 유지되도록 설정."""
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42    # Type 42 (TTF) — 텍스트 선택 가능
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["svg.fonttype"] = "none"  # SVG 텍스트를 path로 변환하지 않음


def apply_paper_style(font_priority: Optional[List[str]] = None) -> None:
    """
    논문 도판에 적합한 전역 스타일 적용.
    - 기본 폰트 자동 탐색(영문/한글 환경 모두 고려)
    - 얇은 라인, 좌/하단 스파인만 노출, 눈금·그리드 통일
    - 저장 시: dpi=300, bbox='tight' 등
    - PDF/PS에서 텍스트 벡터 유지(Type 42)
    """
    import matplotlib as mpl

    # 0) 공통(백엔드 독립) PDF/PS 텍스트 설정
    _apply_common_pdf_text_settings()

    # 1) 폰트 우선순위(플랫폼별 대표 폰트를 포함)
    font_priority = font_priority or [
        # 영문 기본
        "Times New Roman", "Times", "Nimbus Roman",
        "Arial", "Helvetica",
        # 한글 (Windows/Apple/Linux 대표)
        "Malgun Gothic",              # Windows
        "Apple SD Gothic Neo",        # macOS
        "Noto Sans CJK KR", "Noto Sans KR",
        "NanumGothic", "Nanum Gothic",
        # 폴백
        "DejaVu Sans",
    ]
    chosen = _pick_first_available(font_priority)
    mpl.rcParams["font.family"] = chosen

    # 2) 수학/기호 가독성
    #  - 'stix'가 가독성·호환성 좋음. 없으면 기본.
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = True   # 유니코드 마이너스

    # 3) 크기/선굵기/그리드/스파인/여백
    mpl.rcParams.update({
        # 텍스트 크기(논문 1단 기준)
        "font.size":         9,
        "axes.labelsize":    9,
        "axes.titlesize":    9,
        "legend.fontsize":   8,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,

        # 축/스파인
        "axes.titleweight":  "semibold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,

        # 선/마커
        "lines.linewidth":   1.0,
        "lines.markersize":  3,

        # 그리드(잔잔하게)
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    ":",
        "grid.linewidth":    0.5,

        # 기본 Figure(1단 폭 ~3.5in)
        "figure.figsize":    (3.5, 2.8),
        "figure.dpi":        100,  # 편집 화면용; 저장은 savefig.dpi 사용

        # 저장 품질/여백
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches":0.02,

        # 범례/프레임
        "legend.frameon":    False,

        # 컬러 사이클(눈에 익은 Tableau 10 계열 비슷하게)
        "axes.prop_cycle":   mpl.cycler(color=[
            "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
            "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"
        ]),
    })


def apply_report_style() -> None:
    """
    리포트/요약용(조금 더 큼, 발표/슬라이드에 적합).
    - 논문보다 글자/라인을 살짝 키움
    """
    import matplotlib as mpl
    _apply_common_pdf_text_settings()
    mpl.rcParams.update({
        "font.size":        10,
        "axes.labelsize":   10,
        "axes.titlesize":   10,
        "legend.fontsize":  9,
        "lines.linewidth":  1.2,
        "figure.figsize":   (8.0, 5.0),  # 슬라이드/보고서 기본
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.03,
        "axes.grid":        True,
        "grid.alpha":       0.35,
        "grid.linestyle":   ":",
        "grid.linewidth":   0.6,
    })
