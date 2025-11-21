# project/annuity/__init__.py

# 기존 한 줄:
# from .annuity_stream import make_annuity_stream  # 있다면

# → 아래처럼 교체
try:
    from .annuity_stream import make_annuity_stream  # 있다면
except Exception:
    # annuity_stream 쪽에 함수가 없거나, 아직 구현 중이라도
    # mortality_gm 등 다른 모듈을 사용하는 데는 지장이 없도록 한다.
    make_annuity_stream = None  # type: ignore
