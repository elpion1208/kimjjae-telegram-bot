"""
변형티큐(V2) 전략 (KimJjae V2 Trading Strategy) 설정 및 상수 정의
변형티큐(V2) 완전 명세서 기준의 모든 기본 입력값과 고정 상수를 반영합니다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    # 1. 종목 설정
    reference_symbol: str = "QQQ"
    target_symbol: str = "TQQQ"
    spy_symbol: str = "SPY"
    vix_symbol: str = "^VIX"
    dxy_symbol: str = "DX-Y.NYB"
    nasdaq_vol_symbol: str = "^IXIC"
    usdkrw_symbol: str = "KRW=X"

    # FRED 금리 시리즈 ID
    fred_dgs6m: str = "DGS6MO"
    fred_dgs2: str = "DGS2"
    fred_dgs10: str = "DGS10"
    fred_dgs20: str = "DGS20"

    # 2. 가격 보정 및 시차(Lag) 설정
    # V2 명세서: TQQQ/QQQ/SPY는 배당 재투자 보정 수정주가(auto_adjust=True) 사용
    use_adjusted_price: bool = True
    adjusted_mode: str = "dividends"  # "dividends" or "splits"
    use_tv_default_rsi: bool = False  # V2 명세서: QQQ 수정종가(qc) 기준 Wilder RSI(14) 산출
    macro_lag_days: int = 1           # V2 명세서: 국채금리·VIX·DXY 전일 확정치(shift(1), 최대 7일 ffill) 사용

    # 3. [G1] 기본 진입비중: TQQQ 200일 이격도 + QQQ SMA3/SMA161 (V2 정수화: 102.0% / 100.0%)
    dist200_enter_band: float = 102.0   # 200일 이격도 상향 확인(진입, %)
    dist200_exit_band: float = 100.0    # 200일 이격도 하향 확인(청산, %)
    early_full_dist: float = 101.00     # 조기 100% 복귀 기준(TQQQ 200일 이격도, %)
    q_bull_weight_floor_dist: float = 97.00  # QQQ 보조비중 확대 기준(TQQQ 200일 이격도, %)
    q_bull_weight_hi: float = 0.20      # QQQ 보조비중 확대값(0.20=20%, 그 미만은 10%)

    # 4. [G2] 45일 이격도 선형회귀 기울기 부스트 & 20일 변동성 락
    use_slope_boost: bool = True
    slope_len: int = 45                 # 이격도 선형회귀 계산 기간
    slope_thr: float = 0.1100           # 이격도 기울기 임계값
    dist_cap: float = 100.0             # 기울기 적용 최대 이격도(200일 기준, %)
    vol_cap: float = 0.06               # 기울기 적용 최대 변동성(20일 변동성)

    # 5. 변동성 락
    vol_threshold: float = 0.059        # 20일 변동성 락 임계값 (수치오차 1e-10 허용)

    # 6. [G3] 과열 분할매도 (1~4단) 및 4단 해제 가드 (V2: O4B / O4P 예외 없음, 4단은 무조건 0%)
    use_overheat_split: bool = True
    overheat1_enter: float = 140.0
    overheat1_exit: float = 132.0
    overheat2_enter: float = 144.0
    overheat2_exit: float = 138.0
    overheat3_enter: float = 147.0
    overheat3_exit: float = 140.0
    overheat4_enter: float = 158.0
    overheat4_exit: float = 118.0

    # 과열 1·2단 상한 해제용 TQQQ ER20 기준 (고정 상수)
    tqqq_er20_len: int = 20
    tqqq_er20_thr: float = 0.30         # > +0.30 일 때만 90%/80% 제한 해제

    # V2에서는 과열 4단 예외 재진입(O4B/O4P)을 사용하지 않음 (호환성 유지용 상수)
    use_o4b_o4p: bool = False
    o4_booster_pct: float = 0.0
    o4_booster_min_dist: float = 157.0
    o4_booster_max_dist: float = 180.0
    o4_booster_q_ret1_thr: float = 0.0025
    oh4_pre_release_buy_level: float = 129.0
    oh4_pre_release_buy_pct: float = 0.0
    qqq_er59_len: int = 59
    qqq_er59_thr: float = 0.05

    # 과열 4단 해제 후 재진입 가드
    oh4_guard_cap_pct: float = 0.95     # 4단 해제 후 상한 95%
    oh4_release_slope_days: int = 4     # 이격도 기울기 >= 0 연속 4회 시 해제

    # 7. [G4] 원금 보호 손절(-7.0%) 및 RSI(14) 재진입 제한
    use_stable_stops: bool = True
    principal_stop_pct: float = 0.93    # V2: 75% 이상 진입 체결 시가 대비 93.00%(-7.00%) 이하 시 손절
    use_next_open_stop_anchor: bool = True  # 진입 신호 익일 시가(Open)를 손절 기준가로 사용
    rsi_length: int = 14                # QQQ Wilder RSI(14)
    rsi_warmup_bars: int = 300
    rsi_reentry_thr: float = 43.0       # QQQ RSI >= 43.0 시 잠금 우회 허용

    # 8. [G5] SPY 200일선 필터
    use_spy_filter: bool = True
    spy_enter_pct: float = 96.50        # SPY 상승 추세 전환 기준(%)
    spy_exit_pct: float = 95.70         # SPY 하락 추세 전환 기준(%)
    spy_confirm_days: int = 1           # 전환 확인 일수
    spy_bear_cap_pct: float = 0.0       # 하락 추세 시 최대 비중(0%)

    # 9. VIX 필터 및 매수보조
    use_vix_filter: bool = True
    use_vix_force_sell: bool = False    # VIX 강제현금 (기본 비활성)
    vix_sell_threshold: float = 46.0    # VIX 위험 기준
    vix_rebuy_threshold: float = 31.0   # VIX 재매수 허용 기준
    use_vix_buy_support: bool = True    # VIX 매수보조 (기본 활성)
    vix_buy_support_days: int = 5       # VIX <= 31 연속 확인일수 (5일)

    # 10. [G6] T3 거시 금리·달러 발작 필터 (QQQ 단기약세 + DXY + 미 국채 4종)
    use_t3_d80_filter: bool = True
    use_q06: bool = True                # QQQ ER10 <= -0.300 약세 조건 포함
    qqq_er10_len: int = 10
    qqq_er10_thr: float = -0.300
    enable_t3_cap: bool = False         # 독립적인 80% 상한 (기본 비활성)
    t3_q_weak_ma_len: int = 3           # QQQ SMA3
    t3_q_weak_cap_pct: float = 0.80
    t3_use_risk_zero: bool = True       # 약세 AND (SR OR (DXY AND LR)) 시 0%
    use_t3_a2y80: bool = True           # SR 위험: DGS2 + DGS06M
    use_t3_dgs20_risk: bool = True      # LR 위험: DGS20 보조 사용

    t3_dxy_roc_len: int = 120           # DXY 120관측 변화율 기간
    t3_dxy_roc_thr: float = 0.0         # > 0.0%
    t3_dgs6m_chg_len: int = 90          # DGS06M 90관측 변화폭 기간
    t3_dgs6m_chg_thr: float = 0.15      # > +0.15%p
    t3_dgs2_chg_len: int = 90           # DGS02Y 90관측 변화폭 기간
    t3_dgs2_chg_thr: float = 0.45       # > +0.45%p
    t3_us10y_chg_len: int = 60          # DGS10Y 60관측 변화폭 기간
    t3_us10y_chg_thr: float = 0.40      # > +0.40%p
    t3_dgs20_chg_len: int = 100         # DGS20Y 100관측 변화폭 기간
    t3_dgs20_chg_thr: float = 0.15      # > +0.15%p

    # 11. 포지션 표시 방식 (False: V2 원신호 비중만 깔끔하게 표시)
    d7_enabled: bool = False
    d7_same_session: bool = True

    # 12. 경계 근접 재확인 문구 설정
    use_oh_audit_fail_note: bool = True
    oh_audit_border_tol_pp: float = 0.05  # 경계 근접 허용폭 (%p / 지표단위)


DEFAULT_CONFIG = StrategyConfig()
