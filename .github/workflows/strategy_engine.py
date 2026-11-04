"""
김째매매법 판정 엔진 (Strategy Engine)
원본 Pine Script v6 (C097)의 1~17절 규칙과 1차(PRIMARY)·2차(FINAL) 상태 전이 로직을 수행합니다.
"""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from config import StrategyConfig, DEFAULT_CONFIG
from data_fetcher import MarketDataBundle
import kernel as K


@dataclass
class DayState:
    """
    Pine Script C084DayState + D7 이력 상태.
    당일 판정 전 상태(c084DayStart)를 보존하여 같은 날 2차(FINAL) 재판정 시
    동일한 당일 시작 상태에서 최신 확정자료로 다시 판정할 수 있게 합니다.
    """
    last_processed_date: Optional[str] = None
    asset_code: int = 0
    above200_hysteresis: bool = False
    above200_hysteresis_inited: bool = False
    overheat_stage: int = 0
    full_entry_close: Optional[float] = None
    full_high_close: Optional[float] = None
    last_principal_stop_price: Optional[float] = None
    oh4_reentry_guard: bool = False
    oh4_slope_ok_cnt: int = 0
    oh4_pre_release_active: bool = False
    o4_booster_active: bool = False
    pos_entry_avg_close: Optional[float] = None
    reentry_lock: bool = False
    spy_bull: bool = False
    spy_inited: bool = False
    spy_confirm_cnt: int = 0
    vix_risk_lock: bool = False
    vix_below_cnt: int = 0

    # D7 (단기채 / 현금 분리) 최근 8거래일 TQQQ 비중 기록 및 현재 단기채 비중
    d7_keys: List[str] = field(default_factory=list)
    d7_weights: List[float] = field(default_factory=list)
    d7_bill_state: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DayState":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class DecisionSnapshot:
    """1차(PRIMARY) 또는 2차(FINAL) 판정 결과 스냅샷"""
    trade_date: str
    phase: str  # "PRIMARY" or "FINAL"

    # 포지션 (이전 vs 목표)
    prev_asset_code: int
    prev_o4_role: int
    target_asset_code: int
    target_o4_role: int
    before_t3_code: int
    before_t3_o4_role: int

    # D7 비중 (TQ / 단기채 / 현금)
    before_tq: float
    before_bill: float
    before_cash: float
    target_tq: float
    target_bill: float
    target_cash: float
    d7_cap: float
    d7_history_complete: bool

    # 판정 사유 및 태그
    reason_text: str
    tag_text: str
    stop_event_flag: bool
    locked: bool
    vix_force_cash: bool
    vix_buy_support_blocked: bool
    spy_bear_now: bool
    reentry_blocked_now: bool
    t3_risk_off_zero_flag: bool
    t3_q_weak_cap_applied_flag: bool
    oh4_guard_applied_flag: bool
    cap_binding: bool

    # TQQQ 지표
    t_c: float
    t_p: float
    t_ret_pct: float
    t_ma200: float
    dist200: float
    dist_slope: Optional[float]
    t_v20: float
    t_er20: Optional[float]
    above200_hysteresis: bool
    slope_ok_flag: bool
    overheat_stage: int
    prev_overheat_stage: int
    oh4_reentry_guard: bool
    oh4_slope_ok_cnt: int
    o4_booster_ok: bool
    o4_partial_ok: bool
    principal_stop_price: Optional[float]
    pos_entry_avg_close: Optional[float]
    reentry_lock: bool

    # QQQ 지표
    q_c: float
    q_p: float
    q_ret1_pct: float
    q_ret3_pct: Optional[float]
    q_ma3: float
    q_ma161: float
    q_bull: bool
    qqq_rsi14: Optional[float]
    qqq_er10: Optional[float]
    qqq_er59: Optional[float]
    qqq3_emoji: str
    next_qqq_buy_price: Optional[float]
    next_qqq_sell_price: Optional[float]
    other_filters_permit_hold: bool

    # SPY & VIX 지표
    spy_c: float
    spy_ma200: float
    spy_dist200: float
    spy_bull: bool
    vix_c: Optional[float]
    vix_p: Optional[float]
    vix_below_cnt: int
    vix_risk_lock: bool

    # T3 거시위험 지표 (DXY & FRED DGS 4종)
    t3_q_weak: bool
    t3_m09_weak: bool
    dxy_c: Optional[float]
    dxy_ref: Optional[float]
    dxy_roc: Optional[float]
    dxy_risk: bool
    dxy_source_label: str
    dgs6m_c: Optional[float]
    dgs6m_ref: Optional[float]
    dgs6m_chg: Optional[float]
    dgs6m_risk: bool
    dgs2_c: Optional[float]
    dgs2_ref: Optional[float]
    dgs2_chg: Optional[float]
    dgs2_risk: bool
    us10y_c: Optional[float]
    us10y_ref: Optional[float]
    us10y_chg: Optional[float]
    us10y_risk: bool
    dgs20_c: Optional[float]
    dgs20_ref: Optional[float]
    dgs20_chg: Optional[float]
    dgs20_risk: bool
    t3_short_risk: bool
    t3_long_risk: bool
    dgs_obs_date: Optional[str]

    # 환율 및 경계 근접 감사(Audit) 경고
    usdkrw_c: Optional[float]
    usdkrw_p: Optional[float]
    border_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _lookup_obs_and_ref(
    series: pd.Series,
    trade_date: str,
    chg_len: int,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    주석 1 · 12 · 14절:
    해당 거래일(trade_date) 이하의 가장 최신 유효 관측치(c)와
    그로부터 chg_len 관측 이전의 값(ref), 그리고 관측일(obs_date)을 반환합니다.
    """
    sub = series[series.index <= trade_date].dropna()
    if len(sub) <= chg_len:
        return None, None, None
    c_val = float(sub.iloc[-1])
    ref_val = float(sub.iloc[-1 - chg_len])
    obs_date = str(sub.index[-1])
    return c_val, ref_val, obs_date


def _check_border_warnings(
    cfg: StrategyConfig,
    dist200: float,
    dist_slope: Optional[float],
    t_v20: float,
    t_er20: Optional[float],
    qqq_er10: Optional[float],
    qqq_er59: Optional[float],
    qqq_rsi14: Optional[float],
    spy_dist200: Optional[float],
    vix_c: Optional[float],
    dxy_roc: Optional[float],
    dgs6m_chg: Optional[float],
    dgs2_chg: Optional[float],
    us10y_chg: Optional[float],
    dgs20_chg: Optional[float],
) -> List[str]:
    """
    Pine Script useOhAuditFailNote / ohAuditBorderTolPp 구현:
    주요 임계값(경계)에 허용폭(기본 0.05) 이내로 근접한 지표가 있으면
    데이터 공급자 간 미세 오차 가능성을 사용자에게 알려줍니다.
    """
    if not cfg.use_oh_audit_fail_note:
        return []
    tol = cfg.oh_audit_border_tol_pp
    warnings: List[str] = []

    def _near(val: Optional[float], thr: float, custom_tol: Optional[float] = None) -> bool:
        if val is None or math.isnan(val):
            return False
        t = tol if custom_tol is None else custom_tol
        return abs(val - thr) <= t

    # TQQQ 200일 이격도 주요 경계
    dist_thresholds = [
        (cfg.dist200_enter_band, "200일 진입(102.27%)"),
        (cfg.dist200_exit_band, "200일 청산(100.28%)"),
        (cfg.early_full_dist, "조기100%(101.00%)"),
        (cfg.q_bull_weight_floor_dist, "보조비중(97.00%)"),
        (cfg.overheat1_enter, "과열1단 진입(140%)"),
        (cfg.overheat1_exit, "과열1단 해제(132%)"),
        (cfg.overheat2_enter, "과열2단 진입(144%)"),
        (cfg.overheat2_exit, "과열2단 해제(138%)"),
        (cfg.overheat3_enter, "과열3단 진입(147%)"),
        (cfg.overheat3_exit, "과열3단 해제(140%)"),
        (cfg.overheat4_enter, "과열4단 진입(158%)"),
        (cfg.overheat4_exit, "과열4단 해제(118%)"),
        (cfg.oh4_pre_release_buy_level, "O4P 이격도(129%)"),
    ]
    for thr, label in dist_thresholds:
        if _near(dist200, thr, tol):
            warnings.append(f"TQQQ 이격도({dist200:.2f}%)가 [{label}] 경계 근접")

    if _near(t_v20, cfg.vol_threshold, 0.0005):
        warnings.append(f"20일 변동성({t_v20:.4f})이 락 경계({cfg.vol_threshold:.3f}) 근접")
    if _near(dist_slope, cfg.slope_thr, 0.005):
        warnings.append(f"이격도 기울기({dist_slope:.4f})가 부스트 경계({cfg.slope_thr:.4f}) 근접")
    if _near(spy_dist200, cfg.spy_enter_pct, tol) or _near(spy_dist200, cfg.spy_exit_pct, tol):
        warnings.append(f"SPY 이격도({spy_dist200:.2f}%)가 전환 경계(95.70%/96.50%) 근접")
    if _near(vix_c, cfg.vix_rebuy_threshold, tol) or _near(vix_c, cfg.vix_sell_threshold, tol):
        warnings.append(f"VIX({vix_c:.2f})가 기준선(31.0/46.0) 근접")
    if _near(qqq_er10, cfg.qqq_er10_thr, 0.01):
        warnings.append(f"QQQ ER10({qqq_er10:.3f})이 약세 경계(-0.300) 근접")
    if _near(t_er20, cfg.tqqq_er20_thr, 0.01):
        warnings.append(f"TQQQ ER20({t_er20:.3f})이 과열해제 경계(+0.300) 근접")
    if _near(qqq_er59, cfg.qqq_er59_thr, 0.01):
        warnings.append(f"QQQ ER59({qqq_er59:.3f})이 O4P 경계(+0.050) 근접")
    if _near(qqq_rsi14, cfg.rsi_reentry_thr, 0.3):
        warnings.append(f"QQQ RSI({qqq_rsi14:.2f})가 재진입 경계(43.0) 근접")
    if _near(dxy_roc, cfg.t3_dxy_roc_thr, tol):
        warnings.append(f"DXY 120일 변화율({dxy_roc:+.2f}%)이 0.00% 경계 근접")
    if _near(dgs2_chg, cfg.t3_dgs2_chg_thr, 0.02):
        warnings.append(f"DGS2 변화폭({dgs2_chg:+.2f}%p)이 +0.45%p 경계 근접")
    if _near(dgs6m_chg, cfg.t3_dgs6m_chg_thr, 0.02):
        warnings.append(f"DGS06M 변화폭({dgs6m_chg:+.2f}%p)이 +0.15%p 경계 근접")
    if _near(us10y_chg, cfg.t3_us10y_chg_thr, 0.02):
        warnings.append(f"DGS10Y 변화폭({us10y_chg:+.2f}%p)이 +0.40%p 경계 근접")
    if _near(dgs20_chg, cfg.t3_dgs20_chg_thr, 0.02):
        warnings.append(f"DGS20Y 변화폭({dgs20_chg:+.2f}%p)이 +0.15%p 경계 근접")

    return warnings


class StrategyEngine:
    """
    김째매매법 일자별 상태 재생 및 당일 1차(PRIMARY)/2차(FINAL) 판정기
    """

    def __init__(self, bundle: MarketDataBundle, cfg: StrategyConfig = DEFAULT_CONFIG):
        self.bundle = bundle
        self.cfg = cfg
        self._prepare_indicators()

    def _prepare_indicators(self) -> None:
        """전체 시계열에 대해 이동평균, 변동성, 기울기, RSI, ER(10/20/59)을 계산합니다."""
        b = self.bundle
        c = self.cfg

        # TQQQ 파생 지표
        self.t_ma200 = K.calc_sma_series(b.tqqq_close, 200)
        self.t_dist200 = (b.tqqq_close / self.t_ma200) * 100.0
        self.t_ret1 = (b.tqqq_close / b.tqqq_close.shift(1)) - 1.0
        self.t_v20 = K.calc_sample_stdev_series(self.t_ret1, 20)
        self.t_dist_slope = K.calc_linreg_slope_series(self.t_dist200, c.slope_len)
        self.t_er20, self.t_er20_tail, self.t_er20_abs_tail = K.calc_er_series(
            b.tqqq_close, c.tqqq_er20_len
        )

        # QQQ 파생 지표
        self.q_ma3 = K.calc_sma_series(b.qqq_close, 3)
        self.q_ma161 = K.calc_sma_series(b.qqq_close, 161)
        self.q_t3ma = K.calc_sma_series(b.qqq_close, c.t3_q_weak_ma_len)
        self.q_next_ma = (b.qqq_close + b.qqq_close.shift(1)) / 2.0  # 익일 SMA3 역산용
        self.q_ret1 = (b.qqq_close / b.qqq_close.shift(1)) - 1.0
        self.q_ret3 = (b.qqq_close / b.qqq_close.shift(2)) - 1.0  # 주석 6절: 종가 / 2거래일 전 종가 - 1
        self.q_rsi14 = K.calc_rsi_wilder_series(b.qqq_rsi_close, c.rsi_length)
        self.q_er10, self.q_er10_q9, self.q_er10_abs9 = K.calc_er_series(
            b.qqq_close, c.qqq_er10_len
        )
        self.q_er59, _, _ = K.calc_er_series(b.qqq_close, c.qqq_er59_len)

        # SPY 파생 지표
        self.s_ma200 = K.calc_sma_series(b.spy_close, 200)
        self.s_dist200 = (b.spy_close / self.s_ma200) * 100.0

        # 유효 거래일 목록 (200일 이동평균이 준비된 시점부터)
        valid_mask = (~self.t_dist200.isna()) & (~self.q_ma161.isna()) & (~self.s_dist200.isna())
        self.valid_dates: List[str] = list(self.t_dist200[valid_mask].index)

    def step_one_day(
        self,
        base_state: DayState,
        trade_date: str,
        phase: str = "PRIMARY",
        is_latest_day: bool = False,
    ) -> Tuple[DayState, DecisionSnapshot]:
        """
        단일 거래일(trade_date)에 대해 base_state로부터 다음 상태와 판정 스냅샷을 계산합니다.
        Pine Script의 2400~2872행 로직을 그대로 따릅니다.
        """
        cfg = self.cfg
        b = self.bundle
        state = copy.deepcopy(base_state)

        prev_asset = state.asset_code
        prev_o4_role = (
            K.o4_role_from_flags(state.o4_booster_active, state.oh4_pre_release_active)
            if K.code_weight(prev_asset) > 1e-9
            else 0
        )

        # 1. 해당 거래일의 주식 원천 값 추출
        pd_tc = float(b.tqqq_close.loc[trade_date])
        t_idx = b.tqqq_close.index.get_loc(trade_date)
        pd_tp = float(b.tqqq_close.iloc[t_idx - 1]) if t_idx > 0 else pd_tc
        pd_t_ma200 = float(self.t_ma200.loc[trade_date])
        dist200 = float(self.t_dist200.loc[trade_date])
        pd_tv20 = float(self.t_v20.loc[trade_date])
        dist_slope_val = self.t_dist_slope.loc[trade_date]
        dist_slope = None if pd.isna(dist_slope_val) else float(dist_slope_val)

        pd_q0 = float(b.qqq_close.loc[trade_date])
        q_idx = b.qqq_close.index.get_loc(trade_date)
        pd_q1 = float(b.qqq_close.iloc[q_idx - 1]) if q_idx >= 1 else pd_q0
        pd_q2 = float(b.qqq_close.iloc[q_idx - 2]) if q_idx >= 2 else pd_q1
        pd_q3 = float(b.qqq_close.iloc[q_idx - 3]) if q_idx >= 3 else pd_q2
        pd_q_ma3 = float(self.q_ma3.loc[trade_date])
        pd_q_ma161 = float(self.q_ma161.loc[trade_date])
        pd_q_t3ma = float(self.q_t3ma.loc[trade_date])
        pd_q_ret1 = float(self.q_ret1.loc[trade_date])
        pd_q_ret3_val = self.q_ret3.loc[trade_date]
        pd_q_ret3 = None if pd.isna(pd_q_ret3_val) else float(pd_q_ret3_val)

        rsi_val = self.q_rsi14.get(trade_date, np.nan)
        pd_qqq_rsi14 = None if pd.isna(rsi_val) else float(rsi_val)
        pd_qqq_rsi_ready = pd_qqq_rsi14 is not None

        pd_spy_c = float(b.spy_close.loc[trade_date])
        pd_spy_ma200 = float(self.s_ma200.loc[trade_date])
        spy_dist200 = float(self.s_dist200.loc[trade_date])

        # ER(10, 20, 59) 추출
        er20_val = self.t_er20.get(trade_date, np.nan)
        a2_er20 = None if pd.isna(er20_val) else float(er20_val)
        a2_er20_ready = a2_er20 is not None

        er59_val = self.q_er59.get(trade_date, np.nan)
        a2_er59 = None if pd.isna(er59_val) else float(er59_val)
        a2_er59_ready = a2_er59 is not None

        er10_val = self.q_er10.get(trade_date, np.nan)
        pd_q06_er = None if pd.isna(er10_val) else float(er10_val)
        pd_q06_ready = pd_q06_er is not None

        # 나스닥 거래량 유효성 확인
        ixic_vol = b.ixic_volume.get(trade_date, np.nan)
        nas_vol_ready = (not pd.isna(ixic_vol)) and (float(ixic_vol) > 0)

        # 2. VIX 추출 (1차 PRIMARY일 때는 16:00 ET 5분봉 스냅샷 우선, 2차 FINAL일 때는 확정 일봉 사용)
        vix_daily_sub = b.vix_close[b.vix_close.index <= trade_date]
        vix_p = float(vix_daily_sub.iloc[-2]) if len(vix_daily_sub) >= 2 else None
        if phase == "PRIMARY" and is_latest_day and trade_date in b.vix_primary_close_by_date:
            pd_vc = float(b.vix_primary_close_by_date[trade_date])
        elif trade_date in b.vix_close.index:
            pd_vc = float(b.vix_close.loc[trade_date])
        elif not vix_daily_sub.empty:
            pd_vc = float(vix_daily_sub.iloc[-1])
        else:
            pd_vc = None

        # 3. DXY 추출 (주석 12·14절: 당일 확정 B가 있으면 B, 아니면 주식마감 A 스냅샷 사용)
        dxy_sub = b.dxy_daily_close[b.dxy_daily_close.index <= trade_date]
        dxy_b_ready = (
            trade_date in b.dxy_daily_close.index
            and len(dxy_sub) > cfg.t3_dxy_roc_len
        )
        dxy_a_ready = (
            is_latest_day
            and b.dxy_cohort_a_date == trade_date
            and b.dxy_cohort_a_current is not None
            and b.dxy_cohort_a_ref is not None
        )
        if phase == "PRIMARY" and is_latest_day and dxy_a_ready and not dxy_b_ready:
            pd_dxy_c = float(b.dxy_cohort_a_current)
            pd_dxy_ref = float(b.dxy_cohort_a_ref)
            dxy_source_label = "A(장마감 16시)"
        elif len(dxy_sub) > cfg.t3_dxy_roc_len:
            pd_dxy_c = float(dxy_sub.iloc[-1])
            pd_dxy_ref = float(dxy_sub.iloc[-1 - cfg.t3_dxy_roc_len])
            dxy_source_label = "B(일봉 확정)" if trade_date in b.dxy_daily_close.index else "B(직전 일봉)"
        elif dxy_a_ready:
            pd_dxy_c = float(b.dxy_cohort_a_current)
            pd_dxy_ref = float(b.dxy_cohort_a_ref)
            dxy_source_label = "A(장마감 16시)"
        else:
            pd_dxy_c, pd_dxy_ref = None, None
            dxy_source_label = "미확인"

        dxy_ok = (
            pd_dxy_c is not None
            and pd_dxy_ref is not None
            and pd_dxy_c > 0
            and pd_dxy_ref > 0
        )
        dxy_roc = ((pd_dxy_c / pd_dxy_ref) - 1.0) * 100.0 if dxy_ok else None

        # 4. FRED 국채금리 4종 추출
        dgs6m_c, dgs6m_ref, obs_6m = _lookup_obs_and_ref(b.dgs6m, trade_date, cfg.t3_dgs6m_chg_len)
        dgs2_c, dgs2_ref, obs_2y = _lookup_obs_and_ref(b.dgs2, trade_date, cfg.t3_dgs2_chg_len)
        us10y_c, us10y_ref, obs_10y = _lookup_obs_and_ref(b.dgs10, trade_date, cfg.t3_us10y_chg_len)
        dgs20_c, dgs20_ref, obs_20y = _lookup_obs_and_ref(b.dgs20, trade_date, cfg.t3_dgs20_chg_len)

        dgs6m_chg = (dgs6m_c - dgs6m_ref) if (dgs6m_c is not None and dgs6m_ref is not None) else None
        dgs2_chg = (dgs2_c - dgs2_ref) if (dgs2_c is not None and dgs2_ref is not None) else None
        us10y_chg = (us10y_c - us10y_ref) if (us10y_c is not None and us10y_ref is not None) else None
        dgs20_chg = (dgs20_c - dgs20_ref) if (dgs20_c is not None and dgs20_ref is not None) else None
        dgs_obs_date = obs_2y or obs_10y or obs_6m or obs_20y

        # 5. T3 약세 및 거시위험 판정 (주석 11·12절)
        t3_q_weak = cfg.use_t3_d80_filter and (pd_q0 < pd_q_t3ma)
        t3_m09_weak = cfg.use_t3_d80_filter and (
            t3_q_weak or (cfg.use_q06 and pd_q06_ready and pd_q06_er <= cfg.qqq_er10_thr)
        )
        t3_dxy_risk = cfg.use_t3_d80_filter and (dxy_roc is not None) and (dxy_roc > cfg.t3_dxy_roc_thr)
        t3_dgs6m_risk = (
            cfg.use_t3_d80_filter
            and cfg.use_t3_a2y80
            and (dgs6m_chg is not None)
            and (dgs6m_chg > cfg.t3_dgs6m_chg_thr)
        )
        t3_dgs2_risk = (
            cfg.use_t3_d80_filter
            and cfg.use_t3_a2y80
            and (dgs2_chg is not None)
            and (dgs2_chg > cfg.t3_dgs2_chg_thr)
        )
        t3_us10y_risk = (
            cfg.use_t3_d80_filter
            and (us10y_chg is not None)
            and (us10y_chg > cfg.t3_us10y_chg_thr)
        )
        t3_dgs20_risk = (
            cfg.use_t3_d80_filter
            and cfg.use_t3_dgs20_risk
            and (dgs20_chg is not None)
            and (dgs20_chg > cfg.t3_dgs20_chg_thr)
        )

        t3_short_rate_risk = (
            cfg.use_t3_d80_filter
            and cfg.t3_use_risk_zero
            and cfg.use_t3_a2y80
            and t3_m09_weak
            and t3_dgs2_risk
            and t3_dgs6m_risk
        )
        t3_long_rate_risk = (
            cfg.use_t3_d80_filter
            and cfg.t3_use_risk_zero
            and t3_m09_weak
            and t3_dxy_risk
            and (t3_us10y_risk or t3_dgs20_risk)
        )
        t3_risk_off_zero_state = t3_short_rate_risk or t3_long_rate_risk

        # 6. 선행 차단·기본비중·과열 단계 판정 (주석 2~10절 & 13절 순서)
        s1_enter = cfg.overheat1_enter
        s1_exit = min(cfg.overheat1_exit, s1_enter - 0.01)
        s2_enter = max(cfg.overheat2_enter, s1_enter + 0.01)
        s2_exit = min(cfg.overheat2_exit, s2_enter - 0.01)
        s3_enter = max(cfg.overheat3_enter, s2_enter + 0.01)
        s3_exit = min(cfg.overheat3_exit, s3_enter - 0.01)
        s4_enter = max(cfg.overheat4_enter, s3_enter + 0.01)
        s4_exit = min(cfg.overheat4_exit, s4_enter - 0.01)
        oh4_pre_buy_level = min(max(cfg.oh4_pre_release_buy_level, s4_exit + 0.01), s4_enter - 0.01)

        pre_vix_risk_lock, pre_vix_below_cnt = K.next_vix_state(
            cfg.use_vix_filter,
            state.vix_risk_lock,
            state.vix_below_cnt,
            pd_vc,
            cfg.vix_sell_threshold,
            cfg.vix_rebuy_threshold,
            carry_only=(pd_vc is None),
        )
        pre_vix_force_cash = cfg.use_vix_filter and cfg.use_vix_force_sell and pre_vix_risk_lock
        pre_locked = (pd_tv20 + 1e-10) >= max(cfg.vol_threshold, 1e-6)

        pre_spy_bull, pre_spy_inited, pre_spy_confirm_cnt = K.next_spy_state(
            cfg.use_spy_filter,
            state.spy_bull,
            state.spy_inited,
            state.spy_confirm_cnt,
            spy_dist200,
            max(cfg.spy_enter_pct, cfg.spy_exit_pct + 0.01),
            cfg.spy_exit_pct,
            max(cfg.spy_confirm_days, 1),
        )
        pre_spy_bear = cfg.use_spy_filter and pre_spy_inited and (not pre_spy_bull)
        pre_spy_force_cash = pre_spy_bear and (cfg.spy_bear_cap_pct <= 1e-9)

        pre_rsi_reentry_ok = pd_qqq_rsi_ready and (pd_qqq_rsi14 >= cfg.rsi_reentry_thr)
        pre_reentry_blocked = state.reentry_lock and (not pre_rsi_reentry_ok)

        pre_above200, pre_above200_inited = K.next_above200_state(
            state.above200_hysteresis,
            state.above200_hysteresis_inited,
            dist200,
            max(cfg.dist200_enter_band, cfg.dist200_exit_band + 0.01),
            cfg.dist200_exit_band,
        )
        pre_q_bull = pd_q_ma3 > pd_q_ma161
        pre_early_full = (not pre_above200) and pre_q_bull and (dist200 >= cfg.early_full_dist)
        pre_q_bull_weight = cfg.q_bull_weight_hi if dist200 >= cfg.q_bull_weight_floor_dist else 0.10
        pre_q_bull_code = K.weight_to_code(pre_q_bull_weight)
        pre_base0 = 2 if (pre_above200 or pre_early_full) else (pre_q_bull_code if pre_q_bull else 0)
        pre_base0_w = K.code_weight(pre_base0)

        pre_slope_applicable = (
            cfg.use_slope_boost
            and pre_q_bull
            and (pre_base0_w > 1e-9)
            and (pre_base0_w < 0.80 - 1e-9)
            and (dist_slope is not None)
            and (dist200 <= cfg.dist_cap)
            and (pd_tv20 <= cfg.vol_cap)
        )
        pre_slope_ok = pre_slope_applicable and (dist_slope >= cfg.slope_thr)
        pre_hard_cash = pre_locked or pre_spy_force_cash or pre_reentry_blocked or pre_vix_force_cash

        ungated_base_code = 2 if pre_slope_ok else pre_base0
        pre_base_code = 0 if pre_hard_cash else ungated_base_code

        prev_overheat_stage = state.overheat_stage
        ungated_overheat_stage = K.next_overheat_stage(
            prev_overheat_stage,
            cfg.use_overheat_split,
            dist200,
            ungated_base_code,
            False,
            s1_enter,
            s1_exit,
            s2_enter,
            s2_exit,
            s3_enter,
            s3_exit,
            s4_enter,
            s4_exit,
        )
        pre_overheat_stage = (
            ungated_overheat_stage
            if prev_overheat_stage == 4
            else (0 if pre_hard_cash else ungated_overheat_stage)
        )

        # 과열 4단 해제 후 재진입 가드
        pre_oh4_transition = cfg.use_overheat_split and (prev_overheat_stage == 4) and (pre_overheat_stage != 4)
        pre_oh4_reentry_guard = True if pre_oh4_transition else state.oh4_reentry_guard
        pre_oh4_slope_ok_cnt = 0 if pre_oh4_transition else state.oh4_slope_ok_cnt
        if pre_oh4_reentry_guard:
            pre_oh4_slope_ok_cnt = (
                pre_oh4_slope_ok_cnt + 1
                if (dist_slope is not None and dist_slope >= 0)
                else 0
            )
            if pre_oh4_slope_ok_cnt >= max(cfg.oh4_release_slope_days, 1):
                pre_oh4_reentry_guard = False
                pre_oh4_slope_ok_cnt = 0

        # 당일 원금 보호 손절 체크 (주석 9절)
        pre_stop_hit = (
            cfg.use_stable_stops
            and K.is_high_exposure(prev_asset)
            and (state.full_entry_close is not None)
            and (pd_tc <= state.full_entry_close * cfg.principal_stop_pct)
        )

        # ER20 및 O4B / O4P 판정 (주석 5·6·7절)
        a2_oh12_strong = a2_er20_ready and (a2_er20 > cfg.tqqq_er20_thr)
        a2_o4p_allow = a2_er59_ready and (a2_er59 >= cfg.qqq_er59_thr)
        o4_partial_vol_ok = nas_vol_ready and a2_o4p_allow
        o4_booster_cond_ok = (
            cfg.use_overheat_split
            and (pd_q_ret3 is not None)
            and (cfg.o4_booster_min_dist <= dist200 <= cfg.o4_booster_max_dist)
            and (pd_q_ret1 >= cfg.o4_booster_q_ret1_thr)
            and (pd_q_ret3 >= 0.0)
        )

        ungated_o4_booster = (
            cfg.use_overheat_split
            and (ungated_overheat_stage == 4)
            and (ungated_base_code != 0)
            and o4_booster_cond_ok
        )
        ungated_o4_partial_structural = (
            cfg.use_overheat_split
            and (ungated_overheat_stage == 4)
            and (ungated_base_code != 0)
            and (dist200 <= oh4_pre_buy_level)
        )
        ungated_o4_partial_exec = (
            ungated_o4_partial_structural
            and o4_partial_vol_ok
            and (not ungated_o4_booster)
        )
        o4_booster_exec_ok = (not pre_hard_cash) and ungated_o4_booster
        o4_partial_exec_ok = (not pre_hard_cash) and ungated_o4_partial_exec

        ungated_pre_stop_code = ungated_base_code
        if cfg.use_overheat_split:
            if ungated_overheat_stage == 4:
                if ungated_o4_booster:
                    ungated_pre_stop_code = K.weight_to_code(cfg.o4_booster_pct)
                elif ungated_o4_partial_exec:
                    ungated_pre_stop_code = K.weight_to_code(cfg.oh4_pre_release_buy_pct)
                else:
                    ungated_pre_stop_code = 0
            elif ungated_overheat_stage == 3:
                ungated_pre_stop_code = 0 if ungated_base_code == 0 else 5
            elif ungated_overheat_stage == 2:
                ungated_pre_stop_code = (
                    4 if (ungated_base_code == 2 and not a2_oh12_strong) else ungated_base_code
                )
            elif ungated_overheat_stage == 1:
                ungated_pre_stop_code = (
                    3 if (ungated_base_code == 2 and not a2_oh12_strong) else ungated_base_code
                )

        pre_stop_code = 0 if pre_hard_cash else ungated_pre_stop_code
        base_w = K.code_weight(pre_base_code)
        pre_w = K.code_weight(pre_stop_code)
        cap_binding = (
            cfg.use_overheat_split
            and (not pre_locked)
            and (not pre_spy_force_cash)
            and (not pre_reentry_blocked)
            and (not pre_vix_force_cash)
            and (base_w > pre_w + 1e-9)
            and (pre_overheat_stage > 0)
        )

        # SPY 상한 및 과열 4단 해제 가드 상한 적용
        ungated_before_t3_code = ungated_pre_stop_code
        ungated_spy_cap_applied = False
        ungated_oh4_cap_applied = False
        if pre_spy_bear and K.code_weight(ungated_before_t3_code) > cfg.spy_bear_cap_pct + 1e-9:
            ungated_before_t3_code = K.weight_to_code(cfg.spy_bear_cap_pct)
            ungated_spy_cap_applied = True
        if pre_oh4_reentry_guard and K.code_weight(ungated_before_t3_code) > cfg.oh4_guard_cap_pct + 1e-9:
            ungated_before_t3_code = K.weight_to_code(cfg.oh4_guard_cap_pct)
            ungated_oh4_cap_applied = True

        before_t3_code = 0 if (pre_hard_cash or pre_stop_hit) else ungated_before_t3_code
        spy_filter_applied_flag = (not pre_hard_cash) and (not pre_stop_hit) and ungated_spy_cap_applied
        oh4_guard_applied_flag = (not pre_hard_cash) and (not pre_stop_hit) and ungated_oh4_cap_applied

        before_t3_w = K.code_weight(before_t3_code)
        before_t3_o4_role = (
            2
            if (before_t3_w > 1e-9 and before_t3_code == pre_stop_code and o4_booster_exec_ok)
            else (
                1
                if (before_t3_w > 1e-9 and before_t3_code == pre_stop_code and o4_partial_exec_ok)
                else 0
            )
        )

        # 손절 상태 반영
        stop_event_flag = False
        stop_event_reason = ""
        reentry_lock_next = state.reentry_lock
        if cfg.use_stable_stops and K.is_high_exposure(prev_asset) and state.full_entry_close is not None:
            if pre_stop_hit:
                stop_event_flag = True
                stop_event_reason = "리스크 관리(로스 컷)"
                reentry_lock_next = True

        # 7. T3 필터 적용
        final_code = K.apply_t3_d80_filter(
            before_t3_code,
            cfg.use_t3_d80_filter,
            cfg.enable_t3_cap,
            t3_q_weak or t3_risk_off_zero_state,
            t3_risk_off_zero_state,
            cfg.t3_q_weak_cap_pct,
        )
        t3_risk_off_zero_flag = False
        t3_q_weak_cap_applied_flag = False
        if final_code != before_t3_code:
            if t3_risk_off_zero_state:
                t3_risk_off_zero_flag = True
            elif t3_q_weak:
                t3_q_weak_cap_applied_flag = True

        # 8. VIX 매수보조 증액 제한 (주석 10절)
        prev_w_vix_assist = K.code_weight(prev_asset)
        final_w_vix_assist = K.code_weight(final_code)
        is_existing_upsize = (
            cfg.use_vix_filter
            and cfg.use_vix_buy_support
            and (prev_w_vix_assist > 1e-9)
            and (final_w_vix_assist > prev_w_vix_assist + 1e-9)
            and (not pre_vix_force_cash)
        )
        vix_buy_support_ok = (
            (not cfg.use_vix_filter)
            or (not cfg.use_vix_buy_support)
            or (pre_vix_below_cnt >= cfg.vix_buy_support_days)
        )
        vix_buy_support_blocked_flag = False
        if is_existing_upsize and not vix_buy_support_ok:
            final_code = prev_asset
            vix_buy_support_blocked_flag = True

        # 9. 최종 비중 및 O4 역할 확정
        oh4_release_flag = (
            cfg.use_overheat_split
            and (prev_overheat_stage == 4)
            and (pre_overheat_stage != 4)
            and (K.code_weight(final_code) > K.code_weight(prev_asset) + 1e-9)
        )
        o4_booster_target_w = K.code_weight(K.weight_to_code(cfg.o4_booster_pct))
        o4_partial_target_w = K.code_weight(K.weight_to_code(cfg.oh4_pre_release_buy_pct))
        o4_final_w = K.code_weight(final_code)

        o4_booster_now = (
            cfg.use_overheat_split
            and (pre_overheat_stage == 4)
            and o4_booster_exec_ok
            and (o4_booster_target_w > 1e-9)
            and (o4_final_w > 1e-9)
            and (abs(o4_final_w - o4_booster_target_w) <= 0.0005)
        )
        o4_partial_now = (
            cfg.use_overheat_split
            and (pre_overheat_stage == 4)
            and o4_partial_exec_ok
            and (o4_partial_target_w > 1e-9)
            and (o4_final_w > 1e-9)
            and (abs(o4_final_w - o4_partial_target_w) <= 0.0005)
        )
        now_o4_role = K.o4_role_from_flags(o4_booster_now, o4_partial_now)

        # 양수 목표 확정 시 재진입 잠금 해제 (주석 9절)
        if reentry_lock_next and (K.code_weight(final_code) > 1e-9):
            reentry_lock_next = False

        # 고비중(75% 이상) 진입/유지/이탈에 따른 원금보호 손절 기준가 갱신
        prev_high = K.is_high_exposure(prev_asset)
        now_high = K.is_high_exposure(final_code)
        full_entry_next = state.full_entry_close
        full_high_next = state.full_high_close
        last_stop_next = state.last_principal_stop_price

        if now_high and not prev_high:
            full_entry_next = pd_tc
            full_high_next = pd_tc
            last_stop_next = pd_tc * cfg.principal_stop_pct
        elif now_high and prev_high and full_entry_next is not None:
            full_high_next = max(full_high_next or pd_tc, pd_tc)
            last_stop_next = full_entry_next * cfg.principal_stop_pct
        elif not now_high:
            full_entry_next = None
            full_high_next = None
            last_stop_next = None

        # 평균 진입단가 추적
        prev_w_cb = K.code_weight(prev_asset)
        now_w_cb = K.code_weight(final_code)
        pos_avg_next = state.pos_entry_avg_close
        if now_w_cb <= 1e-9:
            pos_avg_next = None
        elif prev_w_cb <= 1e-9:
            pos_avg_next = pd_tc
        elif now_w_cb > prev_w_cb + 1e-9:
            old_entry = pd_tc if pos_avg_next is None else pos_avg_next
            pos_avg_next = (old_entry * prev_w_cb + pd_tc * (now_w_cb - prev_w_cb)) / now_w_cb

        # 10. D7 비TQQQ 자금 (단기채 / 현금) 배분 갱신 (주석 15절)
        before_tq = prev_w_cb
        before_bill = state.d7_bill_state
        before_cash = max(0.0, 1.0 - before_tq - before_bill)

        d7_keys_next = list(state.d7_keys)
        d7_weights_next = list(state.d7_weights)
        if trade_date in d7_keys_next:
            idx_k = d7_keys_next.index(trade_date)
            d7_weights_next[idx_k] = now_w_cb
        else:
            d7_keys_next.append(trade_date)
            d7_weights_next.append(now_w_cb)
            if len(d7_keys_next) > 8:
                d7_keys_next.pop(0)
                d7_weights_next.pop(0)

        d7_history_complete = len(d7_weights_next) >= 8
        # 초기 8거래일 미만일 때는 누락 일자를 TQQQ 100%(1.0)로 보수 처리
        max_8d_tq = max(d7_weights_next) if d7_history_complete else 1.0
        d7_cap = max(0.0, 1.0 - max_8d_tq)
        target_bill, target_cash = K.d7_allocate(
            before_bill,
            now_w_cb,
            d7_cap,
            same_session=cfg.d7_same_session,
        )

        # 11. 판정 사유(reason_text) 및 태그(tag_text) 구성 (composeSignalLabelReason 동일)
        b_entry = (now_o4_role == 2) and (prev_o4_role != 2)
        b_hold = (now_o4_role == 2) and (prev_o4_role == 2)
        b_exit = (now_o4_role != 2) and (prev_o4_role == 2)
        p_entry = (now_o4_role == 1) and (prev_o4_role != 1)
        p_hold = (now_o4_role == 1) and (prev_o4_role == 1)
        p_exit = (now_o4_role != 1) and (prev_o4_role == 1)
        use_weight_change_label = (
            final_code != 0
            and final_code != 2
            and not stop_event_flag
            and not pre_locked
        )
        t3_q_weak_cap_text = f"{K.code_pct_str(K.weight_to_code(cfg.t3_q_weak_cap_pct))}% cap"
        t3_q_weak_reason_text = f"리스크 관리(QQQ T3D↓, TQQQ {t3_q_weak_cap_text})"

        if stop_event_flag:
            reason_text, tag_text = stop_event_reason, "손절"
        elif pre_locked:
            reason_text, tag_text = "리스크 관리(20일 변동성 락)", "락"
        elif pre_vix_force_cash:
            reason_text, tag_text = "리스크 관리(VIX 위험 기준 이상)", "VIX"
        elif vix_buy_support_blocked_flag:
            reason_text, tag_text = "리스크 관리(VIX 매수보조 대기)", "VIX 대기"
        elif pre_spy_bear and spy_filter_applied_flag:
            reason_text, tag_text = "리스크 관리(SPY 200일 이평선 하락 추세)", "SPY 필터"
        elif pre_reentry_blocked:
            reason_text = f"리스크 관리[QQQ rsi({cfg.rsi_length}) {cfg.rsi_reentry_thr:.0f} 미만]"
            tag_text = "재진입 제한"
        elif t3_risk_off_zero_flag:
            branch_str = K.t3_risk_branch_status_text(t3_short_rate_risk, t3_long_rate_risk)
            reason_text = f"리스크 관리(T3 {branch_str})"
            tag_text = f"T3 {branch_str}"
        elif oh4_release_flag:
            reason_text = "O4 해제(T3cap)" if t3_q_weak_cap_applied_flag else "O4 해제"
            tag_text = reason_text
        elif t3_q_weak_cap_applied_flag:
            reason_text, tag_text = t3_q_weak_reason_text, "T3cap"
        elif b_hold:
            reason_text, tag_text = "O4B 유지", "O4B 유지"
        elif b_entry:
            reason_text, tag_text = "O4B 진입", "O4B 진입"
        elif b_exit:
            reason_text, tag_text = "O4B 이탈", "O4B 이탈"
        elif p_hold:
            reason_text, tag_text = "O4P 유지", "O4P 유지"
        elif p_entry:
            reason_text, tag_text = "O4P 진입", "O4P 진입"
        elif p_exit:
            reason_text, tag_text = "O4P 이탈", "O4P 이탈"
        elif oh4_guard_applied_flag:
            reason_text, tag_text = "리스크 관리(과열 4차 재진입 가드)", "과열가드"
        elif cap_binding:
            reason_text, tag_text = K.stage_reason(pre_overheat_stage), ""
        elif use_weight_change_label:
            reason_text, tag_text = "비중 변경", "비중 변경"
        else:
            if now_w_cb >= 0.75:
                reason_text, tag_text = "추세 조건 만족", ""
            elif now_w_cb > 0:
                reason_text, tag_text = "추세 조건 부분 만족", ""
            else:
                reason_text, tag_text = "추세 조건 불만족", ""

        # 12. 익일 QQQ T3 기준가 (매수/매도 경계) 역산 (주석 17절)
        q9_val = self.q_er10_q9.get(trade_date, np.nan)
        abs9_val = self.q_er10_abs9.get(trade_date, np.nan)
        next_ma_val = self.q_next_ma.get(trade_date, np.nan)
        next_buy_price, next_sell_price = K.q06_next_ticks(
            pd_q0,
            None if pd.isna(q9_val) else float(q9_val),
            None if pd.isna(abs9_val) else float(abs9_val),
            None if pd.isna(next_ma_val) else float(next_ma_val),
            0.01,
        )
        other_filters_permit_hold = K.code_weight(before_t3_code) > 1e-9

        # QQQ 최근 3봉 이모지
        qqq3_emoji = (
            f"{K.dir_emoji(pd_q2, pd_q3)} "
            f"{K.dir_emoji(pd_q1, pd_q2)} "
            f"{K.dir_emoji(pd_q0, pd_q1)}"
        )

        # 원/달러 환율
        krw_sub = b.usdkrw_close[b.usdkrw_close.index <= trade_date]
        usdkrw_c = float(krw_sub.iloc[-1]) if len(krw_sub) >= 1 else None
        usdkrw_p = float(krw_sub.iloc[-2]) if len(krw_sub) >= 2 else None

        # 경계 근접 경고 체크 (당일 최신 판정에서만 수행)
        border_warnings = (
            _check_border_warnings(
                cfg,
                dist200,
                dist_slope,
                pd_tv20,
                a2_er20,
                pd_q06_er,
                a2_er59,
                pd_qqq_rsi14,
                spy_dist200,
                pd_vc,
                dxy_roc,
                dgs6m_chg,
                dgs2_chg,
                us10y_chg,
                dgs20_chg,
            )
            if is_latest_day
            else []
        )

        # 다음 상태 저장
        next_state = DayState(
            last_processed_date=trade_date,
            asset_code=final_code,
            above200_hysteresis=pre_above200,
            above200_hysteresis_inited=pre_above200_inited,
            overheat_stage=pre_overheat_stage,
            full_entry_close=full_entry_next,
            full_high_close=full_high_next,
            last_principal_stop_price=last_stop_next,
            oh4_reentry_guard=pre_oh4_reentry_guard,
            oh4_slope_ok_cnt=pre_oh4_slope_ok_cnt,
            oh4_pre_release_active=(now_o4_role == 1),
            o4_booster_active=(now_o4_role == 2),
            pos_entry_avg_close=pos_avg_next,
            reentry_lock=reentry_lock_next,
            spy_bull=pre_spy_bull,
            spy_inited=pre_spy_inited,
            spy_confirm_cnt=pre_spy_confirm_cnt,
            vix_risk_lock=pre_vix_risk_lock,
            vix_below_cnt=pre_vix_below_cnt,
            d7_keys=d7_keys_next,
            d7_weights=d7_weights_next,
            d7_bill_state=target_bill,
        )

        snapshot = DecisionSnapshot(
            trade_date=trade_date,
            phase=phase,
            prev_asset_code=prev_asset,
            prev_o4_role=prev_o4_role,
            target_asset_code=final_code,
            target_o4_role=now_o4_role,
            before_t3_code=before_t3_code,
            before_t3_o4_role=before_t3_o4_role,
            before_tq=before_tq,
            before_bill=before_bill,
            before_cash=before_cash,
            target_tq=now_w_cb,
            target_bill=target_bill,
            target_cash=target_cash,
            d7_cap=d7_cap,
            d7_history_complete=d7_history_complete,
            reason_text=reason_text,
            tag_text=tag_text,
            stop_event_flag=stop_event_flag,
            locked=pre_locked,
            vix_force_cash=pre_vix_force_cash,
            vix_buy_support_blocked=vix_buy_support_blocked_flag,
            spy_bear_now=pre_spy_bear,
            reentry_blocked_now=pre_reentry_blocked,
            t3_risk_off_zero_flag=t3_risk_off_zero_flag,
            t3_q_weak_cap_applied_flag=t3_q_weak_cap_applied_flag,
            oh4_guard_applied_flag=oh4_guard_applied_flag,
            cap_binding=cap_binding,
            t_c=pd_tc,
            t_p=pd_tp,
            t_ret_pct=((pd_tc / pd_tp) - 1.0) * 100.0 if pd_tp > 0 else 0.0,
            t_ma200=pd_t_ma200,
            dist200=dist200,
            dist_slope=dist_slope,
            t_v20=pd_tv20,
            t_er20=a2_er20,
            above200_hysteresis=pre_above200,
            slope_ok_flag=(not pre_hard_cash and pre_slope_ok),
            overheat_stage=pre_overheat_stage,
            prev_overheat_stage=prev_overheat_stage,
            oh4_reentry_guard=pre_oh4_reentry_guard,
            oh4_slope_ok_cnt=pre_oh4_slope_ok_cnt,
            o4_booster_ok=(pre_overheat_stage == 4 and o4_booster_cond_ok),
            o4_partial_ok=(pre_overheat_stage == 4 and dist200 <= oh4_pre_buy_level and o4_partial_vol_ok),
            principal_stop_price=last_stop_next,
            pos_entry_avg_close=pos_avg_next,
            reentry_lock=reentry_lock_next,
            q_c=pd_q0,
            q_p=pd_q1,
            q_ret1_pct=pd_q_ret1 * 100.0,
            q_ret3_pct=(pd_q_ret3 * 100.0) if pd_q_ret3 is not None else None,
            q_ma3=pd_q_ma3,
            q_ma161=pd_q_ma161,
            q_bull=pre_q_bull,
            qqq_rsi14=pd_qqq_rsi14,
            qqq_er10=pd_q06_er,
            qqq_er59=a2_er59,
            qqq3_emoji=qqq3_emoji,
            next_qqq_buy_price=next_buy_price,
            next_qqq_sell_price=next_sell_price,
            other_filters_permit_hold=other_filters_permit_hold,
            spy_c=pd_spy_c,
            spy_ma200=pd_spy_ma200,
            spy_dist200=spy_dist200,
            spy_bull=pre_spy_bull,
            vix_c=pd_vc,
            vix_p=vix_p,
            vix_below_cnt=pre_vix_below_cnt,
            vix_risk_lock=pre_vix_risk_lock,
            t3_q_weak=t3_q_weak,
            t3_m09_weak=t3_m09_weak,
            dxy_c=pd_dxy_c,
            dxy_ref=pd_dxy_ref,
            dxy_roc=dxy_roc,
            dxy_risk=t3_dxy_risk,
            dxy_source_label=dxy_source_label,
            dgs6m_c=dgs6m_c,
            dgs6m_ref=dgs6m_ref,
            dgs6m_chg=dgs6m_chg,
            dgs6m_risk=t3_dgs6m_risk,
            dgs2_c=dgs2_c,
            dgs2_ref=dgs2_ref,
            dgs2_chg=dgs2_chg,
            dgs2_risk=t3_dgs2_risk,
            us10y_c=us10y_c,
            us10y_ref=us10y_ref,
            us10y_chg=us10y_chg,
            us10y_risk=t3_us10y_risk,
            dgs20_c=dgs20_c,
            dgs20_ref=dgs20_ref,
            dgs20_chg=dgs20_chg,
            dgs20_risk=t3_dgs20_risk,
            t3_short_risk=t3_short_rate_risk,
            t3_long_risk=t3_long_rate_risk,
            dgs_obs_date=dgs_obs_date,
            usdkrw_c=usdkrw_c,
            usdkrw_p=usdkrw_p,
            border_warnings=border_warnings,
        )

        return next_state, snapshot

    def replay_up_to(self, target_date: str) -> Tuple[DayState, DayState, DecisionSnapshot]:
        """
        전체 과거 거래일을 순차 재생하여:
        1) target_date 직전까지의 상태(pre_day_state = c084DayStart)
        2) target_date 당일 판정 후 상태(post_day_state)
        3) target_date 판정 스냅샷(snapshot)
        을 반환합니다.
        """
        if not self.valid_dates:
            raise RuntimeError("유효한 200일 이동평균 거래일 데이터가 부족합니다.")
        if target_date not in self.valid_dates:
            target_date = self.valid_dates[-1]

        state = DayState()
        pre_day_state = DayState()
        last_snap: Optional[DecisionSnapshot] = None

        for d in self.valid_dates:
            if d == target_date:
                pre_day_state = copy.deepcopy(state)
                state, last_snap = self.step_one_day(
                    pre_day_state, d, phase="PRIMARY", is_latest_day=True
                )
                break
            else:
                state, _ = self.step_one_day(state, d, phase="HISTORY", is_latest_day=False)

        assert last_snap is not None
        return pre_day_state, state, last_snap
