"""
변형티큐(V2) 전략 판정 엔진 (Strategy Engine)
변형티큐(V2) 완전 명세서의 9단계 상태머신 규칙과 1차(PRIMARY)·2차(FINAL) 상태 전이 로직을 수행합니다.
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
    변형티큐(V2) 일별 상태머신 상태(ByeonhyeongDayState 대응).
    당일 판정 전 상태(pre_day_state)를 보존하여 같은 날 2차(FINAL) 재판정 시
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
    pending_open_anchor: bool = False
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

    # D7 (단기채 / 현금 분리 - V2 기본 비활성)
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

    # 목표 비중 (TQ / 단기채 / 현금)
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

    # 기간별 전략 수익률 및 TQQQ 벤치마크 수익률 (YTD / 최근 3개월 / 이번달 MTD)
    ytd_ret_pct: Optional[float] = None
    tqqq_ytd_pct: Optional[float] = None
    ret_3m_pct: Optional[float] = None
    tqqq_3m_pct: Optional[float] = None
    mtd_ret_pct: Optional[float] = None
    tqqq_mtd_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _align_macro_to_trading_days(
    series: pd.Series,
    trading_idx: pd.Index,
    ffill_limit: int = 7,
) -> Tuple[pd.Series, pd.Series]:
    """
    V2 명세서 4번 변경점:
    거시지표(국채금리·VIX·DXY)를 주식 거래일 인덱스(trading_idx)에 맞춰
    최대 7일 전방 채움(ffill(limit=7))으로 정렬하고, 원천 관측일(obs_date) 시계열을 함께 반환합니다.
    """
    if series is None or series.empty:
        return (
            pd.Series(np.nan, index=trading_idx, dtype=float),
            pd.Series(None, index=trading_idx, dtype=object),
        )
    clean = series.dropna()
    if clean.empty:
        return (
            pd.Series(np.nan, index=trading_idx, dtype=float),
            pd.Series(None, index=trading_idx, dtype=object),
        )
    union_idx = trading_idx.union(clean.index).sort_values()
    val_aligned = clean.reindex(union_idx).ffill(limit=ffill_limit).reindex(trading_idx)
    dt_raw = pd.Series(clean.index.astype(str), index=clean.index, dtype=object)
    dt_aligned = dt_raw.reindex(union_idx).ffill(limit=ffill_limit).reindex(trading_idx)
    return val_aligned, dt_aligned


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

    # TQQQ 200일 이격도 주요 경계 (V2: 102.00% 진입 / 100.00% 청산)
    dist_thresholds = [
        (cfg.dist200_enter_band, f"200일 진입({cfg.dist200_enter_band:.2f}%)"),
        (cfg.dist200_exit_band, f"200일 청산({cfg.dist200_exit_band:.2f}%)"),
        (cfg.early_full_dist, f"조기100%({cfg.early_full_dist:.2f}%)"),
        (cfg.q_bull_weight_floor_dist, f"보조비중({cfg.q_bull_weight_floor_dist:.2f}%)"),
        (cfg.overheat1_enter, "과열1단 진입(140%)"),
        (cfg.overheat1_exit, "과열1단 해제(132%)"),
        (cfg.overheat2_enter, "과열2단 진입(144%)"),
        (cfg.overheat2_exit, "과열2단 해제(138%)"),
        (cfg.overheat3_enter, "과열3단 진입(147%)"),
        (cfg.overheat3_exit, "과열3단 해제(140%)"),
        (cfg.overheat4_enter, "과열4단 진입(158%)"),
        (cfg.overheat4_exit, "과열4단 해제(118%)"),
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
    변형티큐(V2) 일자별 상태 재생 및 당일 1차(PRIMARY)/2차(FINAL) 판정기
    """

    def __init__(self, bundle: MarketDataBundle, cfg: StrategyConfig = DEFAULT_CONFIG):
        self.bundle = bundle
        self.cfg = cfg
        self._prepare_indicators()

    def _prepare_indicators(self) -> None:
        """전체 시계열에 대해 이동평균, 변동성, 기울기, RSI, ER(10/20/59) 및 1거래일 지연 거시지표를 계산합니다."""
        b = self.bundle
        c = self.cfg
        idx = b.tqqq_close.index

        # TQQQ 파생 지표
        self.t_ma200 = K.calc_sma_series(b.tqqq_close, 200)
        self.t_dist200 = (b.tqqq_close / self.t_ma200) * 100.0
        self.t_ret1 = (b.tqqq_close / b.tqqq_close.shift(1)) - 1.0
        self.t_v20 = K.calc_sample_stdev_series(self.t_ret1, 20)
        self.t_dist_slope = K.calc_linreg_slope_series(self.t_dist200, c.slope_len)
        self.t_er20, self.t_er20_tail, self.t_er20_abs_tail = K.calc_er_series(
            b.tqqq_close, c.tqqq_er20_len
        )

        # QQQ 파생 지표 (V2: use_tv_default_rsi=False 이므로 수정종가 qc 기준 Wilder RSI 산출)
        self.q_ma3 = K.calc_sma_series(b.qqq_close, 3)
        self.q_ma161 = K.calc_sma_series(b.qqq_close, 161)
        self.q_t3ma = K.calc_sma_series(b.qqq_close, c.t3_q_weak_ma_len)
        self.q_next_ma = (b.qqq_close + b.qqq_close.shift(1)) / 2.0  # 익일 SMA3 역산용
        self.q_ret1 = (b.qqq_close / b.qqq_close.shift(1)) - 1.0
        self.q_ret3 = (b.qqq_close / b.qqq_close.shift(2)) - 1.0
        rsi_src = b.qqq_rsi_close if c.use_tv_default_rsi else b.qqq_close
        self.q_rsi14 = K.calc_rsi_wilder_series(rsi_src, c.rsi_length)
        self.q_er10, self.q_er10_q9, self.q_er10_abs9 = K.calc_er_series(
            b.qqq_close, c.qqq_er10_len
        )
        self.q_er59, _, _ = K.calc_er_series(b.qqq_close, c.qqq_er59_len)

        # SPY 파생 지표
        self.s_ma200 = K.calc_sma_series(b.spy_close, 200)
        self.s_dist200 = (b.spy_close / self.s_ma200) * 100.0

        # 거시지표(VIX, DXY, FRED 국채금리 4종) 거래일 정렬(ffill(limit=7)) 및 지연(shift(lag))
        lag = max(0, int(c.macro_lag_days))
        vix_al, _ = _align_macro_to_trading_days(b.vix_close, idx, ffill_limit=7)
        dxy_al, _ = _align_macro_to_trading_days(b.dxy_daily_close, idx, ffill_limit=7)
        d6m_al, d6m_dt = _align_macro_to_trading_days(b.dgs6m, idx, ffill_limit=7)
        d2y_al, d2y_dt = _align_macro_to_trading_days(b.dgs2, idx, ffill_limit=7)
        d10y_al, d10y_dt = _align_macro_to_trading_days(b.dgs10, idx, ffill_limit=7)
        d20y_al, d20y_dt = _align_macro_to_trading_days(b.dgs20, idx, ffill_limit=7)

        self.vix_c_lagged = vix_al.shift(lag)
        self.vix_p_lagged = vix_al.shift(lag + 1)

        self.dxy_c_lagged = dxy_al.shift(lag)
        self.dxy_ref_lagged = dxy_al.shift(lag + c.t3_dxy_roc_len)
        self.dxy_roc_lagged = ((self.dxy_c_lagged / self.dxy_ref_lagged) - 1.0) * 100.0

        self.dgs6m_c_lagged = d6m_al.shift(lag)
        self.dgs6m_ref_lagged = d6m_al.shift(lag + c.t3_dgs6m_chg_len)
        self.dgs6m_chg_lagged = self.dgs6m_c_lagged - self.dgs6m_ref_lagged
        self.dgs6m_obs_lagged = d6m_dt.shift(lag)

        self.dgs2_c_lagged = d2y_al.shift(lag)
        self.dgs2_ref_lagged = d2y_al.shift(lag + c.t3_dgs2_chg_len)
        self.dgs2_chg_lagged = self.dgs2_c_lagged - self.dgs2_ref_lagged
        self.dgs2_obs_lagged = d2y_dt.shift(lag)

        self.us10y_c_lagged = d10y_al.shift(lag)
        self.us10y_ref_lagged = d10y_al.shift(lag + c.t3_us10y_chg_len)
        self.us10y_chg_lagged = self.us10y_c_lagged - self.us10y_ref_lagged
        self.us10y_obs_lagged = d10y_dt.shift(lag)

        self.dgs20_c_lagged = d20y_al.shift(lag)
        self.dgs20_ref_lagged = d20y_al.shift(lag + c.t3_dgs20_chg_len)
        self.dgs20_chg_lagged = self.dgs20_c_lagged - self.dgs20_ref_lagged
        self.dgs20_obs_lagged = d20y_dt.shift(lag)

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
        변형티큐(V2) 9단계 일봉 상태머신 판정 순서를 정확히 따릅니다.
        """
        cfg = self.cfg
        b = self.bundle
        state = copy.deepcopy(base_state)

        prev_asset = state.asset_code
        prev_o4_role = 0

        # 1. 해당 거래일의 주식 원천 값 추출 (TQQQ 시가/종가, QQQ, SPY)
        pd_tc = float(b.tqqq_close.loc[trade_date])
        pd_to = (
            float(b.tqqq_open.loc[trade_date])
            if (b.tqqq_open is not None and trade_date in b.tqqq_open.index)
            else pd_tc
        )
        t_idx = b.tqqq_close.index.get_loc(trade_date)
        pd_tp = float(b.tqqq_close.iloc[t_idx - 1]) if t_idx > 0 else pd_tc
        pd_t_ma200 = float(self.t_ma200.loc[trade_date])
        dist200 = float(self.t_dist200.loc[trade_date])
        pd_tv20 = float(self.t_v20.loc[trade_date])
        dist_slope_val = self.t_dist_slope.loc[trade_date]
        dist_slope = None if pd.isna(dist_slope_val) else float(dist_slope_val)

        # [V2 핵심 변경점 3]: 직전 거래일에 75% 이상 신규 진입 신호가 발생했다면,
        # 실제 진입 체결일인 오늘(익일) 시가(pd_to)로 원금보호 손절 기준가(full_entry_close)를 확정합니다.
        if cfg.use_next_open_stop_anchor and state.pending_open_anchor and K.is_high_exposure(prev_asset):
            if pd_to > 0:
                state.full_entry_close = pd_to
                state.full_high_close = max(pd_to, pd_tc)
                state.last_principal_stop_price = pd_to * cfg.principal_stop_pct
            state.pending_open_anchor = False

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

        er10_val = self.q_er10.get(trade_date, np.nan)
        pd_q06_er = None if pd.isna(er10_val) else float(er10_val)
        pd_q06_ready = pd_q06_er is not None

        # 2. VIX, DXY, FRED 국채금리 4종 추출 (V2: 1거래일 지연 shift(1) 전일 확정치 사용)
        def _opt_float(s: pd.Series, dt: str) -> Optional[float]:
            v = s.get(dt, np.nan)
            return None if pd.isna(v) else float(v)

        pd_vc = _opt_float(self.vix_c_lagged, trade_date)
        vix_p = _opt_float(self.vix_p_lagged, trade_date)

        pd_dxy_c = _opt_float(self.dxy_c_lagged, trade_date)
        pd_dxy_ref = _opt_float(self.dxy_ref_lagged, trade_date)
        dxy_roc = _opt_float(self.dxy_roc_lagged, trade_date)
        dxy_source_label = "전일 확정(shift1)" if pd_dxy_c is not None else "미확인"

        dgs6m_c = _opt_float(self.dgs6m_c_lagged, trade_date)
        dgs6m_ref = _opt_float(self.dgs6m_ref_lagged, trade_date)
        dgs6m_chg = _opt_float(self.dgs6m_chg_lagged, trade_date)

        dgs2_c = _opt_float(self.dgs2_c_lagged, trade_date)
        dgs2_ref = _opt_float(self.dgs2_ref_lagged, trade_date)
        dgs2_chg = _opt_float(self.dgs2_chg_lagged, trade_date)

        us10y_c = _opt_float(self.us10y_c_lagged, trade_date)
        us10y_ref = _opt_float(self.us10y_ref_lagged, trade_date)
        us10y_chg = _opt_float(self.us10y_chg_lagged, trade_date)

        dgs20_c = _opt_float(self.dgs20_c_lagged, trade_date)
        dgs20_ref = _opt_float(self.dgs20_ref_lagged, trade_date)
        dgs20_chg = _opt_float(self.dgs20_chg_lagged, trade_date)

        obs_2y = self.dgs2_obs_lagged.get(trade_date, None)
        obs_10y = self.us10y_obs_lagged.get(trade_date, None)
        obs_6m = self.dgs6m_obs_lagged.get(trade_date, None)
        obs_20y = self.dgs20_obs_lagged.get(trade_date, None)
        dgs_obs_date = None
        for cand in (obs_2y, obs_10y, obs_6m, obs_20y):
            if cand is not None and not pd.isna(cand):
                dgs_obs_date = str(cand)
                break

        # 3. T3 약세 및 거시위험 판정 (V2 6단계)
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

        # 4. V2 상태머신 단계별 판정 (1단계 ~ 5단계)
        s1_enter = cfg.overheat1_enter
        s1_exit = min(cfg.overheat1_exit, s1_enter - 0.01)
        s2_enter = max(cfg.overheat2_enter, s1_enter + 0.01)
        s2_exit = min(cfg.overheat2_exit, s2_enter - 0.01)
        s3_enter = max(cfg.overheat3_enter, s2_enter + 0.01)
        s3_exit = min(cfg.overheat3_exit, s3_enter - 0.01)
        s4_enter = max(cfg.overheat4_enter, s3_enter + 0.01)
        s4_exit = min(cfg.overheat4_exit, s4_enter - 0.01)

        # (1단계) VIX 카운터 갱신 (결측 시 vix_buy_support_days=5로 간주)
        pre_vix_risk_lock, pre_vix_below_cnt = K.next_vix_state(
            cfg.use_vix_filter,
            state.vix_risk_lock,
            state.vix_below_cnt,
            pd_vc,
            cfg.vix_sell_threshold,
            cfg.vix_rebuy_threshold,
            carry_only=False,
            default_below_cnt=cfg.vix_buy_support_days,
        )
        pre_vix_force_cash = cfg.use_vix_filter and cfg.use_vix_force_sell and pre_vix_risk_lock

        # (2단계) 3대 강제 현금화(hard_cash): 20일 변동성 락, SPY 200일선 필터, 손절 후 RSI(<43) 재진입 잠금
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

        # (3단계) 기본 추세 비중(base0: 102.0% 진입 / 100.0% 이탈) 및 45일 기울기 부스트(slope_ok)
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

        # (4단계) 과열 1~4단 분할매도 및 과열 4단 해제 가드(95%)
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

        # (5단계) 원금보호 로스컷(-7.0% 손절): 75% 이상 보유 중 진입 체결 시가 대비 93.0% 이하 시 즉시 손절
        pre_stop_hit = (
            cfg.use_stable_stops
            and K.is_high_exposure(prev_asset)
            and (state.full_entry_close is not None)
            and (pd_tc <= state.full_entry_close * cfg.principal_stop_pct)
        )

        # 과열 단계별 비중 판정 (V2: 과열 4단은 O4B/O4P 예외 없이 무조건 0%)
        a2_oh12_strong = a2_er20_ready and (a2_er20 > cfg.tqqq_er20_thr)
        ungated_pre_stop_code = ungated_base_code
        if cfg.use_overheat_split:
            if ungated_overheat_stage == 4:
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
        before_t3_o4_role = 0

        # 손절 상태 반영
        stop_event_flag = False
        stop_event_reason = ""
        reentry_lock_next = state.reentry_lock
        if cfg.use_stable_stops and K.is_high_exposure(prev_asset) and state.full_entry_close is not None:
            if pre_stop_hit:
                stop_event_flag = True
                stop_event_reason = "리스크 관리(로스 컷)"
                reentry_lock_next = True

        # (6단계) T3 거시 금리·달러 발작 필터 적용
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

        # (7단계) VIX 5일 매수보조 제한
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

        # (8~9단계) 최종 비중 확정 및 손절 기준가 갱신
        oh4_release_flag = (
            cfg.use_overheat_split
            and (prev_overheat_stage == 4)
            and (pre_overheat_stage != 4)
            and (K.code_weight(final_code) > K.code_weight(prev_asset) + 1e-9)
        )
        now_o4_role = 0

        # 양수 목표 확정 시 재진입 잠금 해제
        if reentry_lock_next and (K.code_weight(final_code) > 1e-9):
            reentry_lock_next = False

        # 고비중(75% 이상) 진입/유지/이탈에 따른 원금보호 손절 기준가 갱신
        # (당일 신규 진입 신호 시 임시로 당일 종가를 기록하고 pending_open_anchor=True로 설정하여 익일 시가로 확정)
        prev_high = K.is_high_exposure(prev_asset)
        now_high = K.is_high_exposure(final_code)
        full_entry_next = state.full_entry_close
        full_high_next = state.full_high_close
        last_stop_next = state.last_principal_stop_price
        pending_open_next = state.pending_open_anchor

        if now_high and not prev_high:
            full_entry_next = pd_tc
            full_high_next = pd_tc
            last_stop_next = pd_tc * cfg.principal_stop_pct
            pending_open_next = cfg.use_next_open_stop_anchor
        elif now_high and prev_high and full_entry_next is not None:
            full_high_next = max(full_high_next or pd_tc, pd_tc)
            last_stop_next = full_entry_next * cfg.principal_stop_pct
            pending_open_next = False
        elif not now_high:
            full_entry_next = None
            full_high_next = None
            last_stop_next = None
            pending_open_next = False

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

        # 10. V2 원신호 비중 (d7_enabled=False 일 때 TQQQ + 현금으로만 구성)
        before_tq = prev_w_cb
        d7_keys_next = list(state.d7_keys)
        d7_weights_next = list(state.d7_weights)
        if cfg.d7_enabled:
            before_bill = state.d7_bill_state
            before_cash = max(0.0, 1.0 - before_tq - before_bill)
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
            max_8d_tq = max(d7_weights_next) if d7_history_complete else 1.0
            d7_cap = max(0.0, 1.0 - max_8d_tq)
            target_bill, target_cash = K.d7_allocate(
                before_bill,
                now_w_cb,
                d7_cap,
                same_session=cfg.d7_same_session,
            )
        else:
            before_bill = 0.0
            before_cash = max(0.0, 1.0 - before_tq)
            target_bill = 0.0
            target_cash = max(0.0, 1.0 - now_w_cb)
            d7_cap = 0.0
            d7_history_complete = True

        # 11. 판정 사유(reason_text) 및 태그(tag_text) 구성
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

        # 12. 익일 QQQ T3 기준가 (매수/매도 경계) 역산
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
            pending_open_anchor=pending_open_next,
            oh4_reentry_guard=pre_oh4_reentry_guard,
            oh4_slope_ok_cnt=pre_oh4_slope_ok_cnt,
            oh4_pre_release_active=False,
            o4_booster_active=False,
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
            o4_booster_ok=False,
            o4_partial_ok=False,
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
            ytd_ret_pct=getattr(self, "_cached_perf", {}).get("ytd_ret_pct"),
            tqqq_ytd_pct=getattr(self, "_cached_perf", {}).get("tqqq_ytd_pct"),
            ret_3m_pct=getattr(self, "_cached_perf", {}).get("ret_3m_pct"),
            tqqq_3m_pct=getattr(self, "_cached_perf", {}).get("tqqq_3m_pct"),
            mtd_ret_pct=getattr(self, "_cached_perf", {}).get("mtd_ret_pct"),
            tqqq_mtd_pct=getattr(self, "_cached_perf", {}).get("tqqq_mtd_pct"),
        )

        return next_state, snapshot

    def replay_up_to(self, target_date: str) -> Tuple[DayState, DayState, DecisionSnapshot]:
        """
        전체 과거 거래일을 순차 재생하여:
        1) target_date 직전까지의 상태(pre_day_state)
        2) target_date 당일 판정 후 상태(post_day_state)
        3) target_date 판정 스냅샷(snapshot, YTD/3M/MTD 수익률 포함)
        을 반환합니다.
        """
        if not self.valid_dates:
            raise RuntimeError("유효한 200일 이동평균 거래일 데이터가 부족합니다.")
        if target_date not in self.valid_dates:
            target_date = self.valid_dates[-1]

        state = DayState()
        pre_day_state = DayState()
        last_snap: Optional[DecisionSnapshot] = None

        # 일별 V2 전략 누적 자산곡선(eq, 익일 시가 체결 반영) 및 TQQQ 종가 기록
        tqqq_sh = 0.0
        usd = 1.0
        prev_target_w: Optional[float] = None
        pending_sig_w = 0.0
        c = 0.0003  # 편도 3bp
        eq_by_date: Dict[str, float] = {}
        tc_by_date: Dict[str, float] = {}
        replayed_dates: List[str] = []

        for i, d in enumerate(self.valid_dates):
            tc_now = float(self.bundle.tqqq_close.loc[d])
            to_now = (
                float(self.bundle.tqqq_open.loc[d])
                if (self.bundle.tqqq_open is not None and d in self.bundle.tqqq_open.index)
                else tc_now
            )
            if i > 0 and to_now > 0:
                should_rebal = (prev_target_w is None) or (abs(pending_sig_w - prev_target_w) > 1e-9)
                if should_rebal:
                    pre_open = tqqq_sh * to_now + usd
                    cur_w = (tqqq_sh * to_now) / pre_open if pre_open > 0 else 0.0
                    turnover = abs(pending_sig_w - cur_w)
                    post_open = pre_open * (1.0 - c * turnover) / (1.0 - c * pending_sig_w)
                    tqqq_sh = (post_open * pending_sig_w) / to_now
                    usd = post_open * (1.0 - pending_sig_w)
                    prev_target_w = pending_sig_w

            eq_close = tqqq_sh * tc_now + usd
            eq_by_date[d] = eq_close
            tc_by_date[d] = tc_now
            replayed_dates.append(d)

            if d == target_date:
                cur_year = target_date[:4]
                cur_ym = target_date[:7]
                prev_year_dates = [x for x in replayed_dates if x[:4] < cur_year]
                prev_month_dates = [x for x in replayed_dates if x[:7] < cur_ym]

                base_ytd_d = prev_year_dates[-1] if prev_year_dates else replayed_dates[0]
                base_mtd_d = prev_month_dates[-1] if prev_month_dates else replayed_dates[0]
                idx_3m = max(0, len(replayed_dates) - 1 - 63)
                base_3m_d = replayed_dates[idx_3m]

                self._cached_perf = {
                    "ytd_ret_pct": ((eq_by_date[target_date] / eq_by_date[base_ytd_d]) - 1.0) * 100.0,
                    "tqqq_ytd_pct": ((tc_by_date[target_date] / tc_by_date[base_ytd_d]) - 1.0) * 100.0,
                    "ret_3m_pct": ((eq_by_date[target_date] / eq_by_date[base_3m_d]) - 1.0) * 100.0,
                    "tqqq_3m_pct": ((tc_by_date[target_date] / tc_by_date[base_3m_d]) - 1.0) * 100.0,
                    "mtd_ret_pct": ((eq_by_date[target_date] / eq_by_date[base_mtd_d]) - 1.0) * 100.0,
                    "tqqq_mtd_pct": ((tc_by_date[target_date] / tc_by_date[base_mtd_d]) - 1.0) * 100.0,
                }

                pre_day_state = copy.deepcopy(state)
                state, last_snap = self.step_one_day(
                    pre_day_state, d, phase="PRIMARY", is_latest_day=True
                )
                break
            else:
                state, _ = self.step_one_day(state, d, phase="HISTORY", is_latest_day=False)
                pending_sig_w = K.code_weight(state.asset_code)

        assert last_snap is not None
        return pre_day_state, state, last_snap
