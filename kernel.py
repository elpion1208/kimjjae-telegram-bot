"""
변형티큐법 핵심 연산 커널 및 표시 포맷 헬퍼
Pine Script의 KimJjaeDecisionKernel(K), KimJjaeBoardDisplay(boardDisplay) 및 수학 헬퍼 함수를 구현합니다.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


# ============================================================
# 1. 비중 코드(Asset Code) <-> 목표비중(Weight) 변환 (K.codeWeight / K.weightToCode)
# ============================================================

def code_weight(code: Optional[int]) -> float:
    """
    Pine Script K.codeWeight(code) 대응:
    기존 정수 코드:
      0 -> 0.00 (0%, 현금)
      2 -> 1.00 (100%)
      3 -> 0.90 (90%, 과열 1단)
      4 -> 0.80 (80%, 과열 2단)
      5 -> 0.05 (5%, 과열 3단)
    확장 정수 코드 (1000 + 천분율):
      예: 1100 -> 0.10 (10%), 1200 -> 0.20 (20%), 1300 -> 0.30 (30%),
          1700 -> 0.70 (70%), 1950 -> 0.95 (95%)
    """
    if code is None:
        return 0.0
    if code == 0:
        return 0.0
    if code == 2:
        return 1.0
    if code == 3:
        return 0.90
    if code == 4:
        return 0.80
    if code == 5:
        return 0.05
    if code >= 1000:
        return max(0.0, min(1.0, (code - 1000) / 1000.0))
    return 0.0


def weight_to_code(weight: float) -> int:
    """
    Pine Script K.weightToCode(weight) 대응:
    천분율(1/1000) 단위로 반올림하며 기본 코드(0, 2, 3, 4, 5)를 우선 매핑합니다.
    """
    if weight is None or math.isnan(weight) or weight <= 1e-9:
        return 0
    w = max(0.0, min(1.0, weight))
    units = int(round(w * 1000.0))
    if units <= 0:
        return 0
    if units >= 1000:
        return 2
    if units == 900:
        return 3
    if units == 800:
        return 4
    if units == 50:
        return 5
    return 1000 + units


def code_pct_str(code: Optional[int]) -> str:
    """비중 코드를 퍼센트 문자열(예: '100', '90', '5', '0')로 변환합니다."""
    w = code_weight(code) * 100.0
    if abs(w - round(w)) < 1e-6:
        return str(int(round(w)))
    return f"{w:.1f}"


def is_high_exposure(code: Optional[int]) -> bool:
    """직전/현재 목표비중이 75% 이상인지 판정합니다."""
    return code_weight(code) >= 0.75 - 1e-9


def o4_role_from_flags(is_booster: bool, is_partial: bool) -> int:
    """O4 역할 코드: 0=일반, 1=O4P(70%), 2=O4B(30%)"""
    return 2 if is_booster else (1 if is_partial else 0)


# ============================================================
# 2. 수학 및 지표 계산 헬퍼 (Pine Script Math Helpers)
# ============================================================

def calc_sma_series(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()


def calc_sample_stdev_series(series: pd.Series, length: int) -> pd.Series:
    """
    Pine Script sampleStdev(src, len):
    표본표준편차 (ddof=1)
    """
    return series.rolling(window=length, min_periods=length).std(ddof=1)


def calc_linreg_slope_series(dist_series: pd.Series, length: int) -> pd.Series:
    """
    Pine Script dailyDistSlope(src, maLen, len):
    최근 len개 구간의 선형회귀 기울기(slope)
    """
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    return dist_series.rolling(window=length, min_periods=length).apply(_slope, raw=True)


def calc_rsi_wilder_series(close_series: pd.Series, length: int = 14) -> pd.Series:
    """
    Pine Script rsiWilder(src, len):
    Wilder's RMA (alpha = 1 / len) 기반 RSI 계산.
    첫 유효 구간은 단순평균(SMA)으로 초기화한 뒤 RMA 점화식을 수행합니다.
    """
    vals = close_series.to_numpy(dtype=float)
    n = len(vals)
    rsi = np.full(n, np.nan, dtype=float)
    if n <= length:
        return pd.Series(rsi, index=close_series.index)

    diff = np.diff(vals, prepend=np.nan)
    up = np.where(np.isnan(diff), np.nan, np.maximum(diff, 0.0))
    dn = np.where(np.isnan(diff), np.nan, np.maximum(-diff, 0.0))

    au = np.nan
    ad = np.nan
    alpha = 1.0 / float(length)

    valid_idxs = np.where(~np.isnan(diff))[0]
    if len(valid_idxs) < length:
        return pd.Series(rsi, index=close_series.index)

    start_idx = valid_idxs[length - 1]
    window_slice = valid_idxs[:length]
    au = float(np.mean(up[window_slice]))
    ad = float(np.mean(dn[window_slice]))

    def _to_rsi(u: float, d: float) -> float:
        if u == 0.0 and d == 0.0:
            return 50.0
        if d == 0.0:
            return 100.0
        if u == 0.0:
            return 0.0
        return 100.0 - (100.0 / (1.0 + (u / d)))

    rsi[start_idx] = _to_rsi(au, ad)
    for i in range(start_idx + 1, n):
        if np.isnan(up[i]) or np.isnan(dn[i]):
            continue
        au = alpha * up[i] + (1.0 - alpha) * au
        ad = alpha * dn[i] + (1.0 - alpha) * ad
        rsi[i] = _to_rsi(au, ad)

    return pd.Series(rsi, index=close_series.index)


def calc_er_series(close_series: pd.Series, length: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    ER 공통 정의:
    ER_N = ln(P_t / P_{t-N}) / sum_{i=t-N+1..t} |ln(P_i / P_{i-1})|
    """
    vals = close_series.to_numpy(dtype=float)
    n = len(vals)
    er_out = np.full(n, np.nan, dtype=float)
    tail_out = np.full(n, np.nan, dtype=float)
    abs_tail_out = np.full(n, np.nan, dtype=float)

    for i in range(length, n):
        window = vals[i - length : i + 1]
        if np.isnan(window).any() or np.any(window <= 0):
            continue
        log_moves = np.abs(np.log(window[1:] / window[:-1]))
        total_abs = float(np.sum(log_moves))
        abs_tail = float(np.sum(log_moves[1:]))
        tail_price = float(vals[i - (length - 1)])
        net_log = math.log(vals[i] / vals[i - length])
        er_val = 0.0 if total_abs == 0.0 else (net_log / total_abs)

        er_out[i] = er_val
        tail_out[i] = tail_price
        abs_tail_out[i] = abs_tail

    idx = close_series.index
    return (
        pd.Series(er_out, index=idx),
        pd.Series(tail_out, index=idx),
        pd.Series(abs_tail_out, index=idx),
    )


def q06_next_weak(p_q0: float, p_q9: float, p_abs9: float, p_next_ma: float, price: float) -> bool:
    """익일 QQQ 종가가 price일 때 T3 약세(SMA3 하회 또는 ER10 <= -0.300)인지 판정"""
    move = math.log(price / p_q0)
    den = p_abs9 + abs(move)
    er = 0.0 if den == 0.0 else math.log(price / p_q9) / den
    return (price < p_next_ma) or (er <= -0.3)


def q06_next_ticks(
    p_q0: float,
    p_q9: float,
    p_abs9: float,
    p_next_ma: float,
    p_tick: float = 0.01,
) -> Tuple[Optional[float], Optional[float]]:
    """
    익일 QQQ의 T3 약세/해제 경계가 되는 매수 기준가(buy: 약세 해제 첫 가격)와
    매도 기준가(sell: 약세 유지 마지막 가격)를 최소호가(0.01달러) 단위로 계산합니다.
    """
    if (
        p_q0 is None or p_q9 is None or p_abs9 is None or p_next_ma is None
        or math.isnan(p_q0) or math.isnan(p_q9) or math.isnan(p_abs9) or math.isnan(p_next_ma)
        or p_q0 <= 0 or p_q9 <= 0 or p_next_ma <= 0 or p_tick <= 0
    ):
        return None, None

    a = -0.3 * p_abs9 - math.log(p_q0 / p_q9)
    er_price = p_q0 if p_abs9 == 0.0 else p_q0 * math.exp(a / 1.3 if a >= 0.0 else a / 0.7)
    bound = max(p_next_ma, er_price)
    lower = math.floor(bound / p_tick) * p_tick
    if lower <= 0:
        return None, None

    if not q06_next_weak(p_q0, p_q9, p_abs9, p_next_ma, lower):
        lower -= p_tick
    elif q06_next_weak(p_q0, p_q9, p_abs9, p_next_ma, lower + p_tick):
        lower += p_tick

    if (
        lower > 0
        and q06_next_weak(p_q0, p_q9, p_abs9, p_next_ma, lower)
        and not q06_next_weak(p_q0, p_q9, p_abs9, p_next_ma, lower + p_tick)
    ):
        sell = round(lower, 2)
        buy = round(lower + p_tick, 2)
        return buy, sell
    return None, None


# ============================================================
# 3. 상태 전이 함수 (KimJjaeDecisionKernel K.* 구현)
# ============================================================

def next_above200_state(
    prev_state: bool,
    prev_inited: bool,
    dist200: float,
    enter_band: float,
    exit_band: float,
) -> Tuple[bool, bool]:
    """
    V2 명세서 3단계:
    TQQQ 이격도 >= 102.0이면 상향 상태가 켜지고, <= 100.0이면 꺼집니다. 그 사이는 직전 상태를 유지합니다.
    """
    if dist200 is None or math.isnan(dist200):
        return prev_state, prev_inited
    if not prev_inited:
        state = dist200 >= enter_band
        return state, True
    if prev_state:
        state = False if dist200 <= exit_band else True
    else:
        state = True if dist200 >= enter_band else False
    return state, True


def next_spy_state(
    use_spy_filter: bool,
    prev_bull: bool,
    prev_inited: bool,
    prev_confirm_cnt: int,
    spy_dist200: Optional[float],
    enter_pct: float,
    exit_pct: float,
    confirm_days: int,
) -> Tuple[bool, bool, int]:
    """
    SPY 이격도 <= 95.70이면 하락 상태, >= 96.50이면 상승 상태로 전환하며 기본 확인일수는 1회입니다.
    """
    if not use_spy_filter:
        return True, True, 0
    if spy_dist200 is None or math.isnan(spy_dist200):
        return prev_bull, prev_inited, prev_confirm_cnt

    req = max(confirm_days, 1)
    if not prev_inited:
        bull = spy_dist200 >= enter_pct
        return bull, True, 0

    if prev_bull:
        if spy_dist200 <= exit_pct:
            cnt = prev_confirm_cnt + 1
            if cnt >= req:
                return False, True, 0
            return True, True, cnt
        return True, True, 0
    else:
        if spy_dist200 >= enter_pct:
            cnt = prev_confirm_cnt + 1
            if cnt >= req:
                return True, True, 0
            return False, True, cnt
        return False, True, 0


def next_vix_state(
    use_vix_filter: bool,
    prev_lock: bool,
    prev_below_cnt: int,
    vix_close: Optional[float],
    sell_thr: float,
    rebuy_thr: float,
    carry_only: bool = False,
    default_below_cnt: int = 5,
) -> Tuple[bool, int]:
    """
    V2 명세서 1단계:
    VIX <= 31인 연속 일봉 확인횟수를 카운트하며 VIX > 31이면 0으로 초기화합니다.
    VIX 결측 시 V2 명세서에 따라 연속 확인일수를 default_below_cnt(5일)로 간주합니다.
    """
    if not use_vix_filter:
        return False, 0
    if vix_close is None or math.isnan(vix_close):
        if carry_only:
            return prev_lock, prev_below_cnt
        return prev_lock, default_below_cnt

    if prev_lock:
        lock = False if vix_close <= rebuy_thr else True
    else:
        lock = True if vix_close >= sell_thr else False

    below_cnt = (prev_below_cnt + 1) if vix_close <= rebuy_thr else 0
    return lock, below_cnt


def next_overheat_stage(
    prev_stage: int,
    use_overheat_split: bool,
    dist200: Optional[float],
    ungated_base_code: int,
    hard_cash: bool,
    s1_enter: float,
    s1_exit: float,
    s2_enter: float,
    s2_exit: float,
    s3_enter: float,
    s3_exit: float,
    s4_enter: float,
    s4_exit: float,
) -> int:
    """
    과열 단계 전이 규칙:
    - 기존 4단에서는 이격도가 결측이면 4단을 유지하고, 유효값이 <= 118(s4_exit)일 때 0단으로 해제합니다.
    - 진입 경계: 1단 140 / 2단 144 / 3단 147 / 4단 158 이상.
    - 하락 해제 경계: 132 / 138 / 140 / 118 이하.
    """
    if not use_overheat_split:
        return 0
    if prev_stage == 4:
        if dist200 is None or math.isnan(dist200):
            return 4
        return 4 if dist200 > s4_exit else 0

    if hard_cash:
        return 0
    if dist200 is None or math.isnan(dist200):
        return prev_stage

    stage = prev_stage
    if stage == 3 and dist200 <= s3_exit:
        stage = 2
    if stage == 2 and dist200 <= s2_exit:
        stage = 1
    if stage == 1 and dist200 <= s1_exit:
        stage = 0

    can_enter_or_promote = (ungated_base_code == 2) or (stage > 0)
    if can_enter_or_promote:
        if dist200 >= s4_enter:
            stage = 4
        elif dist200 >= s3_enter:
            stage = max(stage, 3)
        elif dist200 >= s2_enter:
            stage = max(stage, 2)
        elif dist200 >= s1_enter and ungated_base_code == 2:
            stage = max(stage, 1)

    return stage


def apply_t3_d80_filter(
    before_t3_code: int,
    use_t3_filter: bool,
    enable_t3_cap: bool,
    weak_or_risk_zero: bool,
    risk_off_zero: bool,
    t3_q_weak_cap: float,
) -> int:
    """
    T3 강한위험(약세 AND (SR OR (DXY AND LR)))이면 목표비중 0%
    """
    if not use_t3_filter:
        return before_t3_code
    w = code_weight(before_t3_code)
    if w <= 1e-9:
        return 0
    if risk_off_zero:
        return 0
    if enable_t3_cap and weak_or_risk_zero and w > t3_q_weak_cap + 1e-9:
        return weight_to_code(t3_q_weak_cap)
    return before_t3_code


# ============================================================
# 4. D7 비TQQQ 자금 (단기채 / 현금 분리) 계산
# ============================================================

def d7_allocate(
    previous_bill: float,
    weight: float,
    capacity: float,
    same_session: bool = True,
) -> Tuple[float, float]:
    weight_units = int(round(weight * 1000.0))
    previous_units = int(round(previous_bill * 1000.0))
    capacity_units = int(round(capacity * 1000.0))

    bill_units = min(max(previous_units, 0), 1000 - weight_units)
    cash_units = 1000 - weight_units - bill_units
    addition_units = min(max(0, capacity_units - bill_units), cash_units)
    if same_session and addition_units >= 5:
        bill_units += addition_units
    cash_units = 1000 - weight_units - bill_units
    return bill_units / 1000.0, cash_units / 1000.0


# ============================================================
# 5. 표시 및 텔레그램 문자열 포맷 헬퍼
# ============================================================

def d7_pct_str(weight: float) -> str:
    val = round(weight * 1000.0) / 10.0
    if abs(val - round(val)) < 1e-6:
        return f"{int(round(val))}"
    return f"{val:.1f}"


def alert_pos_icon(code: int, o4_role: int = 0) -> str:
    w = code_weight(code)
    if w <= 1e-9:
        return "⚪"
    if o4_role == 2:
        return "🩵(O4B)"
    if o4_role == 1:
        return "🔵(O4P)"
    if w >= 0.999:
        return "🟢"
    if w >= 0.70:
        return "🟩"
    return "🟠"


def asset_name_alert(code: int) -> str:
    w = code_weight(code)
    if w <= 1e-9:
        return "TQQQ(0%) · 현금(100%)"
    if w >= 1.0 - 1e-9:
        return "TQQQ(100%)"
    cash_code = weight_to_code(max(0.0, 1.0 - w))
    return f"TQQQ({code_pct_str(code)}%) · 현금({code_pct_str(cash_code)}%)"


def d7_breakdown(
    stock: float,
    bill: float,
    cash: float,
    stock_icon: str = "",
    include_stock: bool = True,
) -> str:
    parts: List[str] = []
    if include_stock:
        icon_suffix = f" {stock_icon}" if stock_icon else ""
        parts.append(f"TQ({d7_pct_str(stock)}%){icon_suffix}")
    if bill > 1e-9:
        parts.append(f"단기채({d7_pct_str(bill)}%)")
    if cash > 1e-9 or (not include_stock and bill <= 1e-9):
        parts.append(f"현금({d7_pct_str(cash)}%)")
    return " · ".join(parts)


def alert_underline_position(pos_value: str) -> str:
    """목표 포지션 밑줄(<u>) 강조"""
    marked = pos_value.replace("%)", "%)</u>")
    marked = marked.replace("TQ(", "<u>TQ(")
    marked = marked.replace("TQQQ(", "<u>TQQQ(")
    marked = marked.replace("단기채(", "<u>단기채(")
    marked = marked.replace("현금(", "<u>현금(")
    marked = marked.replace("현금(SGOV, BOXX)", "<u>현금(SGOV, BOXX)</u>")
    return marked


def stage_reason(stage: int) -> str:
    if stage == 1:
        return "분할매도 1차(200일 이격도 과열)"
    if stage == 2:
        return "분할매도 2차(200일 이격도 과열)"
    if stage == 3:
        return "분할매도 3차(200일 이격도 과열)"
    if stage == 4:
        return "분할매도 4차(200일 이격도 과열)"
    return "기본 비중 유지"


def dir_emoji(curr: Optional[float], prev: Optional[float]) -> str:
    if curr is None or prev is None or math.isnan(curr) or math.isnan(prev):
        return "⬜"
    if curr > prev:
        return "🟥"
    if curr < prev:
        return "🟦"
    return "⬜"


def t3_risk_branch_status_text(short_risk: bool, long_risk: bool) -> str:
    if short_risk and long_risk:
        return "SR+LR 위험"
    if short_risk:
        return "SR 위험"
    if long_risk:
        return "LR 위험"
    return "정상"


def t3_risk_items_text(
    short_risk: bool,
    long_risk: bool,
    risk_6m: bool,
    risk_2y: bool,
    risk_10y: bool,
    risk_20y: bool,
    risk_dxy: bool,
) -> str:
    items: List[str] = []
    if short_risk:
        sr_parts = []
        if risk_2y:
            sr_parts.append("2Y")
        if risk_6m:
            sr_parts.append("6M")
        items.append("SR(" + "+".join(sr_parts) + ")")
    if long_risk:
        lr_parts = ["DXY"] if risk_dxy else []
        if risk_10y:
            lr_parts.append("10Y")
        if risk_20y:
            lr_parts.append("20Y")
        items.append("LR(" + "+".join(lr_parts) + ")")
    return " / ".join(items) if items else "해당 없음"
