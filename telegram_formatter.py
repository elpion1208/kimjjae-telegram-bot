"""
변형티큐(V2) 텔레그램 알림 메시지 포맷터 (Telegram Formatter)
요청하신 간결한 3가지 핵심 항목(날짜, 현재 포지션, 수익률 3종)만 깔끔하게 표시합니다.
"""

from __future__ import annotations
import math
from typing import Optional

from config import StrategyConfig, DEFAULT_CONFIG
from strategy_engine import DecisionSnapshot
import kernel as K


def _fmt_pct(val: Optional[float]) -> str:
    if val is None or math.isnan(val):
        return "집계중"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


def format_telegram_message(
    snap: DecisionSnapshot,
    cfg: StrategyConfig = DEFAULT_CONFIG,
    primary_snap: Optional[DecisionSnapshot] = None,
) -> str:
    """
    날짜, 현재 포지션(V2 원신호 비중), 수익률(YTD / 최근 3달 / 이번달)만 간결하게 생성합니다.
    """
    prev_w = K.code_weight(snap.prev_asset_code)
    next_w = K.code_weight(snap.target_asset_code)
    same_weights = abs(prev_w - next_w) < 1e-9

    pos_str = K.asset_name_alert(snap.target_asset_code)
    if not same_weights:
        prev_str = K.asset_name_alert(snap.prev_asset_code)
        pos_display = f"<b>{pos_str}</b> (변경: {prev_str} → {pos_str})"
    else:
        pos_display = f"<b>{pos_str}</b>"

    if snap.phase == "FINAL" and primary_snap is not None:
        p_str = K.asset_name_alert(primary_snap.target_asset_code)
        pos_display = f"<b>{pos_str}</b> (2차 정정: {p_str} → {pos_str})"

    ytd_str = _fmt_pct(snap.ytd_ret_pct)
    m3_str = _fmt_pct(snap.ret_3m_pct)
    mtd_str = _fmt_pct(snap.mtd_ret_pct)

    lines = [
        f"1. 날짜: <b>{snap.trade_date}</b>",
        f"2. 현재 포지션: {pos_display}",
        "3. 올해 수익률",
        f"  - YTD 수익률: <b>{ytd_str}</b>",
        f"  - 최근 3달 수익률: <b>{m3_str}</b>",
        f"  - 이번달 수익률: <b>{mtd_str}</b>",
    ]
    return "\n".join(lines)
