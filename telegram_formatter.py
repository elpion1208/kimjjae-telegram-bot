"""
변형티큐법 텔레그램 알림 메시지 포맷터 (Telegram Formatter)
원본 Pine Script v6의 HTML 태그(<b>, <u>), 포지션 섹션(alertPositionSection),
D7 단기채/현금 배분, ER10/20/59, 익일 QQQ 기준가, T3 금리·달러 요약을 구성합니다.
"""

from __future__ import annotations
from datetime import datetime
import math
from typing import Optional
import pytz

from config import StrategyConfig, DEFAULT_CONFIG
from strategy_engine import DecisionSnapshot
import kernel as K

KST_TZ = pytz.timezone("Asia/Seoul")


def _fmt_pct(val: Optional[float]) -> str:
    if val is None or math.isnan(val):
        return "미확인"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


def _fmt_pp(val: Optional[float]) -> str:
    if val is None or math.isnan(val):
        return "미확인"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%p"


def _format_position_section(
    prev_code: int,
    prev_role: int,
    next_code: int,
    next_role: int,
    snap: DecisionSnapshot,
    cfg: StrategyConfig,
) -> str:
    """
    Pine Script alertPositionSection 동일 구현:
    - 비중 유지 시: 🔒 <b>현재 포지션</b>
    - 비중 변경 시: ⛔ <b>이전 포지션</b> -> ✅ <b>목표 포지션</b>
    """
    prev_w = K.code_weight(prev_code)
    next_w = K.code_weight(next_code)
    same_weights = abs(prev_w - next_w) < 1e-9 and (
        not cfg.d7_enabled or abs(snap.before_bill - snap.target_bill) < 1e-9
    )

    prev_icon = K.alert_pos_icon(prev_code, prev_role)
    next_icon = K.alert_pos_icon(next_code, next_role)

    if cfg.d7_enabled:
        prev_text = K.d7_breakdown(
            snap.before_tq, snap.before_bill, snap.before_cash, prev_icon, True
        )
        next_raw = K.d7_breakdown(
            snap.target_tq, snap.target_bill, snap.target_cash, next_icon, True
        )
    else:
        prev_text = f"{K.asset_name_alert(prev_code)} {prev_icon}"
        next_raw = f"{K.asset_name_alert(next_code)} {next_icon}"

    next_text = K.alert_underline_position(next_raw)

    if same_weights:
        return f"🔒 <b>현재 포지션</b>\n{next_text}"
    return f"⛔ <b>이전 포지션</b>\n{prev_text}\n✅ <b>목표 포지션</b>\n{next_text}"


def _format_vix_line(snap: DecisionSnapshot, cfg: StrategyConfig) -> str:
    if not cfg.use_vix_filter:
        val_str = f"{snap.vix_c:.2f}" if snap.vix_c is not None else "미확인"
        return f"• VIX: {val_str} (필터 OFF)"
    if snap.vix_c is None:
        return "• VIX: 미확인"

    chg_str = ""
    if snap.vix_p is not None and snap.vix_p > 0:
        chg_pct = ((snap.vix_c / snap.vix_p) - 1.0) * 100.0
        chg_str = f" ({_fmt_pct(chg_pct)})"

    blocked = (cfg.use_vix_force_sell and snap.vix_risk_lock) or (snap.vix_c >= cfg.vix_sell_threshold)
    cnt = max(snap.vix_below_cnt, 0)
    req = cfg.vix_buy_support_days

    if blocked:
        status = "🔴 위험(강제현금)" if cfg.use_vix_force_sell else "🟠 위험구간"
    elif snap.vix_c <= cfg.vix_rebuy_threshold:
        if cnt >= req:
            status = f"🟢 매수보조 확인완료({min(cnt, req)}/{req})"
        else:
            status = f"🟡 매수보조 대기({cnt}/{req})"
    else:
        status = f"🟠 재매수 기준({cfg.vix_rebuy_threshold:.0f}) 상회 (0/{req})"

    return f"• VIX: <b>{snap.vix_c:.2f}</b>{chg_str} | {status}"


def _format_dgs_line(
    label: str,
    curr: Optional[float],
    chg: Optional[float],
    period: int,
    thr: float,
    risk: bool,
) -> str:
    if curr is None or chg is None:
        return f"  - {label}({period}일): 미확인"
    badge = "🔴위험" if risk else "🟢정상"
    return f"  - {label}({period}일): {curr:.2f}% ({_fmt_pp(chg)} / 기준 +{_fmt_pp(thr).lstrip('+')}) {badge}"


def _format_next_qqq_line(snap: DecisionSnapshot, cfg: StrategyConfig) -> str:
    """주석 17절: 익일 QQQ T3 약세/해제 기준가 표시"""
    macro_risk_active = snap.t3_short_risk or snap.t3_long_risk or (
        (snap.dgs2_risk and snap.dgs6m_risk)
        or (snap.dxy_risk and (snap.us10y_risk or snap.dgs20_risk))
    )
    if not cfg.use_t3_d80_filter or not cfg.t3_use_risk_zero:
        return ""

    holding = K.code_weight(snap.target_asset_code) > 1e-9
    if snap.next_qqq_buy_price is None or snap.next_qqq_sell_price is None:
        return "\n• 익일 QQQ T3 기준가: 산출 대기"

    if not snap.other_filters_permit_hold:
        return "\n• 익일 QQQ T3 기준가: 선행 필터 대기(비T3 제한 중)"

    if holding:
        diff_pct = ((snap.next_qqq_sell_price / snap.q_c) - 1.0) * 100.0
        macro_note = "거시위험 활성" if macro_risk_active else "거시위험 발생 시"
        return (
            f"\n• 익일 QQQ T3 매도(약세) 기준: <b>${snap.next_qqq_sell_price:.2f} 이하</b> "
            f"({_fmt_pct(diff_pct)}, {macro_note})"
        )
    else:
        diff_pct = ((snap.next_qqq_buy_price / snap.q_c) - 1.0) * 100.0
        return (
            f"\n• 익일 QQQ T3 매수(해제) 기준: <b>${snap.next_qqq_buy_price:.2f} 이상</b> "
            f"({_fmt_pct(diff_pct)})"
        )


def format_telegram_message(
    snap: DecisionSnapshot,
    cfg: StrategyConfig = DEFAULT_CONFIG,
    primary_snap: Optional[DecisionSnapshot] = None,
) -> str:
    """
    1차(PRIMARY) 정기 알림 또는 2차(FINAL) 정정 알림용 텔레그램 HTML 전문을 생성합니다.
    """
    now_kst = datetime.now(KST_TZ).strftime("%Y-%m-%d %H:%M KST")

    if snap.phase == "FINAL":
        header = (
            f"🚨 <b>[변형티큐법] 2차(FINAL) 확정 신호 변경</b>\n"
            f"📅 미국 거래일: <b>{snap.trade_date}</b> (발송: {now_kst})"
        )
    else:
        header = (
            f"📊 <b>[변형티큐법] 1차(PRIMARY) 마감 판정</b>\n"
            f"📅 미국 거래일: <b>{snap.trade_date}</b> (발송: {now_kst})"
        )

    # 포지션 섹션 (1차는 전일 대비, 2차 변경 알림도 당일 시작 기준 전일 대비 + 1차 대비 병기)
    pos_section = _format_position_section(
        snap.prev_asset_code,
        snap.prev_o4_role,
        snap.target_asset_code,
        snap.target_o4_role,
        snap,
        cfg,
    )

    final_diff_note = ""
    if snap.phase == "FINAL" and primary_snap is not None:
        p_icon = K.alert_pos_icon(primary_snap.target_asset_code, primary_snap.target_o4_role)
        f_icon = K.alert_pos_icon(snap.target_asset_code, snap.target_o4_role)
        p_str = K.d7_breakdown(
            primary_snap.target_tq, primary_snap.target_bill, primary_snap.target_cash, p_icon, True
        )
        f_str = K.d7_breakdown(
            snap.target_tq, snap.target_bill, snap.target_cash, f_icon, True
        )
        final_diff_note = f"\n🔄 <b>1차 판정 대비 변경</b>: {p_str} ➡️ <b>{f_str}</b>"

    # D7 안내 노트
    d7_note_lines = ["<i>※ 목표비중 기준 · 실제 체결 확인 아님</i>"]
    if cfg.d7_enabled and not snap.d7_history_complete:
        d7_note_lines.append("<i>※ D7: 초기 8거래일 이력 누적 중</i>")
    elif cfg.d7_enabled and not cfg.d7_same_session:
        d7_note_lines.append("<i>※ 단기채 신규 편입 보류: 당일 재사용 불가</i>")

    # 판정 사유
    tag_suffix = f" [{snap.tag_text}]" if snap.tag_text else ""
    reason_section = f"📌 <b>판정 사유</b>: <b>{snap.reason_text}</b>{tag_suffix}"

    # 1. TQQQ 핵심 지표 섹션
    t_trend_str = "상향(🟢)" if snap.above200_hysteresis else "하향(⚪)"
    slope_str = f"{snap.dist_slope:+.4f}" if snap.dist_slope is not None else "미확인"
    slope_boost_badge = " (부스트 ON)" if snap.slope_ok_flag else ""
    vol_badge = "🔴락" if snap.locked else "🟢정상"
    oh_str = f"{snap.overheat_stage}단" if snap.overheat_stage > 0 else "0단(정상)"
    if snap.oh4_reentry_guard:
        oh_str += f" [4단해제 가드 {snap.oh4_slope_ok_cnt}/{cfg.oh4_release_slope_days}]"

    er20_line = ""
    if snap.overheat_stage in (1, 2):
        er20_val_str = f"{snap.t_er20:+.3f}" if snap.t_er20 is not None else "대기"
        er20_line = f"\n• TQQQ 20일 ER: <b>{er20_val_str}</b> (과열 1·2단 상한해제 기준 &gt; +0.300)"

    stop_line = ""
    if snap.principal_stop_price is not None:
        stop_dist_pct = ((snap.principal_stop_price / snap.t_c) - 1.0) * 100.0
        stop_line = (
            f"\n• 원금보호 로스컷: <b>${snap.principal_stop_price:.2f}</b> "
            f"({_fmt_pct(stop_dist_pct)} / 진입기준가 대비 {cfg.principal_stop_pct*100:.2f}%)"
        )
    elif snap.reentry_lock:
        stop_line = f"\n• 손절 후 재진입 잠금: 🔴 활성 (QQQ RSI ≥ {cfg.rsi_reentry_thr:.0f} 필요)"

    tqqq_section = (
        f"📈 <b>[TQQQ &amp; 과열/방어 지표]</b>\n"
        f"• TQQQ 종가: <b>${snap.t_c:.2f}</b> ({_fmt_pct(snap.t_ret_pct)}) | 200일선: ${snap.t_ma200:.2f}\n"
        f"• 200일 이격도: <b>{snap.dist200:.2f}%</b> ({t_trend_str})\n"
        f"• 45일 기울기: {slope_str}{slope_boost_badge} | 20일 변동성: {snap.t_v20:.4f} ({vol_badge})\n"
        f"• 과열 단계: <b>{oh_str}</b>"
        f"{er20_line}"
        f"{stop_line}"
    )

    # 2. QQQ & SPY & VIX 섹션
    q_bull_str = "정배열(SMA3&gt;161 🟢)" if snap.q_bull else "역배열(SMA3≤161 ⚪)"
    rsi_str = f"{snap.qqq_rsi14:.2f}" if snap.qqq_rsi14 is not None else "미확인"
    er10_str = f"{snap.qqq_er10:+.3f}" if snap.qqq_er10 is not None else "대기"
    er59_line = ""
    if snap.overheat_stage == 4:
        er59_str = f"{snap.qqq_er59:+.3f}" if snap.qqq_er59 is not None else "대기"
        er59_line = f"\n• QQQ 59일 ER(O4P): <b>{er59_str}</b> (기준 ≥ +0.050)"

    next_qqq_line = _format_next_qqq_line(snap, cfg)
    spy_status = "🟢 상승추세" if snap.spy_bull else "🔴 하락추세(현금제한)"
    vix_line = _format_vix_line(snap, cfg)

    market_section = (
        f"🧭 <b>[QQQ · SPY · VIX 필터]</b>\n"
        f"• QQQ 종가: <b>${snap.q_c:.2f}</b> ({_fmt_pct(snap.q_ret1_pct)}) | 최근 3봉: {snap.qqq3_emoji}\n"
        f"• QQQ 추세: {q_bull_str} | RSI(14): {rsi_str}\n"
        f"• QQQ 10일 ER(T3): <b>{er10_str}</b> (약세 기준 ≤ -0.300)"
        f"{er59_line}"
        f"{next_qqq_line}\n"
        f"• SPY 200일 이격도: <b>{snap.spy_dist200:.2f}%</b> ({spy_status})\n"
        f"{vix_line}"
    )

    # 3. T3 거시위험 (금리 & 달러) 섹션
    weak_str = "🔴 약세(SMA3하회 또는 ER10≤-0.30)" if snap.t3_m09_weak else "🟢 강세(정상)"
    macro_branch_str = K.t3_risk_branch_status_text(snap.t3_short_risk, snap.t3_long_risk)
    dxy_line = (
        f"  - DXY(120일, {snap.dxy_source_label}): {snap.dxy_c:.2f} "
        f"({_fmt_pct(snap.dxy_roc)} / 기준 &gt; 0.00%) {'🔴위험' if snap.dxy_risk else '🟢정상'}"
        if (snap.dxy_c is not None and snap.dxy_roc is not None)
        else "  - DXY(120일): 미확인"
    )
    obs_note = f" (FRED 관측일: {snap.dgs_obs_date})" if snap.dgs_obs_date else ""

    t3_section = (
        f"🏦 <b>[T3 거시위험 관리{obs_note}]</b>\n"
        f"• QQQ 단기상태: {weak_str} | 거시조합: <b>{macro_branch_str}</b>\n"
        f"{dxy_line}\n"
        f"{_format_dgs_line('DGS02Y', snap.dgs2_c, snap.dgs2_chg, cfg.t3_dgs2_chg_len, cfg.t3_dgs2_chg_thr, snap.dgs2_risk)}\n"
        f"{_format_dgs_line('DGS06M', snap.dgs6m_c, snap.dgs6m_chg, cfg.t3_dgs6m_chg_len, cfg.t3_dgs6m_chg_thr, snap.dgs6m_risk)}\n"
        f"{_format_dgs_line('DGS10Y', snap.us10y_c, snap.us10y_chg, cfg.t3_us10y_chg_len, cfg.t3_us10y_chg_thr, snap.us10y_risk)}\n"
        f"{_format_dgs_line('DGS20Y', snap.dgs20_c, snap.dgs20_chg, cfg.t3_dgs20_chg_len, cfg.t3_dgs20_chg_thr, snap.dgs20_risk)}"
    )

    # 4. 원/달러 환율 및 경계 근접 재확인 문구
    footer_lines = []
    if snap.usdkrw_c is not None:
        krw_chg_str = ""
        if snap.usdkrw_p is not None and snap.usdkrw_p > 0:
            krw_chg = snap.usdkrw_c - snap.usdkrw_p
            krw_pct = ((snap.usdkrw_c / snap.usdkrw_p) - 1.0) * 100.0
            krw_chg_str = f" ({krw_chg:+.1f}원, {_fmt_pct(krw_pct)})"
        footer_lines.append(f"💱 <b>원/달러 환율</b>: {snap.usdkrw_c:,.1f}원{krw_chg_str}")

    if snap.border_warnings:
        warn_items = "\n".join(f"  ⚠️ {w}" for w in snap.border_warnings)
        footer_lines.append(
            f"\n🔎 <b>[경계 근접 재확인 안내]</b>\n{warn_items}\n"
            f"<i>※ 데이터 공급자 간 미세 오차 가능 구간이므로 트레이딩뷰 원천 차트와 교차 확인을 권장합니다.</i>"
        )

    sections = [
        header,
        f"{pos_section}{final_diff_note}\n" + "\n".join(d7_note_lines),
        reason_section,
        tqqq_section,
        market_section,
        t3_section,
    ]
    if footer_lines:
        sections.append("\n".join(footer_lines))

    return "\n\n".join(sections)
