"""
변형티큐법 텔레그램 봇 메인 실행 스크립트 (main.py)
- 1차(PRIMARY) 판정: 미국 정규장 마감 직후 실행 -> 당일 1차 알림 발송 및 pre_day_state(c084DayStart) 저장
- 2차(FINAL) 정정 판정: 마감 15분~90분 후 실행 -> 최신 확정 VIX/금리/DXY로 재판정 후 목표비중 변경 시에만 2차 알림 발송
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional
import pytz

from config import DEFAULT_CONFIG
from data_fetcher import fetch_all_market_data, ET_TZ
from strategy_engine import DayState, DecisionSnapshot, StrategyEngine
from telegram_formatter import format_telegram_message
from notifier import send_telegram_message
import kernel as K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STATE_FILE_PATH = Path(__file__).resolve().parent / "state.json"


def load_persisted_state(path: Path = STATE_FILE_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"기존 state.json 로드 실패 (전체 이력 재생으로 복구합니다): {e}")
        return {}


def save_persisted_state(data: Dict[str, Any], path: Path = STATE_FILE_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"상태 파일 저장 완료: {path}")


def _is_us_early_close_day(d: date) -> bool:
    """
    미국 증시 정규장 조기마감일(13:00 ET 마감) 여부 판정:
    1) 독립기념일 전일 (7월 3일, 단 주말이 아닐 때)
    2) 추수감사절 다음날 (11월 넷째 주 목요일의 다음 날 = 블랙프라이데이)
    3) 크리스마스 이브 (12월 24일, 단 주말이 아닐 때)
    """
    if d.weekday() >= 5:
        return False
    if d.month == 7 and d.day == 3:
        return True
    if d.month == 12 and d.day == 24:
        return True
    if d.month == 11 and d.weekday() == 4:  # 금요일
        # 전날(목요일)이 11월 넷째 목요일(22~28일)인지 확인
        thursday_day = d.day - 1
        if 22 <= thursday_day <= 28:
            return True
    return False


def run_bot(
    mode: str = "auto",
    dry_run: bool = False,
    force_send: bool = False,
    state_path: Path = STATE_FILE_PATH,
) -> int:
    cfg = DEFAULT_CONFIG
    now_et = datetime.now(ET_TZ)
    today_et_str = now_et.strftime("%Y-%m-%d")
    close_hour_et = 13 if _is_us_early_close_day(now_et.date()) else 16

    # 정규장 진행 중(예: 동절기 EST 기간에 서머타임용 20:xx UTC 스케줄이 먼저 실행된 경우) 조기 종료 가드
    if mode.lower() != "test" and not force_send:
        if now_et.weekday() < 5:
            market_open_minutes = 9 * 60 + 30
            market_close_grace_minutes = close_hour_et * 60 + 1
            now_minutes = now_et.hour * 60 + now_et.minute
            if market_open_minutes <= now_minutes < market_close_grace_minutes:
                logger.info(
                    f"현재 미국 동부시간({now_et.strftime('%Y-%m-%d %H:%M ET')})은 "
                    f"정규장 마감({close_hour_et}:01 ET) 이전입니다. 장 마감 후 다음 스케줄에서 실행됩니다."
                )
                return 0

    bundle = fetch_all_market_data(cfg)
    engine = StrategyEngine(bundle, cfg)

    if not engine.valid_dates:
        logger.error("유효한 거래일 데이터가 없습니다.")
        return 1

    latest_trade_date = engine.valid_dates[-1]
    persisted = load_persisted_state(state_path)
    saved_trade_date = persisted.get("trade_date"
    if saved_trade_date and latest_trade_date < saved_trade_date and mode.lower() != "test" and not force_send:
        logger.warning(
            f"수집된 최신 거래일({latest_trade_date})이 저장된 거래일({saved_trade_date})보다 과거입니다. "
            "외부 데이터 일시 누락으로 판단하여 실행을 건너뜁니다."
        )
        return 0)
    primary_sent_today = (saved_trade_date == latest_trade_date) and bool(
        persisted.get("primary_sent", False)
    )

    # 전체 과거 일봉을 재생하여 최신 거래일 직전 상태(pre_day_state = c084DayStart) 확보
    replayed_pre_day_state, _, _ = engine.replay_up_to(latest_trade_date)

    # 만약 같은 날 1차에서 저장해둔 pre_day_state가 있으면 그것을 우선 사용 (C084 불변성 보장)
    if saved_trade_date == latest_trade_date and "pre_day_state" in persisted:
        pre_day_state = DayState.from_dict(persisted["pre_day_state"])
    else:
        pre_day_state = replayed_pre_day_state

    # 실행 모드 결정:
    # - auto일 경우:
    #   1) state.json에 당일 1차 발송 이력(primary_sent_today)이 있거나,
    #   2) 미국 정규장 마감 후 18분 이상 경과(예: 16:18 ET 이후 도는 2차 스케줄 또는 서머타임 중 21:xx UTC 스케줄)했으면
    #      무조건 'final'(2차 변경 감시) 모드로 동작하여 1차 리포트가 중복 발송되는 것을 원천 차단합니다.
    now_minutes = now_et.hour * 60 + now_et.minute
    close_minutes = close_hour_et * 60
    minutes_since_close = now_minutes - close_minutes
    # ※ GitHub 스케줄은 수십 분~수 시간 지연될 수 있으므로, 시간 기준이 아니라
    #    state.json의 당일 1차 발송 기록만으로 1차/2차를 판별합니다. (지연되어도 1차는 반드시 1번 발송)
    effective_mode = mode.lower()
    if effective_mode == "auto":
        effective_mode = "final" if primary_sent_today else "primary"

    logger.info(
        f"판정 시작: 미국 거래일={latest_trade_date}, 현재 ET={now_et.strftime('%H:%M')} (마감+{minutes_since_close}분), "
        f"요청 모드={mode} -> 실행 모드={effective_mode.upper()}, 당일 1차 발송 기록={primary_sent_today}"
    )

    # ------------------------------------------------------------
    # 1. TEST 모드 (언제든 즉시 판정 및 텔레그램 테스트 발송)
    # ------------------------------------------------------------
    if effective_mode == "test":
        post_state, snap = engine.step_one_day(
            pre_day_state, latest_trade_date, phase="PRIMARY", is_latest_day=True
        )
        msg = format_telegram_message(snap, cfg)
        msg = "🧪 <b>[테스트 발송]</b>\n" + msg
        ok = send_telegram_message(msg, dry_run=dry_run)
        return 0 if ok else 1

    # ------------------------------------------------------------
    # 2. 1차 (PRIMARY) 판정 모드
    # ------------------------------------------------------------
    if effective_mode == "primary":
        if primary_sent_today and not force_send:
            logger.info(
                f"{latest_trade_date} 거래일의 1차(PRIMARY) 알림은 이미 발송되었습니다. "
                f"2차(FINAL) 정정 감시로 전환합니다."
            )
            effective_mode = "final"
        else:
            post_state, primary_snap = engine.step_one_day(
                pre_day_state, latest_trade_date, phase="PRIMARY", is_latest_day=True
            )
            msg = format_telegram_message(primary_snap, cfg)
            ok = send_telegram_message(msg, dry_run=dry_run)
            if not ok:
                return 1

            new_persisted = {
                "trade_date": latest_trade_date,
                "primary_sent": True,
                "final_sent_count": 0,
                "last_notified_target_code": primary_snap.target_asset_code,
                "last_notified_target_role": primary_snap.target_o4_role,
                "last_notified_bill": primary_snap.target_bill,
                "pre_day_state": pre_day_state.to_dict(),
                "post_day_state": post_state.to_dict(),
                "primary_snapshot": primary_snap.to_dict(),
            }
            save_persisted_state(new_persisted, state_path)
            return 0

    # ------------------------------------------------------------
    # 3. 2차 (FINAL) 확정 신호 변경 감시 모드
    # ------------------------------------------------------------
    if effective_mode == "final":
        # 2차 모드에서는 절대 1차 전체 리포트를 재발송하지 않으며,
        # 당일 1차 판정(또는 이전 통지 비중) 대비 목표비중이 실제로 바뀐 경우에만 간결한 포지션 변경 알림을 보냅니다.
        _, calc_primary_snap = engine.step_one_day(
            pre_day_state, latest_trade_date, phase="PRIMARY", is_latest_day=False
        )

        # 당일 시작 상태(pre_day_state = c084DayStart)에서 최신 확정 VIX·금리·DXY로 재판정
        final_post_state, final_snap = engine.step_one_day(
            pre_day_state, latest_trade_date, phase="FINAL", is_latest_day=True
        )

        primary_snap_obj: Optional[DecisionSnapshot] = calc_primary_snap
        if saved_trade_date == latest_trade_date and "primary_snapshot" in persisted and isinstance(persisted["primary_snapshot"], dict):
            try:
                primary_snap_obj = DecisionSnapshot(**persisted["primary_snapshot"])
            except Exception:
                primary_snap_obj = calc_primary_snap

        if saved_trade_date == latest_trade_date and "last_notified_target_code" in persisted:
            last_code = int(persisted["last_notified_target_code"])
            last_bill = float(persisted.get("last_notified_bill", primary_snap_obj.target_bill))
        else:
            last_code = primary_snap_obj.target_asset_code
            last_bill = primary_snap_obj.target_bill

        weight_changed = abs(K.code_weight(final_snap.target_asset_code) - K.code_weight(last_code)) > 1e-9
        d7_changed = cfg.d7_enabled and (abs(final_snap.target_bill - last_bill) >= 0.005 - 1e-9)
        target_changed = weight_changed or d7_changed

        if target_changed or force_send:
            logger.info(
                f"🚨 2차(FINAL) 확정 목표비중 변경 감지! "
                f"(이전 통지 코드={last_code}[{K.code_pct_str(last_code)}%] -> "
                f"신규 확정 코드={final_snap.target_asset_code}[{K.code_pct_str(final_snap.target_asset_code)}%])"
            )
            msg = format_telegram_message(final_snap, cfg, primary_snap=primary_snap_obj)
            ok = send_telegram_message(msg, dry_run=dry_run)
            if not ok:
                return 1

            persisted["trade_date"] = latest_trade_date
            persisted["primary_sent"] = True
            persisted["final_sent_count"] = int(persisted.get("final_sent_count", 0)) + 1
            persisted["last_notified_target_code"] = final_snap.target_asset_code
            persisted["last_notified_target_role"] = final_snap.target_o4_role
            persisted["last_notified_bill"] = final_snap.target_bill
            persisted["pre_day_state"] = pre_day_state.to_dict()
            persisted["post_day_state"] = final_post_state.to_dict()
            persisted["primary_snapshot"] = primary_snap_obj.to_dict()
            persisted["final_snapshot"] = final_snap.to_dict()
            save_persisted_state(persisted, state_path)
            return 0
        else:
            logger.info(
                f"✅ 2차(FINAL) 점검 완료 ({latest_trade_date}): "
                f"목표비중 변동 없음 (TQ {K.code_pct_str(final_snap.target_asset_code)}%, "
                f"단기채 {K.d7_pct_str(final_snap.target_bill)}%). "
                f"원본 규칙에 따라 중복 알림을 발송하지 않습니다."
            )
            # 상태 파일의 최신 확정 스냅샷만 조용히 갱신
            persisted["trade_date"] = latest_trade_date
            persisted["primary_sent"] = True
            persisted["last_notified_target_code"] = last_code
            persisted["last_notified_bill"] = last_bill
            persisted["pre_day_state"] = pre_day_state.to_dict()
            persisted["post_day_state"] = final_post_state.to_dict()
            persisted["primary_snapshot"] = primary_snap_obj.to_dict()
            persisted["final_snapshot"] = final_snap.to_dict()
            save_persisted_state(persisted, state_path)
            return 0

    logger.error(f"지원하지 않는 실행 모드입니다: {mode}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="변형티큐법 텔레그램 알림 봇")
    parser.add_argument(
        "--mode",
        choices=["auto", "primary", "final", "test"],
        default="auto",
        help="실행 모드: auto(자동 판별), primary(1차 마감판정), final(2차 변경감시), test(즉시 테스트)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 텔레그램 발송 없이 콘솔에 생성된 메시지만 출력",
    )
    parser.add_argument(
        "--force-send",
        action="store_true",
        help="중복 방지 또는 무변동 조건을 무시하고 강제 발송",
    )
    args = parser.parse_args()
    sys.exit(run_bot(mode=args.mode, dry_run=args.dry_run, force_send=args.force_send))


if __name__ == "__main__":
    main()
