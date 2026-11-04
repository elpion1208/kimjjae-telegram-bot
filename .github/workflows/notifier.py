"""
텔레그램 봇 메시지 발송 모듈 (Telegram Notifier)
환경변수 TELEGRAM_BOT_TOKEN 및 TELEGRAM_CHAT_ID를 사용하여 HTML 포맷 메시지를 전송합니다.
"""

from __future__ import annotations
import logging
import os
from typing import List
import requests

logger = logging.getLogger(__name__)


def send_telegram_message(
    message: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    dry_run: bool = False,
) -> bool:
    """
    텔레그램 봇으로 메시지를 전송합니다.
    여러 채팅방에 보내려면 TELEGRAM_CHAT_ID에 콤마(,)로 구분하여 입력할 수 있습니다.
    """
    token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
    raw_chat_ids = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()

    if dry_run:
        logger.info("[DRY-RUN] 텔레그램 발송 생략. 생성된 메시지 전문:\n" + "=" * 60 + "\n" + message + "\n" + "=" * 60)
        return True

    if not token or not raw_chat_ids:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID 환경변수가 설정되지 않았습니다. "
            "GitHub Secrets 또는 환경변수를 확인해 주세요."
        )

    chat_ids: List[str] = [cid.strip() for cid in raw_chat_ids.split(",") if cid.strip()]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    all_ok = True

    for cid in chat_ids:
        payload = {
            "chat_id": cid,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                logger.error(f"텔레그램 발송 실패 (chat_id={cid}, status={resp.status_code}): {resp.text}")
                all_ok = False
            else:
                logger.info(f"텔레그램 발송 성공 (chat_id={cid})")
        except Exception as e:
            logger.error(f"텔레그램 API 호출 중 예외 발생 (chat_id={cid}): {e}")
            all_ok = False

    return all_ok
