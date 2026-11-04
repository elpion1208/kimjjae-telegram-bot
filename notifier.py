"""
텔레그램 봇 메시지 발송 모듈 (Telegram Notifier)
환경변수 TELEGRAM_BOT_TOKEN 및 TELEGRAM_CHAT_ID를 사용하여 HTML 포맷 메시지를 전송합니다.
"""

from __future__ import annotations
import logging
import os
import re
from typing import List
import requests

logger = logging.getLogger(__name__)


def _strip_html_tags(text: str) -> str:
    cleaned = re.sub(r"</?(b|u|i)>", "", text)
    cleaned = cleaned.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return cleaned


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
        logger.error(
            "❌ TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 비어 있습니다! "
            "GitHub 저장소 -> Settings -> Secrets and variables -> Actions -> Repository secrets에 "
            "TELEGRAM_BOT_TOKEN 과 TELEGRAM_CHAT_ID 두 개가 정확히 등록되어 있는지 확인해 주세요."
        )
        return False

    chat_ids: List[str] = [cid.strip() for cid in raw_chat_ids.split(",") if cid.strip()]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    success_count = 0

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
                logger.warning(f"텔레그램 HTML 발송 응답 코드={resp.status_code} ({resp.text}) -> 일반 텍스트 폴백 재시도...")
                fallback_payload = {
                    "chat_id": cid,
                    "text": _strip_html_tags(message),
                    "disable_web_page_preview": True,
                }
                resp2 = requests.post(url, json=fallback_payload, timeout=15)
                if resp2.status_code != 200:
                    logger.error(
                        f"텔레그램 발송 실패 (chat_id={cid}, status={resp2.status_code}): {resp2.text} "
                        f"-> 수신자가 봇 대화방에서 [시작(/start)] 버튼을 눌렀는지, 숫자 ID가 맞는지 확인해 주세요."
                    )
                else:
                    logger.info(f"텔레그램 일반 텍스트 폴백 발송 성공 (chat_id={cid})")
                    success_count += 1
            else:
                logger.info(f"텔레그램 발송 성공 (chat_id={cid})")
                success_count += 1
        except Exception as e:
            logger.error(f"텔레그램 API 호출 중 예외 발생 (chat_id={cid}): {e}")

    return success_count > 0
