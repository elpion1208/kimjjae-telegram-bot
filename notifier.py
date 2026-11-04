"""
텔레그램 & 디스코드 메시지 동시 발송 모듈 (Telegram & Discord Notifier)
- 텔레그램: 환경변수 TELEGRAM_BOT_TOKEN 및 TELEGRAM_CHAT_ID 사용 (HTML 포맷)
- 디스코드: 환경변수 DISCORD_WEBHOOK_URL 사용 (HTML 태그를 디스코드 마크다운 **굵게**, __밑줄__, *기울임*으로 자동 변환)
- 둘 다 설정되어 있으면 텔레그램과 디스코드 양쪽으로 동시에 발송하며, 한쪽만 설정되어 있어도 정상 작동합니다.
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


def _html_to_discord_markdown(text: str) -> str:
    """
    텔레그램용 HTML 태그(<b>, <u>, <i>)와 엔티티(&gt;, &lt;, &amp;)를
    디스코드 마크다운(**굵게**, __밑줄__, *기울임*)으로 깔끔하게 변환합니다.
    """
    md = text
    md = re.sub(r"<b>(.*?)</b>", r"**\1**", md, flags=re.DOTALL)
    md = re.sub(r"<u>(.*?)</u>", r"__\1__", md, flags=re.DOTALL)
    md = re.sub(r"<i>(.*?)</i>", r"*\1*", md, flags=re.DOTALL)
    md = re.sub(r"</?(b|u|i)>", "", md)
    md = md.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return md


def _split_discord_chunks(text: str, limit: int = 1950) -> List[str]:
    """디스코드 메시지 1회 글자수 제한(2,000자) 초과 시 줄바꿈 기준으로 안전하게 분할합니다."""
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    current_lines: List[str] = []
    current_len = 0
    for line in text.split("\n"):
        if current_len + len(line) + 1 > limit and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line) + 1
        else:
            current_lines.append(line)
            current_len += len(line) + 1
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def send_discord_message(
    message: str,
    webhook_url: str | None = None,
    dry_run: bool = False,
) -> bool:
    """
    디스코드 웹후크(DISCORD_WEBHOOK_URL)로 메시지를 전송합니다.
    여러 채널에 보내려면 콤마(,)로 구분하여 여러 웹후크 URL을 등록할 수 있습니다.
    """
    raw_webhooks = (webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")).strip()
    if not raw_webhooks:
        return False

    discord_text = _html_to_discord_markdown(message)
    if dry_run:
        logger.info("[DRY-RUN] 디스코드 발송 생략. 변환된 마크다운:\n" + discord_text)
        return True

    webhook_urls: List[str] = [u.strip() for u in raw_webhooks.split(",") if u.strip()]
    chunks = _split_discord_chunks(discord_text)
    success_count = 0

    for idx, url in enumerate(webhook_urls, start=1):
        webhook_ok = True
        for chunk in chunks:
            payload = {"content": chunk}
            try:
                resp = requests.post(url, json=payload, timeout=15)
                if resp.status_code not in (200, 204):
                    logger.error(
                        f"디스코드 웹후크 발송 실패 (#{idx}, status={resp.status_code}): {resp.text}"
                    )
                    webhook_ok = False
                    break
            except Exception as e:
                logger.error(f"디스코드 웹후크 호출 중 예외 발생 (#{idx}): {e}")
                webhook_ok = False
                break
        if webhook_ok:
            logger.info(f"디스코드 웹후크 발송 성공 (#{idx})")
            success_count += 1

    return success_count > 0


def send_telegram_message(
    message: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    discord_webhook: str | None = None,
    dry_run: bool = False,
) -> bool:
    """
    텔레그램 봇 및 디스코드 웹후크로 메시지를 전송합니다.
    - TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID 가 설정되어 있으면 텔레그램 발송
    - DISCORD_WEBHOOK_URL 이 설정되어 있으면 디스코드 발송
    - 둘 다 설정되어 있으면 양쪽 모두 동시 발송!
    """
    token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
    raw_chat_ids = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
    raw_discord = (discord_webhook or os.environ.get("DISCORD_WEBHOOK_URL", "")).strip()

    if dry_run:
        logger.info(
            "[DRY-RUN] 텔레그램/디스코드 발송 생략. 생성된 메시지 전문:\n"
            + "=" * 60
            + "\n"
            + message
            + "\n"
            + "=" * 60
        )
        return True

    has_telegram = bool(token and raw_chat_ids)
    has_discord = bool(raw_discord)

    if not has_telegram and not has_discord:
        logger.error(
            "❌ 텔레그램(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) 또는 디스코드(DISCORD_WEBHOOK_URL) 설정이 모두 비어 있습니다! "
            "GitHub 저장소 -> Settings -> Secrets and variables -> Actions -> Repository secrets에서 등록해 주세요."
        )
        return False

    tg_ok = False
    if has_telegram:
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
                    logger.warning(
                        f"텔레그램 HTML 발송 응답 코드={resp.status_code} ({resp.text}) -> 일반 텍스트 폴백 재시도..."
                    )
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

        tg_ok = success_count > 0

    dc_ok = False
    if has_discord:
        dc_ok = send_discord_message(message, webhook_url=raw_discord, dry_run=dry_run)

    return tg_ok or dc_ok
