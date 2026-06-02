"""Telegram bot notification delivery."""

import logging
import json
import asyncio
from typing import Optional

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from .config import Config

logger = logging.getLogger(__name__)


async def send_telegram_alerts(jobs: list[dict], config: Config) -> bool:
    """Send job alerts via Telegram bot.

    Args:
        jobs: List of job dicts from dedup engine (sorted by score)
        config: App configuration
    Returns:
        True if sent successfully
    """
    if not jobs:
        logger.info("No jobs to send via Telegram")
        return True

    if config.dry_run:
        logger.info(f"DRY RUN: Would send {len(jobs)} jobs to Telegram chat {config.telegram_chat_id}")
        return True

    bot = Bot(token=config.telegram_bot_token)
    chat_id = config.telegram_chat_id

    apply_jobs = [j for j in jobs if j["recommendation"] == "APPLY"]
    maybe_jobs = [j for j in jobs if j["recommendation"] == "MAYBE"]

    try:
        # Send summary header
        summary = _build_summary(apply_jobs, maybe_jobs)
        await bot.send_message(
            chat_id=chat_id,
            text=summary,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

        # Send APPLY jobs with full detail
        for job in apply_jobs:
            msg, keyboard = _build_job_message(job, is_apply=True)
            await bot.send_message(
                chat_id=chat_id,
                text=msg,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
                disable_web_page_preview=True,
            )
            await asyncio.sleep(0.5)  # Rate limiting

        # Send MAYBE jobs (compact format)
        if maybe_jobs:
            # Group into batches of 5 to reduce message count
            for i in range(0, len(maybe_jobs), 5):
                batch = maybe_jobs[i:i+5]
                msg = _build_maybe_batch(batch)
                await bot.send_message(
                    chat_id=chat_id,
                    text=msg,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                await asyncio.sleep(0.5)

        logger.info(f"Telegram: sent {len(apply_jobs)} APPLY + {len(maybe_jobs)} MAYBE jobs")
        return True

    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def _build_summary(apply_jobs: list[dict], maybe_jobs: list[dict]) -> str:
    """Build the summary header message."""
    total = len(apply_jobs) + len(maybe_jobs)
    return (
        f"<b>Job Alert</b>\n\n"
        f"Found <b>{total}</b> new jobs matching your profile:\n"
        f"  <b>{len(apply_jobs)}</b> worth applying to\n"
        f"  <b>{len(maybe_jobs)}</b> to consider\n\n"
        f"{'=' * 30}"
    )


def _build_job_message(job: dict, is_apply: bool = True) -> tuple[str, Optional[InlineKeyboardMarkup]]:
    """Build a detailed message for an APPLY job."""
    emoji = "🟢" if is_apply else "🟡"
    rec = job.get("recommendation", "SKIP")
    score = job.get("score", 0)
    url = job.get("url", "")
    domain = job.get("url_domain", "")
    salary = job.get("salary_range") or job.get("salary_estimate") or ""

    key_matches = job.get("key_matches", "[]")
    if isinstance(key_matches, str):
        try:
            key_matches = json.loads(key_matches)
        except Exception:
            key_matches = []

    matches_str = ", ".join(key_matches[:5]) if key_matches else "—"
    salary_line = f"\n💰 {salary}" if salary else ""

    msg = (
        f"{emoji} <b>{_escape(job.get('title', 'Unknown'))}</b>\n"
        f"🏢 {_escape(job.get('company', 'Unknown'))}\n"
        f"📍 {_escape(job.get('location', 'Unknown'))}\n"
        f"🎯 Score: <b>{score}/100</b> | {rec}{salary_line}\n"
        f"📋 Source: {job.get('source', 'unknown')}\n\n"
        f"<i>{_escape(job.get('reasoning', ''))}</i>\n\n"
        f"🔑 Matches: {_escape(matches_str)}\n"
        f"🔗 <code>{domain}</code>"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Apply Now → {domain}", url=url)]
    ]) if url else None

    return msg, keyboard


def _build_maybe_batch(jobs: list[dict]) -> str:
    """Build a compact message for a batch of MAYBE jobs."""
    lines = ["🟡 <b>Worth Considering</b>\n"]
    for job in jobs:
        url = job.get("url", "")
        domain = job.get("url_domain", "")
        score = job.get("score", 0)
        lines.append(
            f"• <a href=\"{url}\">{_escape(job.get('title', 'Unknown'))}</a> @ {_escape(job.get('company', 'Unknown'))}\n"
            f"  📍 {_escape(job.get('location', 'Unknown'))} | 🎯 {score}/100\n"
            f"  <i>{_escape(job.get('reasoning', '')[:100])}</i>\n"
        )
    return "\n".join(lines)


def _escape(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
