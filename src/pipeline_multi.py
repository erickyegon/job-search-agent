"""Multi-user pipeline orchestrator. Replaces single-user main.py for SaaS mode."""

import asyncio
import json
import logging
import os
import sys
import time
import yaml
from datetime import datetime, timezone

from .config import Config
from .database import UserDB, PLAN_LIMITS
from .scraper import scrape_aggregators, CORE_QUERIES
from .scrapers import scrape_ai_boards, scrape_company_career_pages
from .link_validator import validate_links
from .analyzer import JobAnalyzer
from .notifier_email import send_email_digest
from .notifier_telegram import send_telegram_alerts
from .models import Job, Recommendation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_profile_text(profile: dict) -> str:
    """Convert a user's profile dict into YAML text for the analyzer."""
    return yaml.dump(profile, default_flow_style=False)


async def run_multi_user_pipeline(notify_email=False, notify_telegram=False,
                                   hours_back=24):
    """Run the job search pipeline for all active users."""
    start = time.time()
    config = Config.from_env()
    db = UserDB(config.db_path)

    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    logger.info("=" * 60)
    logger.info(f"JobMatch AI Multi-User Pipeline - {now_str}")
    logger.info("=" * 60)

    try:
        # Get all active users
        users = db.get_active_users()
        if not users:
            logger.info("No active users. Exiting.")
            return
        logger.info(f"Active users: {len(users)}")

        # STAGE 1: Scrape jobs ONCE (shared across all users)
        logger.info("STAGE 1: Scraping job sources...")
        aggregator_jobs = scrape_aggregators(hours_back=hours_back)
        ai_board_jobs = await scrape_ai_boards()
        company_jobs = await scrape_company_career_pages()
        all_jobs = aggregator_jobs + ai_board_jobs + company_jobs
        logger.info(f"Total scraped: {len(all_jobs)} jobs")

        if not all_jobs:
            logger.info("No jobs found. Exiting.")
            return

        # STAGE 2: For each user, filter new jobs and analyze
        for user in users:
            user_id = user["id"]
            email = user["email"]
            plan = user["plan"]
            limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
            profile = json.loads(user.get("profile_json", "{}"))

            if not profile:
                logger.info(f"Skipping user {user_id} ({email}): no profile")
                continue

            logger.info(f"--- Processing user {user_id} ({email}) [plan={plan}] ---")

            # Filter out already-seen jobs for this user
            new_jobs = [
                j for j in all_jobs
                if not db.is_job_seen(user_id, j.job_hash)
            ]
            logger.info(f"  New jobs for user: {len(new_jobs)}")

            if not new_jobs:
                continue

            # Limit based on plan
            max_queries = limits["max_search_queries"]
            # Cap jobs to analyze based on plan tier
            max_analyze = limits["max_alerts_per_day"] * 10  # Analyze 10x alert limit
            if len(new_jobs) > max_analyze:
                new_jobs = new_jobs[:max_analyze]

            # Analyze with user's profile
            logger.info(f"  Analyzing {len(new_jobs)} jobs...")
            analyzer = JobAnalyzer(config)
            # Override the profile with user-specific one
            analyzer._profile_text = _build_profile_text(profile)
            analyzer._system = f"{analyzer._system.split('--- CANDIDATE PROFILE ---')[0]}--- CANDIDATE PROFILE ---\n{analyzer._profile_text}"

            analyzed = await analyzer.analyze_batch(new_jobs)

            # Save results for this user
            for job in analyzed:
                db.save_user_job(user_id, job.to_dict())

            clean = [j for j in analyzed if not j.is_suspicious]
            ac = sum(1 for j in clean if j.recommendation == Recommendation.APPLY)
            mc = sum(1 for j in clean if j.recommendation == Recommendation.MAYBE)
            logger.info(f"  Results: {ac} APPLY, {mc} MAYBE")

            # Validate links for APPLY/MAYBE jobs only
            worth_validating = [j for j in clean if j.score >= 60]
            if worth_validating:
                verified = await validate_links(worth_validating)
                for job in verified:
                    db.save_user_job(user_id, job.to_dict())

            # STAGE 3: Notifications
            should_email = notify_email and limits["email_notifications"]
            should_telegram = (notify_telegram and limits["telegram_notifications"]
                               and user.get("telegram_chat_id"))

            if should_email:
                email_jobs = db.get_user_unnotified(user_id, "email", min_score=60)
                # Cap to daily limit
                email_jobs = email_jobs[:limits["max_alerts_per_day"]]
                if email_jobs:
                    cfg_copy = Config.from_env()
                    cfg_copy.recipient_email = user["email"]
                    success = send_email_digest(email_jobs, cfg_copy)
                    if success:
                        hashes = [j["job_hash"] for j in email_jobs]
                        db.mark_user_notified(user_id, hashes, "email")
                        logger.info(f"  Emailed {len(email_jobs)} jobs to {email}")

            if should_telegram:
                tg_jobs = db.get_user_unnotified(user_id, "telegram", min_score=60)
                tg_jobs = tg_jobs[:limits["max_alerts_per_day"]]
                if tg_jobs:
                    cfg_copy = Config.from_env()
                    cfg_copy.telegram_chat_id = user["telegram_chat_id"]
                    success = await send_telegram_alerts(tg_jobs, cfg_copy)
                    if success:
                        hashes = [j["job_hash"] for j in tg_jobs]
                        db.mark_user_notified(user_id, hashes, "telegram")
                        logger.info(f"  Telegrammed {len(tg_jobs)} jobs to {email}")

        # Cleanup
        db.cleanup_old_user_jobs(max_age_days=14)
        elapsed = time.time() - start
        logger.info(f"Multi-user pipeline complete in {elapsed:.1f}s for {len(users)} users")

    finally:
        db.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="JobMatch AI Multi-User Pipeline")
    parser.add_argument("--email", action="store_true")
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    asyncio.run(run_multi_user_pipeline(
        notify_email=args.email,
        notify_telegram=args.telegram,
        hours_back=args.hours,
    ))


if __name__ == "__main__":
    main()
