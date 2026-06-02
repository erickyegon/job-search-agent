"""Pipeline orchestrator — runs the full scrape → analyze → deliver pipeline."""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone

from .config import Config
from .scraper import scrape_aggregators
from .scrapers import scrape_ai_boards, scrape_company_career_pages
from .link_validator import validate_links
from .dedup import DedupEngine
from .analyzer import JobAnalyzer
from .notifier_email import send_email_digest
from .notifier_telegram import send_telegram_alerts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_pipeline(
    notify_email: bool = False,
    notify_telegram: bool = False,
    hours_back: int = 24,
):
    """Run the full job search pipeline.

    Args:
        notify_email: Whether to send email digest this run
        notify_telegram: Whether to send Telegram alerts this run
        hours_back: How far back to look for jobs (hours)
    """
    start = time.time()
    config = Config.from_env()
    config.notify_email = notify_email
    config.notify_telegram = notify_telegram

    # Validate config
    errors = config.validate()
    if errors:
        for err in errors:
            logger.error(f"Config error: {err}")
        if not config.dry_run:
            sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Job Search Agent — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"Mode: {'DRY RUN' if config.dry_run else 'LIVE'}")
    logger.info(f"Notifications: email={'yes' if notify_email else 'no'}, telegram={'yes' if notify_telegram else 'no'}")
    logger.info(f"Lookback: {hours_back} hours")
    logger.info("=" * 60)

    dedup = DedupEngine(config.db_path)

    try:
        # ── STAGE 1: SCRAPE ──
        logger.info("STAGE 1: Scraping job sources...")

        # Run all scrapers concurrently
        aggregator_jobs = scrape_aggregators(hours_back=hours_back)

        ai_board_jobs = await scrape_ai_boards()
        company_jobs = await scrape_company_career_pages()

        all_jobs = aggregator_jobs + ai_board_jobs + company_jobs
        logger.info(f"Total scraped: {len(all_jobs)} jobs "
                    f"(aggregators: {len(aggregator_jobs)}, "
                    f"AI boards: {len(ai_board_jobs)}, "
                    f"company pages: {len(company_jobs)})")

        if not all_jobs:
            logger.info("No jobs found. Exiting.")
            return

        # ── STAGE 2: DEDUP ──
        logger.info("STAGE 2: Deduplicating...")
        new_jobs = dedup.filter_new(all_jobs)

        if not new_jobs:
            logger.info("No new jobs since last run. Checking for unnotified jobs...")
        else:
            logger.info(f"New jobs to analyze: {len(new_jobs)}")

            # ── STAGE 3: VALIDATE LINKS ──
            logger.info("STAGE 3: Validating links...")
            verified_jobs = await validate_links(new_jobs)
            logger.info(f"Jobs with verified links: {len(verified_jobs)}")

            if not verified_jobs:
                logger.info("No jobs passed link validation.")
            else:
                # ── STAGE 4: ANALYZE ──
                logger.info("STAGE 4: Analyzing with Claude...")
                analyzer = JobAnalyzer(config)
                analyzed_jobs = analyzer.analyze_batch(verified_jobs)

                # Filter out suspicious jobs
                clean_jobs = [j for j in analyzed_jobs if not j.is_suspicious]
                suspicious_count = len(analyzed_jobs) - len(clean_jobs)
                if suspicious_count:
                    logger.info(f"Filtered {suspicious_count} suspicious jobs")

                # Save to database
                dedup.save_jobs(clean_jobs)

                # Log summary
                apply_count = sum(1 for j in clean_jobs if j.recommendation.value == "APPLY")
                maybe_count = sum(1 for j in clean_jobs if j.recommendation.value == "MAYBE")
                skip_count = sum(1 for j in clean_jobs if j.recommendation.value == "SKIP")
                logger.info(f"Analysis results: {apply_count} APPLY, {maybe_count} MAYBE, {skip_count} SKIP")

        # ── STAGE 5: DELIVER ──
        if notify_email:
            logger.info("STAGE 5a: Sending email digest...")
            email_jobs = dedup.get_unnotified("email", min_score=60)
            if email_jobs:
                success = send_email_digest(email_jobs, config)
                if success:
                    # Mark as notified using job hashes
                    from .models import Job
                    dedup.mark_notified(
                        [type('obj', (object,), {'job_hash': j['job_hash']})() for j in email_jobs],
                        "email"
                    )
            else:
                logger.info("No unnotified jobs for email")

        if notify_telegram:
            logger.info("STAGE 5b: Sending Telegram alerts...")
            tg_jobs = dedup.get_unnotified("telegram", min_score=60)
            if tg_jobs:
                success = await send_telegram_alerts(tg_jobs, config)
                if success:
                    dedup.mark_notified(
                        [type('obj', (object,), {'job_hash': j['job_hash']})() for j in tg_jobs],
                        "telegram"
                    )
            else:
                logger.info("No unnotified jobs for Telegram")

        # ── CLEANUP ──
        logger.info("Cleaning up old jobs...")
        dedup.cleanup_old(max_age_days=14)

        # ── STATS ──
        stats = dedup.get_stats()
        elapsed = time.time() - start
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Database stats: {stats}")

    finally:
        dedup.close()


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Job Search Agent")
    parser.add_argument("--email", action="store_true", help="Send email digest")
    parser.add_argument("--telegram", action="store_true", help="Send Telegram alerts")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back")
    parser.add_argument("--dry-run", action="store_true", help="Don't send notifications")
    args = parser.parse_args()

    if args.dry_run:
        import os
        os.environ["DRY_RUN"] = "true"

    asyncio.run(run_pipeline(
        notify_email=args.email,
        notify_telegram=args.telegram,
        hours_back=args.hours,
    ))


if __name__ == "__m