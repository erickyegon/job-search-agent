"""Pipeline orchestrator."""

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


async def run_pipeline(notify_email=False, notify_telegram=False, hours_back=24):
    start = time.time()
    config = Config.from_env()
    config.notify_email = notify_email
    config.notify_telegram = notify_telegram
    errors = config.validate()
    if errors:
        for err in errors:
            logger.error(f"Config error: {err}")
        if not config.dry_run:
            sys.exit(1)
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    logger.info("=" * 60)
    logger.info(f"Job Search Agent - {now_str}")
    mode = 'DRY RUN' if config.dry_run else 'LIVE'
    logger.info(f"Mode: {mode}")
    em = 'yes' if notify_email else 'no'
    tg = 'yes' if notify_telegram else 'no'
    logger.info(f"Notifications: email={em}, telegram={tg}")
    logger.info(f"Lookback: {hours_back} hours")
    logger.info("=" * 60)
    dedup = DedupEngine(config.db_path)
    try:
        logger.info("STAGE 1: Scraping job sources...")
        aggregator_jobs = scrape_aggregators(hours_back=hours_back)
        ai_board_jobs = await scrape_ai_boards()
        company_jobs = await scrape_company_career_pages()
        all_jobs = aggregator_jobs + ai_board_jobs + company_jobs
        logger.info(f"Total scraped: {len(all_jobs)} jobs (aggregators: {len(aggregator_jobs)}, AI boards: {len(ai_board_jobs)}, company pages: {len(company_jobs)})")
        if not all_jobs:
            logger.info("No jobs found. Exiting.")
            return
        logger.info("STAGE 2: Deduplicating...")
        new_jobs = dedup.filter_new(all_jobs)
        if not new_jobs:
            logger.info("No new jobs since last run.")
        else:
            logger.info(f"New jobs to analyze: {len(new_jobs)}")
            logger.info("STAGE 3: Validating links...")
            verified_jobs = await validate_links(new_jobs)
            logger.info(f"Jobs with verified links: {len(verified_jobs)}")
            if not verified_jobs:
                logger.info("No jobs passed link validation.")
            else:
                logger.info("STAGE 4: Analyzing with Claude...")
                analyzer = JobAnalyzer(config)
                analyzed_jobs = analyzer.analyze_batch(verified_jobs)
                clean_jobs = [j for j in analyzed_jobs if not j.is_suspicious]
                suspicious_count = len(analyzed_jobs) - len(clean_jobs)
                if suspicious_count:
                    logger.info(f"Filtered {suspicious_count} suspicious jobs")
                dedup.save_jobs(clean_jobs)
                ac = sum(1 for j in clean_jobs if j.recommendation.value == "APPLY")
                mc = sum(1 for j in clean_jobs if j.recommendation.value == "MAYBE")
                sc = sum(1 for j in clean_jobs if j.recommendation.value == "SKIP")
                logger.info(f"Analysis results: {ac} APPLY, {mc} MAYBE, {sc} SKIP")
        if notify_email:
            logger.info("STAGE 5a: Sending email digest...")
            email_jobs = dedup.get_unnotified("email", min_score=60)
            if email_jobs:
                success = send_email_digest(email_jobs, config)
                if success:
                    objs = [type('o', (object,), {'job_hash': j['job_hash']})() for j in email_jobs]
                    dedup.mark_notified(objs, "email")
            else:
                logger.info("No unnotified jobs for email")
        if notify_telegram:
            logger.info("STAGE 5b: Sending Telegram alerts...")
            tg_jobs = dedup.get_unnotified("telegram", min_score=60)
            if tg_jobs:
                success = await send_telegram_alerts(tg_jobs, config)
                if success:
                    objs = [type('o', (object,), {'job_hash': j['job_hash']})() for j in tg_jobs]
                    dedup.mark_notified(objs, "telegram")
            else:
                logger.info("No unnotified jobs for Telegram")
        logger.info("Cleaning up old jobs...")
        dedup.cleanup_old(max_age_days=14)
        stats = dedup.get_stats()
        elapsed = time.time() - start
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Database stats: {stats}")
    finally:
        dedup.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Job Search Agent")
    parser.add_argument("--email", action="store_true")
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        import os
        os.environ["DRY_RUN"] = "true"
    asyncio.run(run_pipeline(
        notify_email=args.email,
        notify_telegram=args.telegram,
        hours_back=args.hours,
    ))


if __name__ == "__main__":
    main()
