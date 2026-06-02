"""Multi-source job scraper using JobSpy + custom scrapers."""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from jobspy import scrape_jobs
import pandas as pd

from .models import Job
from .config import load_search_queries

logger = logging.getLogger(__name__)

# Only use sources that work from datacenter IPs (GitHub Actions)
# ZipRecruiter: blocks with 403 Cloudflare WAF
# Glassdoor: API errors from datacenter IPs
# LinkedIn guest API: works but returns limited results
# Indeed: works with server-side rendering
# Google Jobs: most reliable aggregator from any IP
JOBSPY_SITES = ["indeed", "linkedin", "google"]

# Reduced, high-signal queries to avoid timeouts (18 queries × 2 locations was too slow)
CORE_QUERIES = [
    "generative AI engineer",
    "AI ML engineer",
    "LLM engineer",
    "AI solutions architect",
    "machine learning engineer",
    "AI product manager",
    "MLOps engineer",
    "AI developer advocate",
]


def scrape_aggregators(hours_back: int = 24) -> list[Job]:
    """Scrape all JobSpy-supported boards."""
    config = load_search_queries()
    # Use core queries (faster) but fall back to config if needed
    queries = CORE_QUERIES
    locations = config.get("locations", ["Remote"])
    max_results = config.get("max_results_per_query", 25)

    all_jobs: list[Job] = []
    seen_urls: set[str] = set()

    for query in queries:
        for location in locations:
            try:
                logger.info(f"JobSpy: '{query}' in '{location}'")
                df = scrape_jobs(
                    site_name=JOBSPY_SITES,
                    search_term=query,
                    location=location,
                    results_wanted=max_results,
                    hours_old=hours_back,
                    country_indeed="USA",
                    linkedin_fetch_description=False,  # Faster: skip full description fetch
                    is_remote=True,
                )
                if df is not None and not df.empty:
                    jobs = _df_to_jobs(df, seen_urls)
                    all_jobs.extend(jobs)
                    logger.info(f"  Found {len(jobs)} new jobs")
                else:
                    logger.info(f"  Found 0 new jobs")

                # Rate limit: be polite between queries
                time.sleep(1)

            except Exception as e:
                logger.warning(f"JobSpy error for '{query}' in '{location}': {e}")
                continue

    logger.info(f"Total from aggregators: {len(all_jobs)} jobs")
    return all_jobs


def _df_to_jobs(df: pd.DataFrame, seen_urls: set[str]) -> list[Job]:
    """Convert JobSpy DataFrame to Job objects, skipping duplicates."""
    jobs = []
    for _, row in df.iterrows():
        url = str(row.get("job_url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        posted = None
        if pd.notna(row.get("date_posted")):
            try:
                posted = pd.to_datetime(row["date_posted"]).to_pydatetime()
            except Exception:
                pass

        job = Job(
            title=str(row.get("title", "")).strip(),
            company=str(row.get("company_name", "")).strip(),
            location=_normalize_location(row),
            url=url,
            source=str(row.get("site", "unknown")),
            description=str(row.get("description", ""))[:4000],
            posted_date=posted,
            salary_range=_extract_salary(row),
        )
        if job.title and job.company:
            jobs.append(job)
    return jobs


def _no