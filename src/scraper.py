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

JOBSPY_SITES = ["indeed", "linkedin", "google"]

CORE_QUERIES = [
    "generative AI engineer",
    "AI ML engineer",
    "LLM engineer",
    "AI solutions architect",
    "machine learning engineer senior",
    "MLOps engineer",
    "data scientist senior",
    "research scientist AI",
    "biostatistician",
    "epidemiologist data scientist",
    "epidemiologist",
    "health data scientist",
    "AI product manager",
    "developer advocate AI",
    "NLP engineer",
    "AI architect",
"Research Scientist"
]


def scrape_aggregators(hours_back: int = 24) -> list[Job]:
    config = load_search_queries()
    queries = CORE_QUERIES
    locations = ["United States"]
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
                    linkedin_fetch_description=True,
                    #is_remote=True,
                )
                if df is not None and not df.empty:
                    jobs = _df_to_jobs(df, seen_urls)
                    all_jobs.extend(jobs)
                    logger.info(f"  Found {len(jobs)} new jobs")
                else:
                    logger.info("  Found 0 new jobs")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"JobSpy error for '{query}' in '{location}': {e}")
                continue
    logger.info(f"Total from aggregators: {len(all_jobs)} jobs")
    return all_jobs


def _df_to_jobs(df: pd.DataFrame, seen_urls: set[str]) -> list[Job]:
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
            company=str(row.get("company", "")).strip(),
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


def _normalize_location(row: pd.Series) -> str:
    parts = []
    if pd.notna(row.get("location")):
        parts.append(str(row["location"]).strip())
    is_remote = row.get("is_remote", False)
    if is_remote:
        if parts:
            parts[0] = f"Remote ({parts[0]})"
        else:
            parts.append("Remote")
    return parts[0] if parts else "Unknown"


def _extract_salary(row: pd.Series) -> Optional[str]:
    min_sal = row.get("min_amount")
    max_sal = row.get("max_amount")
    currency = row.get("currency", "USD")
    if pd.notna(min_sal) and pd.notna(max_sal):
        return f"${int(min_sal):,}-${int(max_sal):,} {currency}"
    if pd.notna(min_sal):
        return f"${int(min_sal):,}+ {currency}"
    return None
