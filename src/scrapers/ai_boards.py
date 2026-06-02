"""Custom scrapers for AI-specific job boards not covered by JobSpy."""

import logging
import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from ..models import Job

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

AI_KEYWORDS = [
    "ai", "ml", "machine learning", "deep learning", "llm", "genai",
    "generative ai", "nlp", "langchain", "rag", "agent", "mlops",
    "data scientist", "prompt engineer", "solutions architect",
]


async def scrape_ai_boards() -> list[Job]:
    """Scrape all AI-specific job boards."""
    all_jobs: list[Job] = []

    scrapers = [
        _scrape_yc_waaas,
        _scrape_ai_jobs_net,
        _scrape_builtin,
        _scrape_remotive,
    ]

    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        tasks = [scraper(client) for scraper in scrapers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for scraper, result in zip(scrapers, results):
            name = scraper.__name__
            if isinstance(result, Exception):
                logger.warning(f"{name} failed: {result}")
            else:
                logger.info(f"{name}: {len(result)} jobs")
                all_jobs.extend(result)

    logger.info(f"Total from AI boards: {len(all_jobs)} jobs")
    return all_jobs


async def _scrape_yc_waaas(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Y Combinator's Work at a Startup."""
    jobs = []
    try:
        # YC WAAS has a JSON API for Algolia-powered search
        url = "https://45bwzj1sgc-dsn.algolia.net/1/indexes/*/queries"
        params = {
            "x-algolia-agent": "Algolia for JavaScript",
            "x-algolia-application-id": "45BWZJ1SGC",
            "x-algolia-api-key": "MjBjYjRiMzY0NzdhZWY0NjExY2NhZjYxMGIxYjc2MTAwNWFkNTkwNTc4NjgxYjU0YzFhYTY2ZGQ5OGY5NDMzZnJlc3RyaWN0SW5kaWNlcz0lNUIlMjJZQ0NvbXBhbnlfcHJvZHVjdGlvbiUyMiU1RCZ0YWdGaWx0ZXJzPSU1QiUyMiUyMiU1RCZhbmFseXRpY3NUYWdzPSU1QiUyMnljanMlMjIlNUQ=",
        }
        search_queries = ["AI engineer", "ML engineer", "GenAI", "LLM"]

        for query in search_queries:
            payload = {
                "requests": [{
                    "indexName": "YCJob_production",
                    "params": f"query={query}&hitsPerPage=20&filters=isRemote:true"
                }]
            }
            resp = await client.post(url, params=params, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                for hit in data.get("results", [{}])[0].get("hits", []):
                    job = Job(
                        title=hit.get("title", ""),
                        company=hit.get("company", ""),
                        location="Remote" if hit.get("isRemote") else hit.get("location", "Unknown"),
                        url=f"https://www.workatastartup.com/jobs/{hit.get('objectID', '')}",
                        source="yc_waaas",
                        description=hit.get("description", "")[:4000],
                    )
                    if job.title and job.company:
                        jobs.append(job)
    except Exception as e:
        logger.warning(f"YC WAAS scrape error: {e}")
    return jobs


async def _scrape_ai_jobs_net(client: httpx.AsyncClient) -> list[Job]:
    """Scrape aijobs.com for AI-specific roles."""
    jobs = []
    try:
        resp = await client.get("https://aijobs.com/?q=remote")
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for card in soup.select("article.job-card, div.job-listing, .job-item")[:30]:
                title_el = card.select_one("h2 a, h3 a, .job-title a")
                company_el = card.select_one(".company-name, .company, .employer")
                location_el = card.select_one(".location, .job-location")

                if title_el:
                    url = title_el.get("href", "")
                    if url and not url.startswith("http"):
                        url = f"https://aijobs.com{url}"
                    job = Job(
                        title=title_el.get_text(strip=True),
                        company=company_el.get_text(strip=True) if company_el else "Unknown",
                        location=location_el.get_text(strip=True) if location_el else "Unknown",
                        url=url,
                        source="aijobs",
                        description="",
                    )
                    if job.title and job.url:
                        jobs.append(job)
    except Exception as e:
        logger.warning(f"aijobs.com scrape error: {e}")
    return jobs


async def _scrape_builtin(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Built In for remote AI roles."""
    jobs = []
    try:
        url = "https://builtin.com/jobs/remote/dev-engineering/artificial-intelligence"
        resp = await client.get(url)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for card in soup.select("[data-id='job-card'], .job-card, .job-result")[:30]:
                title_el = card.select_one("h2 a, .job-title a, a[data-id='job-card-title']")
                company_el = card.select_one(".company-name, [data-id='company-title']")
                location_el = card.select_one(".job-location, [data-id='job-location']")

                if title_el:
                    href = title_el.get("href", "")
                    if href and not href.startswith("http"):
                        href = f"https://builtin.com{href}"
                    job = Job(
                        title=title_el.get_text(strip=True),
                        company=company_el.get_text(strip=True) if company_el else "Unknown",
                        location=location_el.get_text(strip=True) if location_el else "Remote",
                        url=href,
                        source="builtin",
                        description="",
                    )
                    if job.title and job.url:
                        jobs.append(job)
    except Exception as e:
        logger.warning(f"Built In scrape error: {e}")
    return jobs


async def _scrape_remotive(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Remotive for remote AI/engineering jobs."""
    jobs = []
    try:
        # Remotive has a public JSON API
        resp = await client.get("https://remotive.com/api/remote-jobs?category=software-dev&limit=50")
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("jobs", []):
                title = item.get("title", "").lower()
                tags = " ".join(item.get("tags", [])).lower()
                combined = f"{title} {tags}"
                # Filter for AI-relevant roles
                if any(kw in combined for kw in AI_KEYWORDS):
                    job = Job(
                        title=item.get("title", ""),
                        company=item.get("company_name", ""),
                        location=item.get("candidate_required_location", "Remote"),
                        url=item.get("url", ""),
                        source="remotive",
                        description=item.get("description", "")[:4000],
                        salary_range=item.get("salary", None) or None,
                    )
                    pub_date = item.get("publication_date")
                    if pub_date:
                        try:
                            job.posted_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).replace(tzinfo=None)
                        except Exception:
                            pass
                    if job.title and job.url:
                        jobs.append(job)
    except Exception as e:
        logger.warning(f"Remotive scrape error: {e}")
    return jobs
