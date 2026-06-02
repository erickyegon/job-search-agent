"""Custom scrapers for priority company career pages."""

import logging
import asyncio
import re
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from ..models import Job

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

AI_KEYWORDS = [
    "ai", "ml", "machine learning", "deep learning", "llm", "genai",
    "generative ai", "nlp", "langchain", "rag", "agent", "mlops",
    "data scien", "prompt engineer", "solutions architect", "research scien",
]


async def scrape_company_career_pages() -> list[Job]:
    """Scrape all priority company career pages."""
    all_jobs: list[Job] = []

    scrapers = [
        _scrape_anthropic,
        _scrape_openai,
        _scrape_cohere,
        _scrape_mistral,
        _scrape_huggingface,
        _scrape_deepmind,
        _scrape_nvidia_careers,
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

    logger.info(f"Total from company pages: {len(all_jobs)} jobs")
    return all_jobs


async def _scrape_anthropic(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Anthropic careers via Ashby API."""
    jobs = []
    try:
        resp = await client.post(
            "https://api.ashbyhq.com/posting-api/job-board/anthropic",
            json={},
        )
        if resp.status_code == 200:
            data = resp.json()
            for posting in data.get("jobs", []):
                title = posting.get("title", "")
                location = posting.get("location", "")
                dept = posting.get("departmentName", "").lower()

                if _is_ai_role(title, dept):
                    job = Job(
                        title=title,
                        company="Anthropic",
                        location=location if location else "San Francisco, CA",
                        url=f"https://jobs.ashbyhq.com/anthropic/{posting.get('id', '')}",
                        source="anthropic_careers",
                        description=posting.get("descriptionPlain", "")[:4000],
                    )
                    published = posting.get("publishedDate")
                    if published:
                        try:
                            job.posted_date = datetime.fromisoformat(published.replace("Z", "+00:00")).replace(tzinfo=None)
                        except Exception:
                            pass
                    jobs.append(job)
    except Exception as e:
        logger.warning(f"Anthropic careers error: {e}")
    return jobs


async def _scrape_openai(client: httpx.AsyncClient) -> list[Job]:
    """Scrape OpenAI careers via Ashby API."""
    jobs = []
    try:
        resp = await client.post(
            "https://api.ashbyhq.com/posting-api/job-board/openai",
            json={},
        )
        if resp.status_code == 200:
            data = resp.json()
            for posting in data.get("jobs", []):
                title = posting.get("title", "")
                dept = posting.get("departmentName", "").lower()

                if _is_ai_role(title, dept):
                    job = Job(
                        title=title,
                        company="OpenAI",
                        location=posting.get("location", "San Francisco, CA"),
                        url=f"https://jobs.ashbyhq.com/openai/{posting.get('id', '')}",
                        source="openai_careers",
                        description=posting.get("descriptionPlain", "")[:4000],
                    )
                    published = posting.get("publishedDate")
                    if published:
                        try:
                            job.posted_date = datetime.fromisoformat(published.replace("Z", "+00:00")).replace(tzinfo=None)
                        except Exception:
                            pass
                    jobs.append(job)
    except Exception as e:
        logger.warning(f"OpenAI careers error: {e}")
    return jobs


async def _scrape_cohere(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Cohere careers via Lever API."""
    jobs = []
    try:
        resp = await client.get("https://api.lever.co/v0/postings/cohere?mode=json")
        if resp.status_code == 200:
            for posting in resp.json():
                title = posting.get("text", "")
                categories = posting.get("categories", {})
                dept = categories.get("department", "").lower()
                location = categories.get("location", "")

                if _is_ai_role(title, dept):
                    job = Job(
                        title=title,
                        company="Cohere",
                        location=location if location else "Remote",
                        url=posting.get("hostedUrl", ""),
                        source="cohere_careers",
                        description=posting.get("descriptionPlain", "")[:4000],
                    )
                    created = posting.get("createdAt")
                    if created:
                        try:
                            job.posted_date = datetime.fromtimestamp(created / 1000)
                        except Exception:
                            pass
                    jobs.append(job)
    except Exception as e:
        logger.warning(f"Cohere careers error: {e}")
    return jobs


async def _scrape_mistral(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Mistral AI careers page."""
    jobs = []
    try:
        resp = await client.get("https://mistral.ai/careers/")
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.select("a[href*='/careers/']"):
                title = link.get_text(strip=True)
                href = link.get("href", "")
                if title and href and href != "/careers/" and _is_ai_role(title, ""):
                    if not href.startswith("http"):
                        href = f"https://mistral.ai{href}"
                    job = Job(
                        title=title,
                        company="Mistral AI",
                        location="Paris, France / Remote",
                        url=href,
                        source="mistral_careers",
                    )
                    jobs.append(job)
    except Exception as e:
        logger.warning(f"Mistral careers error: {e}")
    return jobs


async def _scrape_huggingface(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Hugging Face careers via Workable API."""
    jobs = []
    try:
        resp = await client.get(
            "https://apply.workable.com/api/v3/accounts/huggingface/jobs",
            params={"limit": 50},
        )
        if resp.status_code == 200:
            data = resp.json()
            for posting in data.get("results", []):
                title = posting.get("title", "")
                if _is_ai_role(title, posting.get("department", "")):
                    location = posting.get("location", {})
                    loc_str = location.get("city", "Remote")
                    if location.get("telecommuting"):
                        loc_str = f"Remote ({loc_str})" if loc_str != "Remote" else "Remote"

                    shortcode = posting.get("shortcode", "")
                    job = Job(
                        title=title,
                        company="Hugging Face",
                        location=loc_str,
                        url=f"https://apply.workable.com/huggingface/j/{shortcode}/",
                        source="huggingface_careers",
                    )
                    published = posting.get("published_on")
                    if published:
                        try:
                            job.posted_date = datetime.strptime(published, "%Y-%m-%d")
                        except Exception:
                            pass
                    jobs.append(job)
    except Exception as e:
        logger.warning(f"Hugging Face careers error: {e}")
    return jobs


async def _scrape_deepmind(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Google DeepMind careers."""
    jobs = []
    try:
        resp = await client.get(
            "https://careers.google.com/api/v3/search/",
            params={
                "company": "Google DeepMind",
                "q": "AI engineer",
                "page_size": 30,
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            for result in data.get("search_results", []):
                job_data = result.get("job", {})
                title = job_data.get("title", "")
                locations = job_data.get("locations", [])
                loc_str = ", ".join(loc.get("display", "") for loc in locations[:2]) or "Unknown"

                job = Job(
                    title=title,
                    company="Google DeepMind",
                    location=loc_str,
                    url=f"https://careers.google.com/jobs/results/{job_data.get('id', '')}",
                    source="deepmind_careers",
                    description=job_data.get("description", "")[:4000],
                )
                jobs.append(job)
    except Exception as e:
        logger.warning(f"DeepMind careers error: {e}")
    return jobs


async def _scrape_nvidia_careers(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Nvidia careers for AI roles."""
    jobs = []
    try:
        resp = await client.get(
            "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/jobs",
            json={
                "appliedFacets": {},
                "searchText": "AI engineer",
                "limit": 20,
                "offset": 0,
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            for posting in data.get("jobPostings", []):
                title = posting.get("title", "")
                if _is_ai_role(title, ""):
                    loc = posting.get("locationsText", "Unknown")
                    ext_path = posting.get("externalPath", "")
                    job = Job(
                        title=title,
                        company="Nvidia",
                        location=loc,
                        url=f"https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite{ext_path}",
                        source="nvidia_careers",
                    )
                    posted = posting.get("postedOn")
                    if posted:
                        try:
                            job.posted_date = datetime.strptime(posted, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
                        except Exception:
                            pass
                    jobs.append(job)
    except Exception as e:
        logger.warning(f"Nvidia careers error: {e}")
    return jobs


def _is_ai_role(title: str, department: str) -> bool:
    """Check if a job title/department is AI-relevant."""
    combined = f"{title} {department}".lower()
    # Broader match for career pages — we already filtered to the right companies
    return any(kw in combined for kw in AI_KEYWORDS) or any(
        term in combined for term in [
            "engineer", "architect", "scientist", "developer",
            "product manager", "technical", "platform", "infrastructure",
        ]
    )
