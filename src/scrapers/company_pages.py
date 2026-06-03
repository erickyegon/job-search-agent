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
    "Accept": "application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

AI_KEYWORDS = [
    "ai", "ml", "machine learning", "deep learning", "llm", "genai",
    "generative ai", "nlp", "langchain", "rag", "agent", "mlops",
    "data scien", "prompt engineer", "solutions architect", "research scien",
    "biostatist", "statistici", "epidemiolog", "health data", "population health",
]


async def scrape_company_career_pages() -> list[Job]:
    """Scrape all priority company career pages."""
    all_jobs: list[Job] = []

    scrapers = [
        # AI companies (Greenhouse / Ashby)
        _scrape_anthropic,
        _scrape_openai,
        _scrape_cohere,
        _scrape_deepmind,
        _scrape_mistral,
        _scrape_huggingface,
        _scrape_nvidia_careers,
        # Health / Research organizations (Greenhouse)
        _scrape_cdc_foundation,
        _scrape_tempus,
        _scrape_flatiron_health,
        _scrape_verily,
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


# ---------- Greenhouse helper ----------

async def _scrape_greenhouse(client: httpx.AsyncClient, board_token: str,
                              company_name: str, source_tag: str,
                              filter_fn=None) -> list[Job]:
    """Generic Greenhouse job board scraper. No auth needed."""
    jobs = []
    try:
        resp = await client.get(
            f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs",
            params={"content": "true"},
        )
        if resp.status_code != 200:
            logger.warning(f"Greenhouse {board_token}: HTTP {resp.status_code}")
            return jobs

        data = resp.json()
        for posting in data.get("jobs", []):
            title = posting.get("title", "")
            loc = posting.get("location", {}).get("name", "Unknown")
            desc = posting.get("content", "")
            # Strip HTML from content
            if desc:
                desc = BeautifulSoup(desc, "html.parser").get_text(" ", strip=True)[:4000]

            if filter_fn and not filter_fn(title, desc):
                continue

            job = Job(
                title=title,
                company=company_name,
                location=loc,
                url=posting.get("absolute_url", ""),
                source=source_tag,
                description=desc,
            )
            updated = posting.get("updated_at")
            if updated:
                try:
                    job.posted_date = datetime.fromisoformat(
                        updated.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    pass
            jobs.append(job)
    except Exception as e:
        logger.warning(f"Greenhouse {board_token} error: {e}")
    return jobs


# ---------- Ashby helper ----------

async def _scrape_ashby(client: httpx.AsyncClient, board_name: str,
                         company_name: str, source_tag: str,
                         filter_fn=None) -> list[Job]:
    """Generic Ashby job board scraper. Public GET, no auth."""
    jobs = []
    try:
        resp = await client.get(
            f"https://api.ashbyhq.com/posting-api/job-board/{board_name}",
            params={"includeCompensation": "true"},
        )
        if resp.status_code != 200:
            logger.warning(f"Ashby {board_name}: HTTP {resp.status_code}")
            return jobs

        data = resp.json()
        for posting in data.get("jobs", []):
            title = posting.get("title", "")
            loc = posting.get("location", "Unknown")
            dept = posting.get("department", "")
            desc = posting.get("descriptionPlain", "")[:4000]

            if filter_fn and not filter_fn(title, f"{dept} {desc}"):
                continue

            job = Job(
                title=title,
                company=company_name,
                location=loc,
                url=posting.get("jobUrl", posting.get("applyUrl", "")),
                source=source_tag,
                description=desc,
            )
            published = posting.get("publishedAt")
            if published:
                try:
                    job.posted_date = datetime.fromisoformat(
                        published.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    pass
            jobs.append(job)
    except Exception as e:
        logger.warning(f"Ashby {board_name} error: {e}")
    return jobs


# ---------- AI company scrapers ----------

async def _scrape_anthropic(client: httpx.AsyncClient) -> list[Job]:
    """Anthropic — now on Greenhouse."""
    return await _scrape_greenhouse(client, "anthropic", "Anthropic", "anthropic_careers")


async def _scrape_openai(client: httpx.AsyncClient) -> list[Job]:
    """OpenAI — on Ashby."""
    return await _scrape_ashby(client, "openai", "OpenAI", "openai_careers")


async def _scrape_cohere(client: httpx.AsyncClient) -> list[Job]:
    """Cohere — moved to Ashby."""
    return await _scrape_ashby(client, "cohere", "Cohere", "cohere_careers")


async def _scrape_deepmind(client: httpx.AsyncClient) -> list[Job]:
    """Google DeepMind — on Greenhouse."""
    return await _scrape_greenhouse(client, "deepmind", "Google DeepMind", "deepmind_careers")


async def _scrape_mistral(client: httpx.AsyncClient) -> list[Job]:
    """Scrape Mistral AI careers page (HTML)."""
    jobs = []
    try:
        resp = await client.get("https://mistral.ai/careers/")
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Look for job links in the careers page
            for link in soup.select("a[href*='/careers/']"):
                title = link.get_text(strip=True)
                href = link.get("href", "")
                if title and href and href != "/careers/" and len(title) > 3:
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
    """Hugging Face — on Workable. Scrape the public page."""
    jobs = []
    try:
        # Workable public JSON endpoint
        resp = await client.get(
            "https://apply.workable.com/api/v1/widget/accounts/huggingface",
            params={"details": "true"},
        )
        if resp.status_code == 200:
            data = resp.json()
            for posting in data.get("jobs", []):
                title = posting.get("title", "")
                loc = posting.get("city", "Remote")
                if posting.get("telecommuting"):
                    loc = f"Remote ({loc})" if loc and loc != "Remote" else "Remote"

                shortcode = posting.get("shortcode", "")
                job = Job(
                    title=title,
                    company="Hugging Face",
                    location=loc,
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
        else:
            logger.warning(f"HuggingFace Workable: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Hugging Face careers error: {e}")
    return jobs


async def _scrape_nvidia_careers(client: httpx.AsyncClient) -> list[Job]:
    """Nvidia — Workday. Must use POST."""
    jobs = []
    try:
        resp = await client.post(
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
                if _is_relevant_role(title, ""):
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
        else:
            logger.warning(f"Nvidia Workday: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Nvidia careers error: {e}")
    return jobs


# ---------- Health / Research organization scrapers ----------

async def _scrape_cdc_foundation(client: httpx.AsyncClient) -> list[Job]:
    """CDC Foundation — on Greenhouse."""
    def _filter(title, desc):
        combined = f"{title} {desc}".lower()
        return any(kw in combined for kw in [
            "biostatist", "statistici", "epidemiolog", "data scien",
            "research", "health", "analyst", "ai", "ml", "machine learn",
        ])
    return await _scrape_greenhouse(client, "cdcfoundation", "CDC Foundation",
                                     "cdc_foundation_careers", filter_fn=_filter)


async def _scrape_tempus(client: httpx.AsyncClient) -> list[Job]:
    """Tempus AI — precision medicine, on Greenhouse."""
    def _filter(title, desc):
        combined = f"{title} {desc}".lower()
        return any(kw in combined for kw in [
            "biostatist", "statistici", "epidemiolog", "data scien",
            "research scien", "ai", "ml", "machine learn", "engineer",
        ])
    return await _scrape_greenhouse(client, "tempus", "Tempus AI",
                                     "tempus_careers", filter_fn=_filter)


async def _scrape_flatiron_health(client: httpx.AsyncClient) -> list[Job]:
    """Flatiron Health — healthcare data, on Greenhouse."""
    def _filter(title, desc):
        combined = f"{title} {desc}".lower()
        return any(kw in combined for kw in [
            "biostatist", "statistici", "epidemiolog", "data scien",
            "research", "ai", "ml", "machine learn", "engineer", "quantitative",
        ])
    return await _scrape_greenhouse(client, "flatironhealth", "Flatiron Health",
                                     "flatiron_careers", filter_fn=_filter)


async def _scrape_verily(client: httpx.AsyncClient) -> list[Job]:
    """Verily (Alphabet health science) — on Greenhouse."""
    def _filter(title, desc):
        combined = f"{title} {desc}".lower()
        return any(kw in combined for kw in [
            "biostatist", "statistici", "epidemiolog", "data scien",
            "research scien", "ai", "ml", "machine learn", "health",
        ])
    return await _scrape_greenhouse(client, "verily", "Verily",
                                     "verily_careers", filter_fn=_filter)


# ---------- Utility ----------

def _is_relevant_role(title: str, department: str) -> bool:
    """Check if a job title/department is relevant to candidate's profile."""
    combined = f"{title} {department}".lower()
    return any(kw in combined for kw in AI_KEYWORDS) or any(
        term in combined for term in [
            "engineer", "architect", "scientist", "developer",
            "product manager", "technical", "platform", "infrastructure",
        ]
    )
