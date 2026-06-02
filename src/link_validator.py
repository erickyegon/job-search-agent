"""Three-stage link validation pipeline.

Stage 1: URL extraction & normalization
Stage 2: HTTP HEAD validation (status, redirect, content-type)
Stage 3: Content verification for high-score jobs (checks for job-related keywords)
"""

import logging
import asyncio
from urllib.parse import urlparse
from typing import Optional

import httpx

from .models import Job, LinkStatus
from .config import load_trusted_domains

logger = logging.getLogger(__name__)

# Keywords that indicate a live job page
JOB_PAGE_KEYWORDS = [
    "apply", "submit", "application", "candidate", "resume", "cv",
    "qualifications", "responsibilities", "requirements", "experience",
    "salary", "benefits", "job description", "about the role",
    "what you'll do", "who you are", "about you",
]

# Keywords that indicate a dead/generic page
DEAD_PAGE_KEYWORDS = [
    "page not found", "404", "no longer available", "position has been filled",
    "this job has expired", "job no longer exists", "removed",
    "sorry, this position", "role has been closed",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


async def validate_links(jobs: list[Job], deep_check_threshold: int = 80) -> list[Job]:
    """Validate all job links. Returns only jobs with verified links.

    Args:
        jobs: List of jobs to validate
        deep_check_threshold: Score threshold for deep content verification
    """
    validated = []
    trusted = load_trusted_domains()

    async with httpx.AsyncClient(
        headers=HEADERS, timeout=15, follow_redirects=True, max_redirects=5
    ) as client:
        # Stage 1: URL extraction & normalization
        for job in jobs:
            job = _normalize_url(job)
            if not job.url:
                logger.debug(f"No URL for: {job.title} @ {job.company}")
                job.link_status = LinkStatus.DEAD
                continue

            # Extract domain
            parsed = urlparse(job.url)
            job.url_domain = parsed.netloc.lower().lstrip("www.")

        # Stage 2: HTTP HEAD validation (batch with concurrency limit)
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
        tasks = []
        for job in jobs:
            if job.link_status != LinkStatus.DEAD:
                tasks.append(_check_http(client, job, semaphore))
            else:
                tasks.append(asyncio.coroutine(lambda j=job: j)())

        checked_jobs = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(checked_jobs):
            if isinstance(result, Exception):
                jobs[i].link_status = LinkStatus.DEAD
                logger.debug(f"Link check failed for {jobs[i].url}: {result}")
            elif isinstance(result, Job):
                jobs[i] = result

        # Stage 3: Content verification for high-scoring APPLY candidates
        for job in jobs:
            if job.link_status == LinkStatus.VERIFIED and job.score >= deep_check_threshold:
                try:
                    job = await _deep_verify(client, job)
                except Exception as e:
                    logger.debug(f"Deep verify failed for {job.url}: {e}")

        # Filter: only return jobs with verified links
        for job in jobs:
            if job.link_status == LinkStatus.VERIFIED:
                job.url_verified = True
                validated.append(job)
            else:
                logger.info(f"Excluded (link {job.link_status.value}): {job.title} @ {job.company} [{job.url}]")

    logger.info(f"Link validation: {len(validated)}/{len(jobs)} jobs passed")
    return validated


def _normalize_url(job: Job) -> Job:
    """Stage 1: Clean and normalize the URL."""
    url = job.url.strip()
    if not url:
        return job

    # Ensure protocol
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    # Remove tracking parameters
    parsed = urlparse(url)
    # Keep essential query params, strip tracking ones
    job.url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        clean_params = []
        for param in parsed.query.split("&"):
            key = param.split("=")[0].lower()
            # Keep job-relevant params
            if key not in ("utm_source", "utm_medium", "utm_campaign", "utm_content",
                          "utm_term", "fbclid", "gclid", "ref", "source"):
                clean_params.append(param)
        if clean_params:
            job.url += "?" + "&".join(clean_params)

    return job


async def _check_http(client: httpx.AsyncClient, job: Job, semaphore: asyncio.Semaphore) -> Job:
    """Stage 2: HTTP HEAD request to validate the URL resolves."""
    async with semaphore:
        try:
            # Try HEAD first (lighter)
            resp = await client.head(job.url)

            # Some sites don't support HEAD, fall back to GET
            if resp.status_code == 405:
                resp = await client.get(job.url)

            # Check response
            if resp.status_code == 200:
                job.link_status = LinkStatus.VERIFIED
                # Track final URL after redirects
                if str(resp.url) != job.url:
                    job.redirect_url = str(resp.url)
            elif resp.status_code in (301, 302, 307, 308):
                # Should have been followed by httpx, but just in case
                job.link_status = LinkStatus.VERIFIED
            elif resp.status_code in (403, 429):
                # Might be rate-limited or bot-blocked; still could be real
                job.link_status = LinkStatus.VERIFIED
                logger.debug(f"Got {resp.status_code} for {job.url} â€” keeping as may be bot protection")
            elif resp.status_code == 404:
                job.link_status = LinkStatus.DEAD
            else:
                job.link_status = LinkStatus.SUSPICIOUS
                logger.debug(f"Unexpected status {resp.status_code} for {job.url}")

        except httpx.TimeoutException:
            # Timeout might mean the page is slow, not dead
            job.link_status = LinkStatus.SUSPICIOUS
        except httpx.ConnectError:
            job.link_status = LinkStatus.DEAD
        except Exception as e:
            job.link_status = LinkStatus.SUSPICIOUS
            logger.debug(f"Link check error for {job.url}: {e}")

        # Small delay to be polite
        await asyncio.sleep(0.5)
        return job


async def _deep_verify(client: httpx.AsyncClient, job: Job) -> Job:
    """Stage 3: Fetch page content and verify it's a real job listing."""
    try:
        resp = await client.get(job.url)
        if resp.status_code != 200:
            return job

        text = resp.text.lower()

        # Check for dead page indicators
        for keyword in DEAD_PAGE_KEYWORDS:
            if keyword in text:
                job.link_status = LinkStatus.DEAD
                logger.info(f"Dead listing detected: {job.title} @ {job.company} â€” '{keyword}' found")
                return job

        # Check for job page indicators
        job_keyword_count = sum(1 for kw in JOB_PAGE_KEYWORDS if kw in text)
        if job_keyword_count < 2:
            job.link_status = LinkStatus.SUSPICIOUS
            logger.info(f"Suspicious page (only {job_keyword_count} job keywords): {job.url}")

    except Exception:
        pass  # Keep existing status if deep verify fails

    return job
