"""Claude Haiku job analyzer — scores jobs against your profile."""

import logging
import json
import yaml
from datetime import datetime
from typing import Optional

import anthropic

from .models import Job, Recommendation
from .config import Config, load_profile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a job matching assistant for a senior AI/ML engineer. Your job is to evaluate job listings against the candidate's profile and return a structured JSON assessment.

You must evaluate each job across 6 dimensions:
1. Technical Match (30%): Overlap between candidate's skills and job requirements
2. Seniority Fit (20%): Does the level match 17+ years of experience? Penalize junior/mid roles.
3. Domain Alignment (15%): Healthcare AI, agentic systems, RAG, LLM engineering overlap
4. Company Tier (15%): Is this a priority company or sector?
5. Remote/Location (10%): Remote-friendly? Compatible with Richmond, KY?
6. Growth Potential (10%): Does it advance the candidate's career?

FRESHNESS BONUS: Add +5 points if posted <24h ago, +3 if posted <48h ago.

FAKE JOB DETECTION: Flag as suspicious if:
- Vague description with no specific tech stack mentioned
- Unrealistic compensation ($500K+ for non-staff roles with no equity mention)
- No company website or verifiable information
- Requests personal/financial information upfront
- Generic "work from home" language instead of specific remote policy

Return ONLY valid JSON in this exact format:
{
  "score": <0-100>,
  "recommendation": "APPLY" | "MAYBE" | "SKIP",
  "reasoning": "<2-3 sentence explanation>",
  "key_matches": ["<skill1>", "<skill2>"],
  "gaps": ["<gap1>", "<gap2>"],
  "salary_estimate": "<estimated range if not listed, null if listed>",
  "is_suspicious": false,
  "suspicious_reason": null
}

SCORING THRESHOLDS:
- APPLY: score >= 80
- MAYBE: score 60-79
- SKIP: score < 60"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """JOB: Senior GenAI Engineer @ Anthropic
LOCATION: Remote (US)
POSTED: 2 hours ago
DESCRIPTION: We're looking for a Senior GenAI Engineer to build production agentic systems using Claude. You'll work with LangChain, MCP, and our Agent SDK. Requirements: 8+ years ML/AI, experience with RAG pipelines, agentic AI orchestration, Python, FastAPI. Strong plus: experience with LLM fine-tuning, evaluation frameworks."""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 97,
            "recommendation": "APPLY",
            "reasoning": "Near-perfect match. Requires LangChain, MCP, agentic AI, RAG, and Claude expertise — all candidate's expert skills. Senior level fits 17+ years. Anthropic is priority company #1. Remote compatible. Freshness bonus: +5 (posted 2h ago).",
            "key_matches": ["LangChain", "MCP", "Agentic AI", "RAG", "Python", "FastAPI", "LLM fine-tuning", "Claude SDK"],
            "gaps": [],
            "salary_estimate": "$250,000–$350,000 TC",
            "is_suspicious": False,
            "suspicious_reason": None
        })
    },
    {
        "role": "user",
        "content": """JOB: Junior Data Analyst @ Unknown Corp
LOCATION: Remote
POSTED: 10 days ago
DESCRIPTION: Looking for a data analyst to create dashboards. Must know Excel and SQL. 0-2 years experience. $45,000-55,000. Send resume to personal email."""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 12,
            "recommendation": "SKIP",
            "reasoning": "Junior role (0-2 years) far below candidate's 17+ years. No AI/ML component. Low compensation. Company not identifiable. Personal email for applications is a red flag.",
            "key_matches": ["SQL"],
            "gaps": [],
            "salary_estimate": None,
            "is_suspicious": True,
            "suspicious_reason": "Unknown company, personal email for applications, no verifiable company info"
        })
    }
]


class JobAnalyzer:
    def __init__(self, config: Config):
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.model = config.claude_model
        self.profile = load_profile()
        self._profile_text = yaml.dump(self.profile, default_flow_style=False)

    def analyze_batch(self, jobs: list[Job]) -> list[Job]:
        """Analyze a batch of jobs against the profile."""
        analyzed = []
        for job in jobs:
            try:
                result = self._analyze_single(job)
                analyzed.append(result)
            except Exception as e:
                logger.warning(f"Analysis failed for '{job.title}' @ {job.company}: {e}")
                job.score = 0
                job.recommendation = Recommendation.SKIP
                job.reasoning = f"Analysis failed: {str(e)[:100]}"
                analyzed.append(job)
        return analyzed

    def _analyze_single(self, job: Job) -> Job:
        """Analyze a single job listing."""
        # Build the job description for Claude
        age_info = ""
        if job.age_hours is not None:
            age_info = f"\nPOSTED: {job.age_display}"

        salary_info = ""
        if job.salary_range:
            salary_info = f"\nSALARY: {job.salary_range}"

        user_message = f"""JOB: {job.title} @ {job.company}
LOCATION: {job.location}
SOURCE: {job.source}{age_info}{salary_info}
URL DOMAIN: {job.url_domain}
DESCRIPTION: {job.description[:2000]}"""

        messages = [
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": user_message}
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=f"{SYSTEM_PROMPT}\n\n--- CANDIDATE PROFILE ---\n{self._profile_text}",
            messages=messages,
        )

        # Parse response
        raw = response.content[0].text.strip()
        # Handle potential markdown wrapping
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Claude response for {job.title}: {raw[:200]}")
            job.score = 50
            job.recommendation = Recommendation.MAYBE
            job.reasoning = "Analysis returned non-JSON response"
            return job

        # Apply results to job
        job.score = min(100, max(0, data.get("score", 0)))
        rec = data.get("recommendation", "SKIP").upper()
        job.recommendation = Recommendation[rec] if rec in Recommendation.__members__ else Recommendation.SKIP
        job.reasoning = data.get("reasoning", "")
        job.key_matches = data.get("key_matches", [])
        job.gaps = data.get("gaps", [])
        job.salary_estimate = data.get("salary_estimate")
        job.is_suspicious = data.get("is_suspicious", False)

        # Override recommendation based on score thresholds (in case Claude is inconsistent)
        if job.score >= 80:
            job.recommendation = Recommendation.APPLY
        elif job.score >= 60:
            job.recommendation = Recommendation.MAYBE
        else:
            job.recommendation = Recommendation.SKIP

        # Mark suspicious jobs as SKIP regardless of score
        if job.is_suspicious:
            job.recommendation = Recommendation.SKIP
            logger.info(f"Suspicious job flagged: {job.title} @ {job.company} — {data.get('suspicious_reason', 'unknown')}")

        logger.info(f"  {job.recommendation.value} ({job.score}): {job.title} @ {job.company}")
        return job
