"""Claude Haiku job analyzer — dual-track scoring for health executive + AI architect profiles."""

import asyncio
import logging
import json
import yaml
from datetime import datetime
from typing import Optional

import anthropic

from .models import Job, Recommendation
from .config import Config, load_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Track A — Global Health Executive
# Domain 35% | Leadership 25% | Technical 15% | Mission 15% | Location 10%
# Target: Gates, PATH, CHAI, UNICEF, WHO, Truveta, IHME, Optum, Humana,
#         research institutes, global health NGOs, health-focused foundations
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TRACK_A = """You are a job matching assistant evaluating roles for a Global Health Executive.

CANDIDATE IDENTITY (Track A lens):
Erick Yegon is a health analytics executive with 17+ years leading data science and research programs
across global health. Former Global Director at Living Goods. PhD Epidemiology. Expert in population
health, causal inference, impact evaluation, LiST modeling, and health systems analytics. He also builds
production AI systems (LangChain, LangGraph, RAG, multi-agent), but for this track the health executive
identity is primary.

Evaluate each job across 5 dimensions:
1. Domain Alignment (35%): Population health, epidemiology, global health, health economics, outcomes
   research, real-world evidence, health systems, digital health, impact evaluation, causal inference.
   This is where his PhD and 17 years are irreplaceable — score generously.
2. Leadership/Seniority Fit (25%): Director, VP, Head, Principal, Chief, Senior-level leadership.
   Does it match executive experience? Penalize anything below senior individual contributor.
3. Technical Match (15%): Analytical methods, statistical modeling, AI/ML for health, R, Python, SQL,
   Power BI, Tableau. GenAI/LLM skills are a bonus but not required for this track.
4. Mission/Strategic Fit (15%): Global health mission, public health impact, health equity, foundation
   funding, government/USAID alignment, research institute culture.
5. Remote/Location (10%): Remote-friendly, or located in Richmond KY / Washington State / Oregon.

HEALTHCARE/GLOBAL HEALTH BONUS: Add +12 points to the raw score when the role explicitly involves
ANY of: population health, epidemiology, public health, outcomes research, health economics,
real-world evidence, global health, community health, health systems, digital health, health equity,
reproductive health, maternal health, RMNCH, MNCH, LiST, DHS, HMIS, DHIS2, impact evaluation.

FRESHNESS BONUS: +5 if posted <24h ago, +3 if posted <48h ago.

FAKE JOB DETECTION: Flag suspicious if vague description, unverifiable company, requests personal/financial
info upfront, or no specific job responsibilities listed.

Return ONLY valid JSON:
{
  "score": <0-100>,
  "track": "A",
  "recommendation": "APPLY" | "MAYBE" | "SKIP",
  "reasoning": "<2-3 sentence explanation emphasizing health/leadership fit>",
  "key_matches": ["<match1>", "<match2>"],
  "gaps": ["<gap1>"],
  "salary_estimate": "<range or null if listed>",
  "is_suspicious": false,
  "suspicious_reason": null,
  "health_bonus_applied": <true|false>
}

SCORING THRESHOLDS:
- APPLY: score >= 75  (lower threshold for domain-heavy health/foundation roles)
- MAYBE: score 55-74
- SKIP: score < 55"""

# ---------------------------------------------------------------------------
# Track B — AI / Agentic Architecture
# Technical 35% | Leadership 20% | Domain 20% | Company 15% | Location 10%
# Target: Deloitte, NVIDIA, Microsoft, Accenture, OpenAI, Netflix, Teradata,
#         Humana AI, Optum AI, enterprise AI consultancies
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TRACK_B = """You are a job matching assistant evaluating roles for an AI/Agentic Architecture leader.

CANDIDATE IDENTITY (Track B lens):
Erick Yegon is an AI architect and engineer who builds production agentic systems: LangChain, LangGraph,
CrewAI, AutoGen, MCP, Advanced RAG, multi-agent orchestration, FastAPI, NeMo Guardrails, LLM fine-tuning,
RAGAS evaluation, FAISS, Qdrant. 17 years of experience including executive leadership. PhD Epidemiology
gives him an edge in healthcare AI and responsible AI. He is NOT a junior engineer — score seniority fit
strictly against staff/senior/principal/director level.

Evaluate each job across 5 dimensions:
1. Technical Match (35%): LangChain, LangGraph, RAG, agentic AI, MCP, LLM fine-tuning, multi-agent
   systems, FastAPI, Python, AI guardrails/safety, MLOps/LLMOps, vector databases.
2. Leadership/Seniority Fit (20%): Senior, Staff, Principal, Director, Lead, Architect level.
   Penalize anything junior or mid-level.
3. Domain Alignment (20%): Healthcare AI gives a natural edge. AI safety, responsible AI, enterprise AI.
   Health + AI intersection is especially valuable.
4. Company Quality (15%): Priority companies: Anthropic, OpenAI, Microsoft, NVIDIA, Google DeepMind,
   Databricks, Hugging Face, Deloitte, Booz Allen, McKinsey QuantumBlack, Accenture, Perficient.
   Established enterprises with AI investment > unknown startups.
5. Remote/Location (10%): Remote-friendly, or located in Richmond KY / Washington State / Oregon.

HEALTHCARE/GLOBAL HEALTH BONUS: Add +8 points when the role explicitly combines AI/ML with healthcare,
health outcomes, clinical data, population health, or public health domains.

FRESHNESS BONUS: +5 if posted <24h ago, +3 if posted <48h ago.

FAKE JOB DETECTION: Flag suspicious if vague, unverifiable company, upfront personal data requests,
no tech stack mentioned, or generic "AI company" with no verifiable presence.

Return ONLY valid JSON:
{
  "score": <0-100>,
  "track": "B",
  "recommendation": "APPLY" | "MAYBE" | "SKIP",
  "reasoning": "<2-3 sentence explanation emphasizing AI/technical fit>",
  "key_matches": ["<match1>", "<match2>"],
  "gaps": ["<gap1>"],
  "salary_estimate": "<range or null if listed>",
  "is_suspicious": false,
  "suspicious_reason": null,
  "health_bonus_applied": <true|false>
}

SCORING THRESHOLDS:
- APPLY: score >= 80
- MAYBE: score 60-79
- SKIP: score < 60"""

# ---------------------------------------------------------------------------
# Few-shot examples — one per track
# ---------------------------------------------------------------------------
FEW_SHOT_TRACK_A = [
    {
        "role": "user",
        "content": """JOB: Director of Health Analytics @ Gates Foundation
LOCATION: Remote (US) / Seattle, WA preferred
POSTED: 6 hours ago
DESCRIPTION: We seek a Director to lead population health analytics, drive impact evaluation of RMNCH
programs, and build AI-augmented measurement systems. Requires PhD in epidemiology or biostatistics,
10+ years in global health analytics, experience with LiST modeling, causal inference, and DHS data.
Python/R required. AI/ML for health outcomes is a strong plus."""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 99,
            "track": "A",
            "recommendation": "APPLY",
            "reasoning": "Near-perfect Track A match. PhD Epidemiology, 17 years global health, LiST modeling, RMNCH expertise, and causal inference are all exact matches. Gates Foundation is top-tier mission alignment. AI/ML skills are a strong bonus. Health bonus applied: +12. Freshness bonus: +5.",
            "key_matches": ["PhD Epidemiology", "Global Health", "LiST Modeling", "RMNCH", "Causal Inference", "DHS", "Python", "R", "Impact Evaluation"],
            "gaps": [],
            "salary_estimate": "$180,000–$240,000",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": True
        })
    },
    {
        "role": "user",
        "content": """JOB: Junior Data Analyst @ Unknown Corp
LOCATION: Remote
POSTED: 10 days ago
DESCRIPTION: Looking for a data analyst to create dashboards. Must know Excel. 0-2 years experience.
$45,000. Send resume to personal email."""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 8,
            "track": "A",
            "recommendation": "SKIP",
            "reasoning": "Junior role (0-2 years) far below 17+ years executive experience. No health/global health domain. Unknown company. Personal email is a red flag.",
            "key_matches": [],
            "gaps": ["Leadership level", "Domain alignment", "Company verifiability"],
            "salary_estimate": None,
            "is_suspicious": True,
            "suspicious_reason": "Unknown company, personal email for applications",
            "health_bonus_applied": False
        })
    }
]

FEW_SHOT_TRACK_B = [
    {
        "role": "user",
        "content": """JOB: Senior GenAI Engineer @ Anthropic
LOCATION: Remote (US)
POSTED: 2 hours ago
DESCRIPTION: Build production agentic systems using Claude. Work with LangChain, MCP, Agent SDK.
Requirements: 8+ years ML/AI, RAG pipelines, agentic AI orchestration, Python, FastAPI.
Strong plus: LLM fine-tuning, evaluation frameworks, AI safety."""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 97,
            "track": "B",
            "recommendation": "APPLY",
            "reasoning": "Near-perfect Track B match. LangChain, MCP, agentic AI, RAG, Claude SDK are all expert-level skills. Senior level fits 17+ years. Anthropic is highest-tier priority company. AI safety is a bonus match. Freshness bonus: +5.",
            "key_matches": ["LangChain", "MCP", "Agentic AI", "RAG", "Python", "FastAPI", "LLM Fine-Tuning", "AI Safety", "Claude SDK"],
            "gaps": [],
            "salary_estimate": "$250,000–$350,000 TC",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": False
        })
    },
    {
        "role": "user",
        "content": """JOB: Senior AI Engineer — Clinical Decision Support @ Optum AI
LOCATION: Remote (US)
POSTED: 1 day ago
DESCRIPTION: Build LLM-powered clinical decision support tools using RAG and agentic workflows.
Healthcare data experience strongly preferred. Python, LangChain, vector databases required.
8+ years experience. Population health or clinical AI background a strong plus."""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 98,
            "track": "B",
            "recommendation": "APPLY",
            "reasoning": "Exceptional Track B match with healthcare AI bonus. LangChain, RAG, agentic workflows are expert skills. Clinical AI + population health background is an exact differentiator. Optum is a priority company. Healthcare AI bonus applied: +8. Freshness: +3.",
            "key_matches": ["LangChain", "RAG", "Agentic AI", "Python", "Vector Databases", "Population Health", "Clinical AI"],
            "gaps": [],
            "salary_estimate": "$180,000–$230,000",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": True
        })
    }
]


def _classify_track(job: Job) -> str:
    """
    Classify a job as Track A (Global Health Executive) or Track B (AI/Agentic).
    Jobs containing strong AI/ML engineering signals → Track B.
    Jobs containing strong health/global health executive signals → Track A.
    Jobs at the intersection → scored on BOTH tracks, best score wins.
    """
    title_lower = (job.title or "").lower()
    desc_lower = (job.description or "").lower()
    combined = title_lower + " " + desc_lower

    health_signals = [
        "epidemiolog", "population health", "global health", "public health",
        "health system", "health economics", "outcomes research", "real-world evidence",
        "impact evaluation", "rmnch", "mnch", "reproductive health", "maternal health",
        "biostatistic", "digital health", "health equity", "community health",
        "dhis", "hmis", "lst model", "gates", "path ", "chai ", "unicef", "who ",
        "usaid", "ihme", "truveta", "health analytics director", "health data director"
    ]

    ai_signals = [
        "langchain", "langgraph", "llm", "large language model", "genai", "gen ai",
        "agentic", "rag pipeline", "vector database", "llmops", "mlops", "mcp",
        "prompt engineer", "fine-tun", "multimodal", "ai architect", "ai engineer",
        "machine learning engineer", "ml engineer", "nemo guardrails", "crewai", "autogen"
    ]

    health_score = sum(1 for s in health_signals if s in combined)
    ai_score = sum(1 for s in ai_signals if s in combined)

    if health_score >= 2 and ai_score >= 2:
        return "BOTH"
    elif health_score > ai_score:
        return "A"
    else:
        return "B"


class JobAnalyzer:
    def __init__(self, config: Config):
        self.client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        self.model = config.claude_model
        self.profile = load_profile()
        self._profile_text = yaml.dump(self.profile, default_flow_style=False)
        self._system_a = f"{SYSTEM_PROMPT_TRACK_A}\n\n--- CANDIDATE PROFILE ---\n{self._profile_text}"
        self._system_b = f"{SYSTEM_PROMPT_TRACK_B}\n\n--- CANDIDATE PROFILE ---\n{self._profile_text}"

    async def analyze_batch(self, jobs: list[Job], max_concurrent: int = 10) -> list[Job]:
        """Analyze a batch of jobs concurrently using dual-track scoring."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _analyze_with_limit(job: Job) -> Job:
            async with semaphore:
                try:
                    return await self._analyze_single(job)
                except Exception as e:
                    logger.warning(f"Analysis failed for '{job.title}' @ {job.company}: {e}")
                    job.score = 0
                    job.recommendation = Recommendation.SKIP
                    job.reasoning = f"Analysis failed: {str(e)[:100]}"
                    return job

        results = await asyncio.gather(*[_analyze_with_limit(j) for j in jobs])
        return list(results)

    async def _analyze_single(self, job: Job) -> Job:
        """Analyze a single job — routes to Track A, B, or both, takes best score."""
        track = _classify_track(job)

        if track == "BOTH":
            # Score on both tracks concurrently, take the higher score
            result_a, result_b = await asyncio.gather(
                self._score_with_track(job, "A"),
                self._score_with_track(job, "B"),
            )
            # Apply the winning track's result to the job
            winning = result_a if result_a.score >= result_b.score else result_b
            job.score = winning.score
            job.recommendation = winning.recommendation
            job.reasoning = winning.reasoning
            job.key_matches = winning.key_matches
            job.gaps = winning.gaps
            job.salary_estimate = winning.salary_estimate
            job.is_suspicious = winning.is_suspicious
            logger.info(f"  DUAL-TRACK A={result_a.score} B={result_b.score} → {winning.recommendation.value} ({winning.score}): {job.title} @ {job.company}")
        else:
            job = await self._score_with_track(job, track)
            logger.info(f"  Track{track} {job.recommendation.value} ({job.score}): {job.title} @ {job.company}")

        return job

    async def _score_with_track(self, job: Job, track: str) -> Job:
        """Run a single track's scoring against the job, return a scored Job copy."""
        import copy
        scored = copy.copy(job)

        age_info = f"\nPOSTED: {job.age_display}" if job.age_hours is not None else ""
        salary_info = f"\nSALARY: {job.salary_range}" if job.salary_range else ""

        user_message = f"""JOB: {job.title} @ {job.company}
LOCATION: {job.location}
SOURCE: {job.source}{age_info}{salary_info}
URL DOMAIN: {job.url_domain}
DESCRIPTION: {job.description[:2000]}"""

        system = self._system_a if track == "A" else self._system_b
        few_shot = FEW_SHOT_TRACK_A if track == "A" else FEW_SHOT_TRACK_B

        messages = [*few_shot, {"role": "user", "content": user_message}]

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=system,
            messages=messages,
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Claude response (Track {track}) for {job.title}: {raw[:200]}")
            scored.score = 50
            scored.recommendation = Recommendation.MAYBE
            scored.reasoning = f"Track {track} analysis returned non-JSON response"
            return scored

        scored.score = min(100, max(0, data.get("score", 0)))
        rec = data.get("recommendation", "SKIP").upper()
        scored.recommendation = Recommendation[rec] if rec in Recommendation.__members__ else Recommendation.SKIP
        scored.reasoning = data.get("reasoning", "")
        scored.key_matches = data.get("key_matches", [])
        scored.gaps = data.get("gaps", [])
        scored.salary_estimate = data.get("salary_estimate")
        scored.is_suspicious = data.get("is_suspicious", False)

        # Thresholds differ by track
        apply_threshold = 75 if track == "A" else 80
        maybe_threshold = 55 if track == "A" else 60

        if scored.score >= apply_threshold:
            scored.recommendation = Recommendation.APPLY
        elif scored.score >= maybe_threshold:
            scored.recommendation = Recommendation.MAYBE
        else:
            scored.recommendation = Recommendation.SKIP

        if scored.is_suspicious:
            scored.recommendation = Recommendation.SKIP
            logger.info(f"  Suspicious (Track {track}): {job.title} @ {job.company}")

        return scored
