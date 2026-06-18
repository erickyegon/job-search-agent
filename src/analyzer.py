"""Claude Haiku job analyzer — dual-track scoring for health executive + AI architect profiles."""

import asyncio
import logging
import json
import re
import yaml
from typing import Optional

import anthropic

from .models import Job, Recommendation
from .config import Config, load_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Org-category taxonomy
# Primary role: provide Claude with sector context so it can score
# Mission/Strategic Fit and Company Quality accurately even for
# organisations that are not on any fixed list.
# The description drives track routing; org category validates and boosts.
# ---------------------------------------------------------------------------

# Each entry: (category_label, tier, [name substrings])
# tier A1 = highest mission/prestige for Track A
# tier A2 = strong Track A
# tier B1 = highest prestige for Track B
# tier B2 = strong Track B
# tier AB = high value on both tracks
ORG_CATEGORIES = [
    (
        "Global Health Funder",
        "A1",
        [
            "gates foundation", "bill & melinda gates", "wellcome trust", "wellcome leap",
            "rockefeller foundation", "novo nordisk foundation", "packard foundation",
            "children's investment fund", "ciff", "skoll foundation", "macarthur foundation",
            "elma foundation", "conrad n. hilton", "hilton foundation", "bloomberg philanthropies",
            "open philanthropy", "mastercard foundation", "hewlett foundation",
        ],
    ),
    (
        "Global Health Implementer",
        "A1",
        [
            "path ", " path,", "path.", "chai ", "clinton health access",
            "jhpiego", "fhi 360", "fhi360", "jsi ", "management sciences for health",
            "msh ", "ipas ", "engenderhealth", "villagereach", "living goods",
            "last mile health", "mothers2mothers", "amref", "dimagi", "medic mobile",
            "ona data", "thinkmde", "muso ", "intrahealth", "marie stopes",
            "international planned parenthood", "ippf", "save the children",
            "helen keller", "nutrition international", "action against hunger",
        ],
    ),
    (
        "Research / Evaluation / Economics Firm",
        "A2",
        [
            "ihme", "idinsight", "mathematica", "abt global", "abt associates",
            "rti international", "j-pal", "jpal", "innovations for poverty action",
            "ipa ", "norc ", "air ", "american institutes for research",
            "rand corporation", "rand health", "urban institute", "vital strategies",
            "results for development", "r4d", "palladium", "icf ", "icf international",
            "measure evaluation", "jhpiego", "population council",
            "population services international", "psi ",
        ],
    ),
    (
        "Health-Tech / Payer / Provider Analytics",
        "A2",
        [
            "optum", "humana", "elevance", "cvs health", "aetna", "unitedhealth",
            "blue cross", "blue shield", "highmark", "kaiser permanente",
            "truveta", "vizient", "komodo health", "arcadia.io", "health catalyst",
            "olive ai", "waystar", "privia health", "novu health",
        ],
    ),
    (
        "Pharma / Life Sciences / RWE / HEOR",
        "A2",
        [
            "iqvia", "clarivate", "norstella", "flatiron health", "concertai",
            "aetion", "evidation", "parexel", "icon plc", "syneos", "ucb ",
            "abbvie", "roche", "novartis", "pfizer", "gsk", "glaxosmithkline",
            "sanofi", "merck", "johnson & johnson", "j&j", "lilly", "eli lilly",
            "bristol myers", "astrazeneca", "regeneron", "biogen", "moderna",
        ],
    ),
    (
        "Government / Multilateral",
        "A2",
        [
            "usaid", "cdc foundation", "centers for disease control", " cdc ",
            "nih ", "national institutes of health", "hrsa", "cms ", "world bank",
            "unicef", " who ", "world health organization", "unfpa", "unaids",
            "gavi ", "global fund", "pan american health", "paho",
            "department of health", "hhs ", "u.s. department",
        ],
    ),
    (
        "AI Lab / Public-Interest AI",
        "B1",
        [
            "anthropic", "openai", "google deepmind", "deepmind", "microsoft research",
            "meta ai", "meta llama", "ibm research", "stanford hai",
            "mit jameel clinic", "duke ai health", "coalition for health ai",
            "chai ", "allen institute", "ai2 ", "hugging face",
        ],
    ),
    (
        "Enterprise AI / Consulting",
        "B2",
        [
            "deloitte", "accenture", "booz allen", "mckinsey", "bain ", "bcg ",
            "kpmg", "pwc ", "ernst & young", "ey ", "ibm ", "nvidia", "microsoft",
            "amazon web services", " aws ", "google cloud", "databricks",
            "palantir", "scale ai", "weights & biases", "teradata",
            "perficient", "cognizant", "infosys", "wipro", "capgemini",
            "leidos", "saic ", "general dynamics it", "mitre ", "mitre corporation",
        ],
    ),
]

# Keyword triggers used for description-first track classification
# and for org-sector inference when the company name is unknown.
HEALTH_DESC_SIGNALS = [
    "population health", "global health", "public health", "epidemiolog",
    "health system", "health economics", "outcomes research", "real.world evidence",
    "rwe ", " heor", "cost.effectiveness", "health equity", "health disparit",
    "community health", "primary care", "reproductive health", "maternal health",
    "child health", "rmnch", " mnch", "nutrition program", "hiv ", "malaria",
    "tuberculosis", "neglected tropical", "immunization", "vaccination program",
    "impact evaluation", "causal inference", "program evaluation", "mel ",
    "monitoring.*evaluation", "learning.*evaluation", "lst model", "demographic.*health",
    "dhs survey", "hmis", "dhis2", "health information system", "digital health",
    "health informatics", "biostatistic", "clinical trial", "randomized controlled",
    "observational stud", "pharmacoepidemiol", "claims.*analytic", "claims data",
    "electronic health record", "ehr ", " emr ", "clinical data",
]

AI_DESC_SIGNALS = [
    "langchain", "langgraph", "llm", "large language model", "generative ai",
    "gen ai", "genai", "agentic", "rag pipeline", "retrieval.augmented",
    "vector database", "llmops", "mlops", "model context protocol", " mcp ",
    "prompt engineer", "fine.tun", "multimodal", "ai architect", "ai engineer",
    "machine learning engineer", "ml engineer", "nemo guardrails", "crewai",
    "autogen", "ai safety", "responsible ai", "ai guardrails", "foundation model",
    "transformer model", "diffusion model", "computer vision", "nlp engineer",
    "natural language processing engineer",
]


def _detect_org_category(company: str, description: str) -> tuple[str, str]:
    """
    Return (category_label, tier) for a company/org.
    Matches against company name first; falls back to description signals.
    Returns ("Unknown", "unknown") when no match found.
    """
    combined = (company or "").lower() + " " + (description or "")[:500].lower()

    for label, tier, patterns in ORG_CATEGORIES:
        for p in patterns:
            if p in combined:
                return label, tier

    # Fallback: infer sector from description keywords alone
    desc_lower = (description or "").lower()
    health_hits = sum(1 for s in HEALTH_DESC_SIGNALS if re.search(s, desc_lower))
    ai_hits = sum(1 for s in AI_DESC_SIGNALS if re.search(s, desc_lower))

    if health_hits >= 3:
        return "Health Sector (inferred)", "A2"
    if ai_hits >= 3:
        return "AI/Tech Sector (inferred)", "B2"
    if health_hits >= 1 and ai_hits >= 1:
        return "Health-AI Intersection (inferred)", "AB"

    return "Unknown", "unknown"


def _classify_track(job: Job) -> str:
    """
    Route a job to Track A (Global Health Executive), Track B (AI/Agentic),
    or BOTH (scored on both, best wins).

    Rule: the JOB DESCRIPTION drives routing; org category is a tiebreaker.
    Thresholds are deliberately permissive so the intersection always fires.
    """
    desc_lower = (job.description or "").lower()
    title_lower = (job.title or "").lower()
    combined = title_lower + " " + desc_lower

    health_hits = sum(1 for s in HEALTH_DESC_SIGNALS if re.search(s, combined))
    ai_hits = sum(1 for s in AI_DESC_SIGNALS if re.search(s, combined))

    _, org_tier = _detect_org_category(job.company or "", job.description or "")

    # Intersection: meaningful signals on both sides → always score both tracks
    if health_hits >= 2 and ai_hits >= 2:
        return "BOTH"

    # Strong health signal OR org is a health-primary category
    if health_hits >= 2 or org_tier in ("A1", "A2"):
        # But if there are also solid AI signals, still score both
        if ai_hits >= 2:
            return "BOTH"
        return "A"

    # Strong AI signal OR org is an AI/tech-primary category
    if ai_hits >= 2 or org_tier in ("B1", "B2"):
        # But if there are also solid health signals, score both
        if health_hits >= 2:
            return "BOTH"
        return "B"

    # Weak signals on both sides — tiebreak by org tier, else default to B
    if org_tier in ("A1", "A2"):
        return "A"
    return "B"


# ---------------------------------------------------------------------------
# System prompts — description-first, org-category as secondary signal
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TRACK_A = """You are a job matching assistant evaluating roles for a Global Health Executive.

CANDIDATE IDENTITY (Track A lens):
Erick Yegon is a health analytics executive with 17+ years leading data science and research programs
in global health. Former Global Director at Living Goods. PhD Epidemiology. Expert in population health,
causal inference, impact evaluation, LiST modeling, health economics, DHIS2/HMIS, and DHS surveys.
He also builds production AI systems (LangChain, LangGraph, RAG, multi-agent), but for this track the
health executive identity is primary.

CLASSIFICATION RULE (most important):
Read the job description first. Infer what sector and organisation type this role belongs to, even if
the company name is unfamiliar. Use these sector signals:
- Global health / epidemiology / impact evaluation / health systems / MEL / RMNCH / MNCH → health sector
- RWE / HEOR / outcomes research / pharmacoepidemiology / claims analytics → life sciences / health sector
- Digital health / population health / health equity / community health → health sector
- Clinical AI / health informatics / electronic health records → health-tech sector
Organisation name is secondary — an unknown NGO working on RMNCH programs scores as well as a named
foundation if the description confirms the mission.

ORG CATEGORY CONTEXT: You will receive an ORG_CATEGORY field in each job. Use it to validate sector
alignment, but do not penalise a job solely because its organisation is not a famous name.

Evaluate each job across 5 dimensions:
1. Domain Alignment (35%): Population health, epidemiology, global health, health economics, outcomes
   research, real-world evidence, health systems, digital health, impact evaluation, causal inference,
   MEL, HEOR, RWE, pharmacoepidemiology, clinical data science. Score generously — this is his moat.
2. Leadership/Seniority Fit (25%): Director, VP, Head, Principal, Chief, Senior IC. Penalise anything
   below senior individual contributor or that lacks programme/team ownership.
3. Technical Match (15%): Epidemiological methods, statistical modelling, R, Python, SQL, Stata,
   Power BI, Tableau, DHIS2, DHS. GenAI/LLM skills are a meaningful bonus.
4. Mission/Strategic Fit (15%): Public health impact, health equity, evidence-based programmes,
   foundation or government funding, research institute or implementer culture.
5. Remote/Location (10%): Fully remote, or in Richmond KY / Washington State / Oregon / Seattle /
   Portland. Penalise hard on-site outside these areas.

HEALTHCARE/GLOBAL HEALTH BONUS: Add +12 points when the description explicitly involves ANY of:
population health, epidemiology, public health, outcomes research, health economics, real-world evidence,
global health, community health, health systems, digital health, health equity, reproductive health,
maternal health, RMNCH, MNCH, LiST, DHS, HMIS, DHIS2, impact evaluation, MEL, HEOR, RWE, causal
inference, pharmacoepidemiology, claims analytics, clinical trials, health informatics.

FRESHNESS BONUS: +5 if posted <24 h ago, +3 if posted <48 h ago.

SUSPICIOUS JOB FLAGS: Vague with no responsibilities; unverifiable company and no domain; requests
personal/financial info upfront; compensation wildly out of range with no explanation.

Return ONLY valid JSON:
{
  "score": <0-100>,
  "track": "A",
  "recommendation": "APPLY" | "MAYBE" | "SKIP",
  "reasoning": "<2-3 sentences emphasising health/leadership fit and org sector>",
  "key_matches": ["<match1>", "<match2>"],
  "gaps": ["<gap1>"],
  "salary_estimate": "<range, or null if already listed>",
  "is_suspicious": false,
  "suspicious_reason": null,
  "health_bonus_applied": <true|false>,
  "org_category_used": "<category label from ORG_CATEGORY field>"
}

SCORING THRESHOLDS (Track A — lower to surface foundation/NGO/RWE roles):
- APPLY: score >= 75
- MAYBE: score 55-74
- SKIP: score < 55"""

SYSTEM_PROMPT_TRACK_B = """You are a job matching assistant evaluating roles for an AI/Agentic Architecture leader.

CANDIDATE IDENTITY (Track B lens):
Erick Yegon is an AI architect and engineer building production agentic systems: LangChain, LangGraph,
CrewAI, AutoGen, MCP, Advanced RAG, multi-agent orchestration, FastAPI, NeMo Guardrails, LLM fine-tuning,
RAGAS evaluation, FAISS, Qdrant. 17 years of experience including executive leadership. PhD Epidemiology
gives him a genuine edge in healthcare AI, responsible AI, and health LLM evaluation. He is NOT a junior
engineer — score seniority strictly against staff/senior/principal/director/architect level.

CLASSIFICATION RULE (most important):
Read the job description first. Infer what type of AI work is required, even if the company name is
unfamiliar. Use these signals:
- LLM / RAG / agentic / fine-tuning / multi-agent / MCP / vector DB → AI engineering
- AI safety / guardrails / responsible AI / red-teaming → AI safety
- MLOps / LLMOps / model serving / inference infra → AI infrastructure
- Healthcare AI / clinical NLP / health LLM / population health ML → health-AI intersection
Organisation name is secondary — a strong AI description at an unknown company can outperform a
weak AI description at a prestige name.

ORG CATEGORY CONTEXT: You will receive an ORG_CATEGORY field. Use it to inform the Company Quality
dimension score, but do not let an unfamiliar name alone suppress the score if the description is strong.
Tier B1 (AI Labs) > Tier B2 (Enterprise AI) > Tier AB (Health-AI crossover) > Tier A1/A2 (health orgs
building AI) > Unknown. An A1/A2 org doing genuine AI engineering work can still score well here.

Evaluate each job across 5 dimensions:
1. Technical Match (35%): LangChain, LangGraph, RAG, agentic AI, MCP, LLM fine-tuning, multi-agent
   systems, FastAPI, Python, AI guardrails, MLOps/LLMOps, vector databases, NeMo, RAGAS.
2. Leadership/Seniority Fit (20%): Senior, Staff, Principal, Director, Lead, Architect. Penalise
   junior or mid-level; reward roles with technical ownership or team leadership.
3. Domain Alignment (20%): Healthcare AI and population health are a natural differentiator. AI safety,
   responsible AI, and enterprise AI are strong. Health + AI intersection scores highest.
4. Company Quality (15%): Guided by ORG_CATEGORY. B1 tier (Anthropic, OpenAI, DeepMind, Microsoft
   Research) = 15/15. B2 tier (NVIDIA, Deloitte, Accenture, Databricks, Booz Allen) = 12/15.
   AB tier (Gates + AI, Optum AI, Humana AI) = 11/15. A1/A2 doing AI = 9/15. Unknown = 7/15.
   Do not score lower than 7 solely due to an unfamiliar name — let the description determine the rest.
5. Remote/Location (10%): Fully remote, or in Richmond KY / Washington State / Oregon / Seattle /
   Portland.

HEALTHCARE AI BONUS: Add +8 points when the role explicitly combines AI/ML engineering with healthcare
data, clinical outcomes, population health, health equity, or public health domains.

FRESHNESS BONUS: +5 if posted <24 h ago, +3 if posted <48 h ago.

SUSPICIOUS JOB FLAGS: Vague with no tech stack; unverifiable company and no URL domain; personal data
requested upfront; "AI company" with no verifiable product or mission.

Return ONLY valid JSON:
{
  "score": <0-100>,
  "track": "B",
  "recommendation": "APPLY" | "MAYBE" | "SKIP",
  "reasoning": "<2-3 sentences emphasising technical/AI fit and org tier>",
  "key_matches": ["<match1>", "<match2>"],
  "gaps": ["<gap1>"],
  "salary_estimate": "<range, or null if already listed>",
  "is_suspicious": false,
  "suspicious_reason": null,
  "health_bonus_applied": <true|false>,
  "org_category_used": "<category label from ORG_CATEGORY field>"
}

SCORING THRESHOLDS (Track B):
- APPLY: score >= 80
- MAYBE: score 60-79
- SKIP: score < 60"""

# ---------------------------------------------------------------------------
# Few-shot examples — one per track, including ORG_CATEGORY field
# ---------------------------------------------------------------------------
FEW_SHOT_TRACK_A = [
    {
        "role": "user",
        "content": """JOB: Director of Health Analytics @ Luminos Fund
LOCATION: Remote (US)
POSTED: 5 hours ago
ORG_CATEGORY: Global Health Implementer (A1)
DESCRIPTION: Lead population health analytics for our accelerated education programmes across
sub-Saharan Africa. Drive impact evaluation, causal inference, and DHS-linked reporting.
PhD in epidemiology or public health required. 10+ years experience. Python/R. AI/ML for health
outcomes strongly preferred.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 97,
            "track": "A",
            "recommendation": "APPLY",
            "reasoning": "Near-perfect Track A match. The description confirms global health, impact evaluation, DHS, causal inference, and population health — all core strengths. PhD Epidemiology is required. Org category A1 validates mission alignment. Health bonus +12, freshness +5.",
            "key_matches": ["PhD Epidemiology", "Impact Evaluation", "Causal Inference", "DHS", "Population Health", "Python", "R", "AI/ML for Health"],
            "gaps": [],
            "salary_estimate": "$150,000–$200,000",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": True,
            "org_category_used": "Global Health Implementer (A1)",
        }),
    },
    {
        "role": "user",
        "content": """JOB: Junior Data Analyst @ Unknown Corp
LOCATION: Remote
POSTED: 10 days ago
ORG_CATEGORY: Unknown
DESCRIPTION: Create dashboards. Excel. 0-2 years experience. $45,000. Send resume to personal email.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 8,
            "track": "A",
            "recommendation": "SKIP",
            "reasoning": "Junior role (0-2 years) far below 17+ years executive experience. No health domain. Unknown org with personal email is suspicious.",
            "key_matches": [],
            "gaps": ["Leadership level", "Domain alignment", "Company verifiability"],
            "salary_estimate": None,
            "is_suspicious": True,
            "suspicious_reason": "Unknown company, personal email for applications",
            "health_bonus_applied": False,
            "org_category_used": "Unknown",
        }),
    },
]

FEW_SHOT_TRACK_B = [
    {
        "role": "user",
        "content": """JOB: Senior GenAI Engineer @ Anthropic
LOCATION: Remote (US)
POSTED: 2 hours ago
ORG_CATEGORY: AI Lab / Public-Interest AI (B1)
DESCRIPTION: Build production agentic systems using Claude. LangChain, MCP, Agent SDK.
8+ years ML/AI, RAG pipelines, agentic AI orchestration, Python, FastAPI.
Plus: LLM fine-tuning, evaluation frameworks, AI safety.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 97,
            "track": "B",
            "recommendation": "APPLY",
            "reasoning": "Near-perfect Track B match. LangChain, MCP, agentic AI, RAG, Claude SDK are all expert skills. Tier B1 org (Anthropic). AI safety is a bonus skill match. Freshness +5.",
            "key_matches": ["LangChain", "MCP", "Agentic AI", "RAG", "Python", "FastAPI", "LLM Fine-Tuning", "AI Safety"],
            "gaps": [],
            "salary_estimate": "$250,000–$350,000 TC",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": False,
            "org_category_used": "AI Lab / Public-Interest AI (B1)",
        }),
    },
    {
        "role": "user",
        "content": """JOB: Senior AI Engineer — Clinical Decision Support @ HealthFlow AI
LOCATION: Remote (US)
POSTED: 1 day ago
ORG_CATEGORY: Health-AI Intersection (inferred)
DESCRIPTION: Build LLM-powered clinical decision support using RAG and agentic workflows.
Healthcare data experience strongly preferred. Python, LangChain, vector databases required.
8+ years. Population health or clinical AI background a strong plus.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 94,
            "track": "B",
            "recommendation": "APPLY",
            "reasoning": "Strong Track B match at the health-AI intersection. LangChain, RAG, agentic workflows are expert skills. Clinical AI and population health background is an exact differentiator — few AI engineers can match that. Org inferred as health-AI sector from description. Healthcare AI bonus +8, freshness +3.",
            "key_matches": ["LangChain", "RAG", "Agentic AI", "Python", "Vector Databases", "Population Health", "Clinical AI"],
            "gaps": [],
            "salary_estimate": "$170,000–$220,000",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": True,
            "org_category_used": "Health-AI Intersection (inferred)",
        }),
    },
]


# ---------------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------------
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
        """Route job to Track A, B, or both; apply best score."""
        track = _classify_track(job)

        if track == "BOTH":
            result_a, result_b = await asyncio.gather(
                self._score_with_track(job, "A"),
                self._score_with_track(job, "B"),
            )
            winning = result_a if result_a.score >= result_b.score else result_b
            job.score = winning.score
            job.recommendation = winning.recommendation
            job.reasoning = winning.reasoning
            job.key_matches = winning.key_matches
            job.gaps = winning.gaps
            job.salary_estimate = winning.salary_estimate
            job.is_suspicious = winning.is_suspicious
            logger.info(
                f"  DUAL-TRACK A={result_a.score} B={result_b.score} "
                f"→ {winning.recommendation.value} ({winning.score}): {job.title} @ {job.company}"
            )
        else:
            job = await self._score_with_track(job, track)
            logger.info(f"  Track{track} {job.recommendation.value} ({job.score}): {job.title} @ {job.company}")

        return job

    async def _score_with_track(self, job: Job, track: str) -> Job:
        """Score a single job on one track; return a scored copy."""
        import copy
        scored = copy.copy(job)

        org_label, org_tier = _detect_org_category(job.company or "", job.description or "")

        age_info = f"\nPOSTED: {job.age_display}" if job.age_hours is not None else ""
        salary_info = f"\nSALARY: {job.salary_range}" if job.salary_range else ""

        user_message = (
            f"JOB: {job.title} @ {job.company}\n"
            f"LOCATION: {job.location}\n"
            f"SOURCE: {job.source}{age_info}{salary_info}\n"
            f"URL DOMAIN: {job.url_domain}\n"
            f"ORG_CATEGORY: {org_label} ({org_tier})\n"
            f"DESCRIPTION: {job.description[:2000]}"
        )

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
        scored.recommendation = (
            Recommendation[rec] if rec in Recommendation.__members__ else Recommendation.SKIP
        )
        scored.reasoning = data.get("reasoning", "")
        scored.key_matches = data.get("key_matches", [])
        scored.gaps = data.get("gaps", [])
        scored.salary_estimate = data.get("salary_estimate")
        scored.is_suspicious = data.get("is_suspicious", False)

        # Enforce thresholds (overrides Claude's recommendation if inconsistent)
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
