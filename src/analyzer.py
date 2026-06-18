"""Claude Haiku job analyzer — three-track scoring aligned to competitive advantage.

Track A: HEOR / RWE / Health Economics     (40% of searches — largest moat)
Track B: Global Health Analytics Leadership (35% — executive + epidemiology)
Track C: Health AI specifically             (25% — health domain required)

Generic AI engineering roles (no health domain) are scored on Track C and
will score low on Domain Alignment, surfacing only at MAYBE/SKIP.
"""

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
# Org-category taxonomy — description-first, name as secondary validator
# ---------------------------------------------------------------------------
ORG_CATEGORIES = [
    ("Global Health Funder", "B1", [
        "gates foundation", "bill & melinda gates", "wellcome trust", "wellcome leap",
        "rockefeller foundation", "novo nordisk foundation", "packard foundation",
        "children's investment fund", "ciff", "skoll foundation", "macarthur foundation",
        "elma foundation", "hilton foundation", "bloomberg philanthropies",
        "open philanthropy", "mastercard foundation", "hewlett foundation",
    ]),
    ("Global Health Implementer", "B1", [
        "path ", " path,", "path.", "chai ", "clinton health access",
        "jhpiego", "fhi 360", "fhi360", "jsi ", "management sciences for health",
        "msh ", "ipas ", "engenderhealth", "villagereach", "living goods",
        "last mile health", "mothers2mothers", "amref", "dimagi", "medic mobile",
        "ona data", "muso ", "intrahealth", "marie stopes",
        "international planned parenthood", "ippf",
    ]),
    ("Research / Evaluation / Economics Firm", "B2", [
        "ihme", "idinsight", "mathematica", "abt global", "abt associates",
        "rti international", "j-pal", "jpal", "innovations for poverty action",
        "ipa ", "norc ", "analysis group", "rand corporation", "rand health",
        "urban institute", "vital strategies", "results for development",
        "palladium", "icf ", "icf international", "population council",
        "population services international", "psi ",
    ]),
    ("HEOR / RWE / Life Sciences", "A1", [
        "iqvia", "clarivate", "norstella", "flatiron health", "concertai",
        "aetion", "evidation", "parexel", "icon plc", "syneos", "cytel",
        "precision aq", "eversana", "analysis group",
        "ucb ", "abbvie", "roche", "novartis", "pfizer", "gsk",
        "sanofi", "merck", "johnson & johnson", "lilly", "eli lilly",
        "bristol myers", "astrazeneca", "regeneron", "biogen",
    ]),
    ("Health-Tech / Payer / Data", "A2", [
        "optum", "humana", "elevance", "cvs health", "aetna", "unitedhealth",
        "blue cross", "blue shield", "highmark", "kaiser permanente",
        "truveta", "vizient", "komodo health", "arcadia.io", "health catalyst",
        "tempus ai", "tempus ", "natera", "dandelion health", "veradigm",
        "om1 ", "aetion", "aetion.",
    ]),
    ("Government / Multilateral", "B2", [
        "usaid", "cdc foundation", "centers for disease control", " cdc ",
        "nih ", "national institutes of health", "hrsa", "cms ", "world bank",
        "unicef", " who ", "world health organization", "unfpa", "unaids",
        "gavi ", "global fund", "pan american health", "paho",
    ]),
    ("AI Lab / Public-Interest AI", "C1", [
        "anthropic", "openai", "google deepmind", "deepmind", "microsoft research",
        "meta ai", "ibm research", "stanford hai", "mit jameel clinic",
        "duke ai health", "coalition for health ai", "allen institute",
    ]),
    ("Enterprise AI / Consulting", "C2", [
        "deloitte", "accenture", "booz allen", "mckinsey", "kpmg", "pwc ",
        "ibm ", "nvidia", "microsoft", "amazon web services", " aws ",
        "databricks", "palantir", "scale ai", "perficient", "cognizant",
        "leidos", "saic ", "mitre ",
    ]),
]

# ---------------------------------------------------------------------------
# Description keyword signals for track routing
# ---------------------------------------------------------------------------
HEOR_SIGNALS = [
    r"\bheor\b", r"health econom", r"outcomes research", r"real.world evidence",
    r"\brwe\b", r"pharmacoepidemiol", r"comparative effectiveness",
    r"cost.effectiveness", r"value.based", r"market access", r"payer evidence",
    r"health technology assessment", r"\bhta\b", r"reimbursement evidence",
    r"claims.analytic", r"claims data", r"health utilization",
    r"medical cost", r"indirect treatment comparison", r"\bitc\b",
    r"network meta.analysis", r"\bnma\b", r"budget impact",
]

GLOBAL_HEALTH_SIGNALS = [
    r"global health", r"population health", r"public health", r"epidemiolog",
    r"health system", r"health equity", r"community health", r"primary care.*program",
    r"reproductive health", r"maternal health", r"child health", r"\brmnch\b",
    r"\bmnch\b", r"nutrition program", r"\bhiv\b", r"malaria.*program",
    r"tuberculosis.*program", r"neglected tropical", r"immunization program",
    r"impact evaluation", r"program evaluation", r"\bmel\b",
    r"monitoring.*evaluation", r"lst model", r"demographic.*health",
    r"dhs survey", r"\bhmis\b", r"\bdhis2\b", r"health information system",
    r"health systems strengthening", r"biostatistic", r"clinical trial.*health",
]

HEALTH_AI_SIGNALS = [
    r"clinical ai", r"healthcare ai", r"health ai", r"medical ai",
    r"clinical.*machine learning", r"clinical.*deep learning",
    r"clinical nlp", r"clinical.*llm", r"clinical decision support.*ai",
    r"health.*llm", r"population health.*machine learning",
    r"ai.*outcomes", r"outcomes.*ai", r"responsible ai.*health",
    r"ai.*clinical", r"clinical.*agentic", r"health.*rag",
    r"ai.*epidemiol", r"health.*generative ai",
]

# Generic AI signals (no health domain) — these are Track C at best, often SKIP
GENERIC_AI_SIGNALS = [
    r"langchain", r"langgraph", r"\bllm\b", r"large language model",
    r"generative ai", r"agentic", r"rag pipeline", r"vector database",
    r"llmops", r"\bmlops\b", r"model context protocol", r"\bmcp\b",
    r"prompt engineer", r"fine.tun.*model", r"ai architect",
    r"machine learning engineer", r"ml engineer", r"nemo guardrails",
    r"\bcrewai\b", r"\bautogen\b",
]


def _detect_org_category(company: str, description: str) -> tuple[str, str]:
    combined = (company or "").lower() + " " + (description or "")[:500].lower()
    for label, tier, patterns in ORG_CATEGORIES:
        for p in patterns:
            if p in combined:
                return label, tier

    desc_lower = (description or "").lower()
    heor_hits = sum(1 for s in HEOR_SIGNALS if re.search(s, desc_lower))
    gh_hits = sum(1 for s in GLOBAL_HEALTH_SIGNALS if re.search(s, desc_lower))
    hai_hits = sum(1 for s in HEALTH_AI_SIGNALS if re.search(s, desc_lower))

    if heor_hits >= 2:
        return "HEOR/RWE Sector (inferred)", "A1"
    if gh_hits >= 3:
        return "Global Health Sector (inferred)", "B2"
    if hai_hits >= 2:
        return "Health-AI Sector (inferred)", "A2"
    return "Unknown", "unknown"


def _classify_track(job: Job) -> str:
    """
    Route to Track A (HEOR/RWE), Track B (Global Health Leadership),
    Track C (Health AI), or combinations.

    Description drives routing. Org category breaks ties.
    Generic AI-only roles (no health domain) go to Track C and will score
    low on Domain Alignment — surfacing only if compensation is extraordinary.
    """
    desc = (job.description or "").lower()
    title = (job.title or "").lower()
    combined = title + " " + desc

    heor_hits = sum(1 for s in HEOR_SIGNALS if re.search(s, combined))
    gh_hits = sum(1 for s in GLOBAL_HEALTH_SIGNALS if re.search(s, combined))
    hai_hits = sum(1 for s in HEALTH_AI_SIGNALS if re.search(s, combined))
    gen_ai_hits = sum(1 for s in GENERIC_AI_SIGNALS if re.search(s, combined))

    _, org_tier = _detect_org_category(job.company or "", job.description or "")

    # Strong HEOR/RWE signal → Track A (possibly + B if global health too)
    if heor_hits >= 2:
        if gh_hits >= 2:
            return "AB"
        return "A"

    # Strong global health signal → Track B (possibly + C if health AI)
    if gh_hits >= 2:
        if hai_hits >= 2:
            return "BC"
        if org_tier in ("A1", "A2") and heor_hits >= 1:
            return "AB"
        return "B"

    # Health AI signal → Track C
    if hai_hits >= 2:
        return "C"

    # Generic AI only (no health domain) → Track C, will score low on domain
    if gen_ai_hits >= 2:
        return "C"

    # Org-tier tiebreaker
    if org_tier in ("A1",):
        return "A"
    if org_tier in ("B1", "B2"):
        return "B"
    if org_tier in ("C1", "C2"):
        return "C"

    # Default: Track C (will score low if no health domain in description)
    return "C"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TRACK_A = """You are a job matching assistant evaluating HEOR / RWE / Health Economics roles.

CANDIDATE IDENTITY (Track A lens):
Erick Yegon has a PhD in Epidemiology and 17+ years of experience in health outcomes research,
causal inference, impact evaluation, cost-effectiveness analysis, LiST modeling, and health
systems analytics. He has led evidence generation programs at the Global Director level.
He also builds production AI systems, which is a strong differentiator in HEOR/RWE roles
that increasingly require data science and AI capability.

THIS TRACK COVERS: HEOR, RWE, outcomes research, health economics, pharmacoepidemiology,
comparative effectiveness, cost-effectiveness, market access evidence, claims analytics,
value evidence & outcomes, health technology assessment, payer evidence, budget impact modeling.

CLASSIFICATION RULE: Read the description first. If the role involves generating health economics
evidence, modeling health outcomes, or producing RWE for regulatory or market access purposes —
even at an unfamiliar company — it belongs here. The org name is secondary.

ORG_CATEGORY context: Use to validate sector. A1 = HEOR/RWE orgs (IQVIA, Flatiron, Aetion,
Parexel, Cytel, Precision AQ, EVERSANA, pharma). A2 = health-tech/payer.
B2 = research firms that do health economics work.

Evaluate each job across 5 dimensions:
1. Domain Alignment (40%): HEOR, RWE, outcomes research, health economics, pharmacoepidemiology,
   causal inference, comparative effectiveness, cost-effectiveness, claims analytics, LiST modeling.
   PhD Epidemiology + causal inference here is rare — score this generously.
2. Leadership/Seniority Fit (25%): Director, VP, Principal, Senior Scientist, Lead.
   Penalise anything below senior individual contributor.
3. Technical Match (20%): R, Python, SAS, Stata, causal inference methods, propensity scoring,
   survival analysis, ITC/NMA, claims data, DHIS2, health AI is a strong bonus.
4. Org/Sector Fit (10%): A1 orgs (pharma, HEOR CROs) > A2 (health-tech/payer) > B2 (research
   firms) > unknown. Do not penalise an unknown org if the description is strong.
5. Remote/Location (5%): Fully remote, or Richmond KY / Seattle WA / Portland OR.

HEALTH/HEOR BONUS: Add +15 points when the role explicitly involves HEOR, RWE, outcomes research,
health economics, pharmacoepidemiology, comparative effectiveness, or cost-effectiveness analysis.

FRESHNESS BONUS: +5 if posted <24h ago, +3 if posted <48h ago.

Return ONLY valid JSON:
{
  "score": <0-100>,
  "track": "A",
  "recommendation": "APPLY" | "MAYBE" | "SKIP",
  "reasoning": "<2-3 sentences on HEOR/RWE/health economics fit>",
  "key_matches": ["<match1>", "<match2>"],
  "gaps": ["<gap1>"],
  "salary_estimate": "<range or null>",
  "is_suspicious": false,
  "suspicious_reason": null,
  "health_bonus_applied": <true|false>,
  "org_category_used": "<from ORG_CATEGORY field>"
}

SCORING THRESHOLDS (Track A — lower to catch niche HEOR/RWE roles):
- APPLY: score >= 72
- MAYBE: score 52-71
- SKIP: score < 52"""

SYSTEM_PROMPT_TRACK_B = """You are a job matching assistant evaluating Global Health Analytics & Research Leadership roles.

CANDIDATE IDENTITY (Track B lens):
Erick Yegon is a global health executive with 17+ years leading data science and analytics programs.
Former Global Director at Living Goods (Kenya). PhD Epidemiology. Expert in population health,
causal inference, impact evaluation, LiST modeling, DHS surveys, DHIS2/HMIS, MEL, and health
systems analytics. His credibility with Gates Foundation, USAID, PATH, and UNICEF-funded programs
is an irreplaceable differentiator. He also builds AI systems, which strengthens his profile at
research organisations adding AI capability.

THIS TRACK COVERS: Director/VP/Head of health analytics, epidemiology leadership, biostatistics
leadership, MEL/M&E director roles, impact evaluation leadership, global health research director,
population health science, health informatics, digital health leadership.

ORG_CATEGORY context: B1 = foundations + implementers (Gates, PATH, CHAI, Living Goods).
B2 = research/eval firms (IHME, IDinsight, Mathematica, RTI, Abt, J-PAL, ICF).
Government/multilaterals (WHO, USAID, UNICEF, World Bank) also B2.
A2 = health-tech doing population health work.

Evaluate each job across 5 dimensions:
1. Domain Alignment (35%): Global health, population health, epidemiology, impact evaluation, MEL,
   biostatistics, health systems, digital health, RMNCH, reproductive health, health equity.
   17 years + Living Goods + PhD here is irreplaceable — score very generously.
2. Leadership/Seniority Fit (30%): Director, VP, Head, Principal, Chief. Must involve programme
   or team ownership. Penalise anything below senior IC with no leadership scope.
3. Technical Match (20%): R, Python, Stata, DHIS2, DHS, causal inference, statistical modelling,
   Power BI, Tableau. AI/LLM skills are a meaningful bonus — not required.
4. Mission/Org Fit (10%): B1 > B2 > A2 > unknown. Unknown org with a clear global health mission
   in the description is still a strong fit — do not over-penalise name unfamiliarity.
5. Remote/Location (5%): Fully remote, or Richmond KY / Seattle WA / Portland OR.

GLOBAL HEALTH BONUS: Add +12 points when the description explicitly involves global health,
population health, epidemiology, reproductive health, RMNCH, MEL, impact evaluation, DHIS2,
DHS, health systems strengthening, health equity, or community health programs.

FRESHNESS BONUS: +5 if posted <24h ago, +3 if posted <48h ago.

Return ONLY valid JSON:
{
  "score": <0-100>,
  "track": "B",
  "recommendation": "APPLY" | "MAYBE" | "SKIP",
  "reasoning": "<2-3 sentences on global health / leadership fit>",
  "key_matches": ["<match1>", "<match2>"],
  "gaps": ["<gap1>"],
  "salary_estimate": "<range or null>",
  "is_suspicious": false,
  "suspicious_reason": null,
  "health_bonus_applied": <true|false>,
  "org_category_used": "<from ORG_CATEGORY field>"
}

SCORING THRESHOLDS (Track B — lower threshold for foundation/NGO roles):
- APPLY: score >= 72
- MAYBE: score 52-71
- SKIP: score < 52"""

SYSTEM_PROMPT_TRACK_C = """You are a job matching assistant evaluating Health AI roles.

CANDIDATE IDENTITY (Track C lens):
Erick Yegon is an AI engineer and architect (LangChain, LangGraph, RAG, multi-agent, MCP,
NeMo Guardrails, FastAPI) with a PhD in Epidemiology and 17+ years in global health. His health
domain expertise is his strongest differentiator among AI engineers. Roles that explicitly require
or value health, clinical, or population health domain knowledge are high-value targets.

THIS TRACK COVERS: Clinical AI, healthcare AI science, AI outcomes research, responsible AI
in healthcare, clinical NLP, health LLM engineering, population health ML, AI evaluation in
clinical settings.

IMPORTANT FILTER — What does NOT belong here:
Roles where health domain is completely absent (Netflix AI platform, JPMorgan AI engineer,
Roku agentic systems, generic SaaS AI engineer, etc.) should score LOW on Domain Alignment
(5-10/20) and will surface only at MAYBE if compensation is extraordinary. Do not inflate
scores for AI engineering roles where health is irrelevant to the work.

ORG_CATEGORY context: A1/A2 (HEOR/health-tech) doing AI = highest value.
B1/B2 (health orgs) adding AI capability = high value.
C1 (Anthropic, OpenAI, DeepMind, Microsoft Research) = high value due to AI safety/responsible AI fit.
C2 (generic enterprise AI, consulting) without health domain = low-medium value.
Unknown generic tech company = low value.

Evaluate each job across 5 dimensions:
1. Technical Match (30%): LangChain, LangGraph, RAG, agentic AI, MCP, LLM fine-tuning,
   multi-agent orchestration, FastAPI, Python, NeMo Guardrails, RAGAS, vector databases.
2. Domain Alignment (30%): Health domain requirement is the key filter.
   Clinical AI / healthcare AI / population health ML = 25-30 points.
   Health adjacent (pharma, insurance) = 15-20 points.
   No health domain (generic tech company) = 5-10 points — do not exceed this.
3. Leadership/Seniority Fit (20%): Senior, Staff, Principal, Director, Lead, Architect.
   Penalise junior or mid-level.
4. Org Quality (15%): A1/A2 health orgs doing AI > B1/B2 health orgs > C1 AI labs > C2 generic
   consulting > unknown generic tech. Generic tech with no health domain ≤ 8/15.
5. Remote/Location (5%): Fully remote, or Richmond KY / Seattle WA / Portland OR.

HEALTH AI BONUS: Add +10 points when the role explicitly combines AI/ML engineering with
healthcare data, clinical outcomes, population health, or responsible AI for health.

FRESHNESS BONUS: +5 if posted <24h ago, +3 if posted <48h ago.

Return ONLY valid JSON:
{
  "score": <0-100>,
  "track": "C",
  "recommendation": "APPLY" | "MAYBE" | "SKIP",
  "reasoning": "<2-3 sentences on health AI fit — explicitly state if health domain is absent>",
  "key_matches": ["<match1>", "<match2>"],
  "gaps": ["<gap1>"],
  "salary_estimate": "<range or null>",
  "is_suspicious": false,
  "suspicious_reason": null,
  "health_bonus_applied": <true|false>,
  "org_category_used": "<from ORG_CATEGORY field>"
}

SCORING THRESHOLDS (Track C):
- APPLY: score >= 80
- MAYBE: score 62-79
- SKIP: score < 62

Note: generic AI engineering roles without health domain should rarely exceed 65 points total,
meaning they surface as MAYBE only — never APPLY — unless compensation is extraordinary (>$400K TC)."""

# ---------------------------------------------------------------------------
# Few-shot examples — one per track
# ---------------------------------------------------------------------------
FEW_SHOT_TRACK_A = [
    {
        "role": "user",
        "content": """JOB: Senior HEOR Scientist @ IQVIA
LOCATION: Remote (US)
POSTED: 4 hours ago
ORG_CATEGORY: HEOR/RWE Life Sciences (A1)
DESCRIPTION: Lead real-world evidence studies using claims data, EHR, and registry data.
Design observational studies with propensity score matching and causal inference methods.
Collaborate with market access and payer teams. PhD in epidemiology or health economics
required. 8+ years experience in HEOR or outcomes research. R and Python required.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 98,
            "track": "A",
            "recommendation": "APPLY",
            "reasoning": "Near-perfect Track A match. PhD Epidemiology, causal inference, propensity scoring, RWE, claims data — all core strengths. IQVIA is the top-tier HEOR/RWE employer globally. Health bonus +15, freshness +5.",
            "key_matches": ["PhD Epidemiology", "Real World Evidence", "Causal Inference", "Propensity Scoring", "Claims Data", "R", "Python", "HEOR"],
            "gaps": [],
            "salary_estimate": "$140,000–$190,000",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": True,
            "org_category_used": "HEOR/RWE Life Sciences (A1)",
        }),
    },
    {
        "role": "user",
        "content": """JOB: Junior Data Analyst @ Unknown Corp
LOCATION: Remote
POSTED: 10 days ago
ORG_CATEGORY: Unknown
DESCRIPTION: Create dashboards. Excel. 0-2 years experience. $45,000.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 5,
            "track": "A",
            "recommendation": "SKIP",
            "reasoning": "Junior role (0-2 years) far below 17+ years executive experience. No HEOR/RWE content. Unknown company.",
            "key_matches": [],
            "gaps": ["HEOR/RWE domain", "Seniority level", "Company verifiability"],
            "salary_estimate": None,
            "is_suspicious": True,
            "suspicious_reason": "Unknown company, no HEOR content",
            "health_bonus_applied": False,
            "org_category_used": "Unknown",
        }),
    },
]

FEW_SHOT_TRACK_B = [
    {
        "role": "user",
        "content": """JOB: Director of Evidence & Learning @ IDinsight
LOCATION: Remote (US/Global)
POSTED: 8 hours ago
ORG_CATEGORY: Research / Evaluation / Economics Firm (B2)
DESCRIPTION: Lead the evidence and learning agenda for health programs across sub-Saharan Africa.
Drive impact evaluation, causal inference, and DHS-linked outcome monitoring. Manage a team of
10+ data scientists and epidemiologists. PhD in epidemiology, public health, or economics required.
12+ years experience. Python/R. AI/ML for health outcomes a strong plus.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 97,
            "track": "B",
            "recommendation": "APPLY",
            "reasoning": "Near-perfect Track B match. Leads evidence and learning at a top global health research firm. Causal inference, DHS, epidemiology, team leadership — all exact strengths. AI/ML is a bonus match. Global health bonus +12, freshness +5.",
            "key_matches": ["PhD Epidemiology", "Impact Evaluation", "Causal Inference", "DHS", "Team Leadership", "Python", "R", "Sub-Saharan Africa"],
            "gaps": [],
            "salary_estimate": "$150,000–$200,000",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": True,
            "org_category_used": "Research / Evaluation / Economics Firm (B2)",
        }),
    },
]

FEW_SHOT_TRACK_C = [
    {
        "role": "user",
        "content": """JOB: Senior Clinical AI Scientist @ Optum AI
LOCATION: Remote (US)
POSTED: 1 day ago
ORG_CATEGORY: Health-Tech / Payer / Data (A2)
DESCRIPTION: Build LLM-powered clinical decision support using RAG and agentic workflows.
Population health background strongly preferred. Python, LangChain, vector databases required.
8+ years. Clinical AI or health informatics experience required.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 95,
            "track": "C",
            "recommendation": "APPLY",
            "reasoning": "Strong Track C match at the health-AI intersection. Clinical AI + population health are required — exactly where PhD Epidemiology + LangChain/RAG skills converge. Optum is a top health-tech AI employer. Health AI bonus +10, freshness +3.",
            "key_matches": ["LangChain", "RAG", "Clinical AI", "Population Health", "Python", "Agentic Workflows", "Health Informatics"],
            "gaps": [],
            "salary_estimate": "$170,000–$220,000",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": True,
            "org_category_used": "Health-Tech / Payer / Data (A2)",
        }),
    },
    {
        "role": "user",
        "content": """JOB: Agentic AI Engineer @ Roku
LOCATION: Remote (US)
POSTED: 2 hours ago
ORG_CATEGORY: Unknown
DESCRIPTION: Build multi-agent AI systems for our streaming platform recommendation engine.
LangChain, LangGraph, vector databases, Python. 5+ years. No health domain mentioned.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "score": 52,
            "track": "C",
            "recommendation": "SKIP",
            "reasoning": "Strong technical match on AI skills but zero health domain relevance. Streaming/entertainment context provides no advantage from PhD Epidemiology or 17 years of health experience. Domain alignment score is low (5/30). Not a strategic fit.",
            "key_matches": ["LangChain", "LangGraph", "Python", "Multi-Agent"],
            "gaps": ["Health domain entirely absent", "No use of health expertise"],
            "salary_estimate": "$180,000–$240,000",
            "is_suspicious": False,
            "suspicious_reason": None,
            "health_bonus_applied": False,
            "org_category_used": "Unknown",
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
        self._system = {
            "A": f"{SYSTEM_PROMPT_TRACK_A}\n\n--- CANDIDATE PROFILE ---\n{self._profile_text}",
            "B": f"{SYSTEM_PROMPT_TRACK_B}\n\n--- CANDIDATE PROFILE ---\n{self._profile_text}",
            "C": f"{SYSTEM_PROMPT_TRACK_C}\n\n--- CANDIDATE PROFILE ---\n{self._profile_text}",
        }
        self._few_shot = {
            "A": FEW_SHOT_TRACK_A,
            "B": FEW_SHOT_TRACK_B,
            "C": FEW_SHOT_TRACK_C,
        }
        self._apply_threshold = {"A": 72, "B": 72, "C": 80}
        self._maybe_threshold = {"A": 52, "B": 52, "C": 62}

    async def analyze_batch(self, jobs: list[Job], max_concurrent: int = 10) -> list[Job]:
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

        return list(await asyncio.gather(*[_analyze_with_limit(j) for j in jobs]))

    async def _analyze_single(self, job: Job) -> Job:
        route = _classify_track(job)

        # Multi-track: score on each applicable track, take the best
        tracks_to_run = list(set(route.replace("AB", "AB").replace("BC", "BC")))
        if route == "AB":
            tracks_to_run = ["A", "B"]
        elif route == "BC":
            tracks_to_run = ["B", "C"]
        else:
            tracks_to_run = [route]

        if len(tracks_to_run) > 1:
            results = await asyncio.gather(*[self._score_with_track(job, t) for t in tracks_to_run])
            winning = max(results, key=lambda j: j.score)
            track_scores = " ".join(f"{t}={r.score}" for t, r in zip(tracks_to_run, results))
            logger.info(
                f"  MULTI-TRACK [{track_scores}] → "
                f"{winning.recommendation.value} ({winning.score}): {job.title} @ {job.company}"
            )
        else:
            winning = await self._score_with_track(job, tracks_to_run[0])
            logger.info(
                f"  Track{tracks_to_run[0]} {winning.recommendation.value} "
                f"({winning.score}): {job.title} @ {job.company}"
            )

        job.score = winning.score
        job.recommendation = winning.recommendation
        job.reasoning = winning.reasoning
        job.key_matches = winning.key_matches
        job.gaps = winning.gaps
        job.salary_estimate = winning.salary_estimate
        job.is_suspicious = winning.is_suspicious
        return job

    async def _score_with_track(self, job: Job, track: str) -> Job:
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

        messages = [*self._few_shot[track], {"role": "user", "content": user_message}]

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=self._system[track],
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

        if scored.score >= self._apply_threshold[track]:
            scored.recommendation = Recommendation.APPLY
        elif scored.score >= self._maybe_threshold[track]:
            scored.recommendation = Recommendation.MAYBE
        else:
            scored.recommendation = Recommendation.SKIP

        if scored.is_suspicious:
            scored.recommendation = Recommendation.SKIP

        return scored
