"""Parse resumes (PDF/DOCX) and extract structured profile using Claude."""

import io
import json
import logging
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

PARSE_PROMPT = """You are a resume parser. Extract a structured job-seeker profile from the resume text below.

Return ONLY valid JSON in this exact format:
{
  "name": "Full Name",
  "years_experience": 5,
  "education": ["Degree — University"],
  "target_roles": ["Role 1", "Role 2"],
  "skills": {
    "expert": ["Skill 1", "Skill 2"],
    "proficient": ["Skill 3", "Skill 4"],
    "domain_expertise": ["Domain 1"]
  },
  "preferences": {
    "remote": true,
    "locations": ["City, State"],
    "exclude_keywords": ["internship", "junior", "entry level"]
  },
  "summary": "2-3 sentence professional summary"
}

Rules:
- Infer target_roles from their experience and job titles (senior-level if 8+ years)
- Categorize skills by proficiency based on how prominently they appear
- Include both technical and domain skills
- Set remote=true unless resume shows preference for on-site
- Be generous with skills — include frameworks, tools, languages, methodologies
- Include AI/ML safety, guardrails, NeMo skills if mentioned
- Include statistics, epidemiology, biostatistics if mentioned"""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
    except ImportError:
        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except ImportError:
            logger.error("No PDF library available. Install pdfplumber or PyPDF2.")
            return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        logger.error("python-docx not installed. Install it for DOCX support.")
        return ""


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract text from a resume file."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_bytes)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")


async def parse_resume(filename: str, file_bytes: bytes,
                       api_key: str) -> dict:
    """Parse a resume file and return a structured profile.

    Returns:
        dict with keys: profile (dict), resume_text (str)
    """
    resume_text = extract_text(filename, file_bytes)
    if not resume_text.strip():
        raise ValueError("Could not extract text from resume. Try a different format.")

    # Use Claude to parse into structured profile
    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=PARSE_PROMPT,
        messages=[{"role": "user", "content": resume_text[:8000]}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        profile = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse Claude profile response: {raw[:200]}")
        raise ValueError("Could not parse resume. Please try again.")

    return {"profile": profile, "resume_text": resume_text}
