"""Tests for data models."""

import pytest
from datetime import datetime, timedelta
from src.models import Job, Recommendation, LinkStatus


def make_job(**kwargs):
    defaults = {
        "title": "Senior AI Engineer",
        "company": "Anthropic",
        "location": "Remote",
        "url": "https://anthropic.com/careers/123",
        "source": "anthropic_careers",
    }
    defaults.update(kwargs)
    return Job(**defaults)


class TestJobHash:
    def test_same_job_same_hash(self):
        j1 = make_job(title="AI Engineer", company="Anthropic", location="Remote")
        j2 = make_job(title="AI Engineer", company="Anthropic", location="Remote")
        assert j1.job_hash == j2.job_hash

    def test_case_insensitive(self):
        j1 = make_job(title="AI Engineer", company="ANTHROPIC", location="REMOTE")
        j2 = make_job(title="ai engineer", company="anthropic", location="remote")
        assert j1.job_hash == j2.job_hash

    def test_different_title_different_hash(self):
        j1 = make_job(title="AI Engineer")
        j2 = make_job(title="ML Engineer")
        assert j1.job_hash != j2.job_hash

    def test_whitespace_stripped(self):
        j1 = make_job(title=" AI Engineer ", company=" Anthropic ")
        j2 = make_job(title="AI Engineer", company="Anthropic")
        assert j1.job_hash == j2.job_hash


class TestJobAge:
    def test_age_hours(self):
        job = make_job()
        job.posted_date = datetime.utcnow() - timedelta(hours=5)
        assert job.age_hours == 5

    def test_age_display_hours(self):
        job = make_job()
        job.posted_date = datetime.utcnow() - timedelta(hours=3)
        assert "3h ago" in job.age_display

    def test_age_display_days(self):
        job = make_job()
        job.posted_date = datetime.utcnow() - timedelta(days=2)
        assert "2d ago" in job.age_display

    def test_age_none_when_no_date(self):
        job = make_job()
        assert job.age_hours is None
        assert job.age_display == "Unknown"


class TestJobDefaults:
    def test_default_recommendation(self):
        job = make_job()
        assert job.recommendation == Recommendation.SKIP

    def test_default_link_status(self):
        job = make_job()
        assert job.link_status == LinkStatus.UNCHECKED

    def test_to_dict(self):
        job = make_job(title="Test", company="Co", location="Remote")
        d = job.to_dict()
        assert d["title"] == "Test"
        assert d["company"] == "Co"
        assert "job_hash" in d
