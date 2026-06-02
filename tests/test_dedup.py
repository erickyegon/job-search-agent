"""Tests for dedup engine."""

import pytest
import tempfile
import os
from src.dedup import DedupEngine
from src.models import Job, Recommendation, LinkStatus


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = DedupEngine(path)
    yield engine
    engine.close()
    os.unlink(path)


def make_job(title="AI Engineer", company="Anthropic", **kwargs):
    defaults = {
        "title": title,
        "company": company,
        "location": "Remote",
        "url": f"https://example.com/{title.replace(' ', '-').lower()}",
        "source": "test",
    }
    defaults.update(kwargs)
    return Job(**defaults)


class TestFilterNew:
    def test_all_new(self, db):
        jobs = [make_job("Job A"), make_job("Job B")]
        new = db.filter_new(jobs)
        assert len(new) == 2

    def test_filters_existing(self, db):
        job = make_job("Job A")
        job.recommendation = Recommendation.APPLY
        job.link_status = LinkStatus.VERIFIED
        db.save_jobs([job])

        new = db.filter_new([make_job("Job A"), make_job("Job B")])
        assert len(new) == 1
        assert new[0].title == "Job B"


class TestSaveAndRetrieve:
    def test_save_and_get_unnotified(self, db):
        job = make_job()
        job.score = 85
        job.recommendation = Recommendation.APPLY
        job.link_status = LinkStatus.VERIFIED
        db.save_jobs([job])

        results = db.get_unnotified("email", min_score=60)
        assert len(results) == 1
        assert results[0]["title"] == "AI Engineer"

    def test_min_score_filter(self, db):
        job = make_job()
        job.score = 50
        job.recommendation = Recommendation.SKIP
        job.link_status = LinkStatus.VERIFIED
        db.save_jobs([job])

        results = db.get_unnotified("email", min_score=60)
        assert len(results) == 0


class TestMarkNotified:
    def test_mark_email_notified(self, db):
        job = make_job()
        job.score = 85
        job.recommendation = Recommendation.APPLY
        job.link_status = LinkStatus.VERIFIED
        db.save_jobs([job])

        db.mark_notified([job], "email")
        results = db.get_unnotified("email", min_score=60)
        assert len(results) == 0


class TestCleanup:
    def test_cleanup_old(self, db):
        job = make_job()
        job.score = 85
        job.recommendation = Recommendation.APPLY
        job.link_status = LinkStatus.VERIFIED
        db.save_jobs([job])

        # Artificially age the record
        db.conn.execute(
            "UPDATE jobs SET first_seen = '2020-01-01T00:00:00'"
        )
        db.conn.commit()

        db.cleanup_old(max_age_days=14)
        results = db.get_unnotified("email", min_score=0)
        assert len(results) == 0
