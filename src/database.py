"""Multi-tenant database for SaaS job search agent."""

import sqlite3
import json
import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    telegram_chat_id TEXT DEFAULT '',
    -- Plan: free, pro, enterprise
    plan TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT DEFAULT '',
    stripe_subscription_id TEXT DEFAULT '',
    plan_expires_at TEXT DEFAULT NULL,
    -- Profile (parsed from resume)
    profile_json TEXT DEFAULT '{}',
    resume_text TEXT DEFAULT '',
    -- Status
    is_active INTEGER DEFAULT 1,
    email_verified INTEGER DEFAULT 0,
    verify_token TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Discount codes
CREATE TABLE IF NOT EXISTS discount_codes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    -- type: 'percent_off' or 'free_trial_days'
    discount_type TEXT NOT NULL,
    -- value: percentage (e.g. 50) or days (e.g. 14)
    value REAL NOT NULL,
    max_uses INTEGER DEFAULT NULL,
    used_count INTEGER DEFAULT 0,
    expires_at TEXT DEFAULT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);

-- Tracks which user used which discount
CREATE TABLE IF NOT EXISTS user_discounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    discount_id INTEGER NOT NULL,
    applied_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (discount_id) REFERENCES discount_codes(id),
    UNIQUE(user_id, discount_id)
);

-- Per-user jobs (extends existing jobs table)
CREATE TABLE IF NOT EXISTS user_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    job_hash TEXT NOT NULL,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    url TEXT,
    url_domain TEXT,
    source TEXT,
    description TEXT,
    posted_date TEXT,
    first_seen TEXT NOT NULL,
    score INTEGER DEFAULT 0,
    recommendation TEXT DEFAULT 'SKIP',
    reasoning TEXT,
    key_matches TEXT,
    gaps TEXT,
    salary_range TEXT,
    salary_estimate TEXT,
    notified_email INTEGER DEFAULT 0,
    notified_telegram INTEGER DEFAULT 0,
    is_suspicious INTEGER DEFAULT 0,
    link_status TEXT DEFAULT 'UNCHECKED',
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, job_hash)
);

CREATE INDEX IF NOT EXISTS idx_user_jobs_user ON user_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_user_jobs_score ON user_jobs(score);
CREATE INDEX IF NOT EXISTS idx_user_jobs_first_seen ON user_jobs(first_seen);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active, plan);
CREATE INDEX IF NOT EXISTS idx_discount_code ON discount_codes(code);
"""

# Plan limits
PLAN_LIMITS = {
    "free": {
        "max_alerts_per_day": 5,
        "email_notifications": True,
        "telegram_notifications": False,
        "max_search_queries": 5,
        "company_scrapers": False,
    },
    "pro": {
        "max_alerts_per_day": 50,
        "email_notifications": True,
        "telegram_notifications": True,
        "max_search_queries": 30,
        "company_scrapers": True,
    },
    "enterprise": {
        "max_alerts_per_day": 999,
        "email_notifications": True,
        "telegram_notifications": True,
        "max_search_queries": 50,
        "company_scrapers": True,
    },
}


class UserDB:
    def __init__(self, db_path: str = "jobs.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- User CRUD ----

    def create_user(self, email: str, name: str = "",
                    telegram_chat_id: str = "") -> dict:
        now = datetime.utcnow().isoformat()
        token = secrets.token_urlsafe(32)
        self.conn.execute(
            """INSERT INTO users (email, name, telegram_chat_id,
               verify_token, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (email, name, telegram_chat_id, token, now, now),
        )
        self.conn.commit()
        return self.get_user_by_email(email)

    def get_user(self, user_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return dict(row) if row else None

    def get_active_users(self) -> list[dict]:
        """Get all active users with valid plans."""
        rows = self.conn.execute(
            """SELECT * FROM users WHERE is_active = 1
               AND (plan_expires_at IS NULL OR plan_expires_at > ?)""",
            (datetime.utcnow().isoformat(),),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_user(self, user_id: int, **kwargs):
        kwargs["updated_at"] = datetime.utcnow().isoformat()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [user_id]
        self.conn.execute(f"UPDATE users SET {sets} WHERE id = ?", vals)
        self.conn.commit()

    def update_profile(self, user_id: int, profile: dict, resume_text: str = ""):
        self.update_user(
            user_id,
            profile_json=json.dumps(profile),
            resume_text=resume_text,
        )

    def set_plan(self, user_id: int, plan: str,
                 stripe_sub_id: str = "", expires_at: str = None):
        self.update_user(
            user_id, plan=plan,
            stripe_subscription_id=stripe_sub_id,
            plan_expires_at=expires_at,
        )

    def get_user_profile(self, user_id: int) -> dict:
        user = self.get_user(user_id)
        if not user:
            return {}
        return json.loads(user.get("profile_json", "{}"))

    def get_plan_limits(self, user_id: int) -> dict:
        user = self.get_user(user_id)
        if not user:
            return PLAN_LIMITS["free"]
        return PLAN_LIMITS.get(user["plan"], PLAN_LIMITS["free"])

    # ---- Per-user jobs ----

    def is_job_seen(self, user_id: int, job_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM user_jobs WHERE user_id = ? AND job_hash = ?",
            (user_id, job_hash),
        ).fetchone()
        return row is not None

    def save_user_job(self, user_id: int, job_data: dict):
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO user_jobs
               (user_id, job_hash, title, company, location, url, url_domain,
                source, description, posted_date, first_seen, score,
                recommendation, reasoning, key_matches, gaps, salary_range,
                salary_estimate, is_suspicious, link_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                user_id, job_data["job_hash"], job_data["title"],
                job_data["company"], job_data["location"], job_data["url"],
                job_data.get("url_domain", ""), job_data["source"],
                job_data.get("description", "")[:2000],
                job_data.get("posted_date"), now,
                job_data.get("score", 0), job_data.get("recommendation", "SKIP"),
                job_data.get("reasoning", ""),
                json.dumps(job_data.get("key_matches", [])),
                json.dumps(job_data.get("gaps", [])),
                job_data.get("salary_range"), job_data.get("salary_estimate"),
                1 if job_data.get("is_suspicious") else 0,
                job_data.get("link_status", "UNCHECKED"),
            ),
        )
        self.conn.commit()

    def get_user_unnotified(self, user_id: int, channel: str,
                            min_score: int = 60) -> list[dict]:
        col = f"notified_{channel}"
        rows = self.conn.execute(
            f"""SELECT * FROM user_jobs
                WHERE user_id = ? AND {col} = 0 AND score >= ?
                AND is_suspicious = 0
                ORDER BY score DESC, first_seen DESC""",
            (user_id, min_score),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_user_notified(self, user_id: int, job_hashes: list[str],
                           channel: str):
        col = f"notified_{channel}"
        for jh in job_hashes:
            self.conn.execute(
                f"UPDATE user_jobs SET {col} = 1 WHERE user_id = ? AND job_hash = ?",
                (user_id, jh),
            )
        self.conn.commit()

    def get_user_jobs(self, user_id: int, min_score: int = 0,
                      limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM user_jobs
               WHERE user_id = ? AND score >= ? AND is_suspicious = 0
               ORDER BY score DESC, first_seen DESC LIMIT ?""",
            (user_id, min_score, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def cleanup_old_user_jobs(self, max_age_days: int = 14):
        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
        c = self.conn.execute(
            "DELETE FROM user_jobs WHERE first_seen < ?", (cutoff,)
        )
        self.conn.commit()
        if c.rowcount:
            logger.info(f"Cleaned {c.rowcount} old user_jobs")

    # ---- Discount codes ----

    def create_discount(self, code: str, discount_type: str, value: float,
                        max_uses: int = None, expires_at: str = None) -> dict:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """INSERT INTO discount_codes
               (code, discount_type, value, max_uses, expires_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (code.upper(), discount_type, value, max_uses, expires_at, now),
        )
        self.conn.commit()
        return self.get_discount(code)

    def get_discount(self, code: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM discount_codes WHERE code = ? AND is_active = 1",
            (code.upper(),),
        ).fetchone()
        return dict(row) if row else None

    def validate_discount(self, code: str, user_id: int) -> dict:
        """Validate a discount code. Returns {valid, message, discount}."""
        disc = self.get_discount(code)
        if not disc:
            return {"valid": False, "message": "Invalid discount code"}

        # Check expiry
        if disc["expires_at"]:
            if datetime.fromisoformat(disc["expires_at"]) < datetime.utcnow():
                return {"valid": False, "message": "Discount code has expired"}

        # Check max uses
        if disc["max_uses"] and disc["used_count"] >= disc["max_uses"]:
            return {"valid": False, "message": "Discount code has been fully redeemed"}

        # Check if user already used it
        row = self.conn.execute(
            "SELECT 1 FROM user_discounts WHERE user_id = ? AND discount_id = ?",
            (user_id, disc["id"]),
        ).fetchone()
        if row:
            return {"valid": False, "message": "You have already used this code"}

        return {"valid": True, "message": "Discount applied!", "discount": disc}

    def apply_discount(self, user_id: int, code: str) -> dict:
        """Apply a discount code to a user."""
        result = self.validate_discount(code, user_id)
        if not result["valid"]:
            return result

        disc = result["discount"]
        now = datetime.utcnow().isoformat()

        # Record usage
        self.conn.execute(
            "INSERT INTO user_discounts (user_id, discount_id, applied_at) VALUES (?,?,?)",
            (user_id, disc["id"], now),
        )
        self.conn.execute(
            "UPDATE discount_codes SET used_count = used_count + 1 WHERE id = ?",
            (disc["id"],),
        )

        # Apply the discount
        if disc["discount_type"] == "free_trial_days":
            expires = datetime.utcnow() + timedelta(days=int(disc["value"]))
            self.update_user(user_id, plan="pro", plan_expires_at=expires.isoformat())
            result["message"] = f"Free trial activated! {int(disc['value'])} days of Pro access."
        # percent_off is handled at checkout time by Stripe
        else:
            result["message"] = f"{int(disc['value'])}% off applied at checkout!"

        self.conn.commit()
        return result

    def close(self):
        self.conn.close()
