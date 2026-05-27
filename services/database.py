import json
import sqlite3
from datetime import datetime
from pathlib import Path
from services.logger import get_logger

log     = get_logger(__name__)
DB_PATH = Path("campaigns.db")


def get_connection() -> sqlite3.Connection:
    conn             = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # access columns by name
    return conn


def init_db() -> None:
    """Create the campaigns table if it doesn't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id        TEXT,
                timestamp     TEXT,
                status        TEXT,
                user_prompt   TEXT,
                content       TEXT,
                hashtags      TEXT,
                full_post     TEXT,
                verdict       TEXT,
                issues        TEXT,
                denial_reason TEXT,
                platform      TEXT,
                sources       TEXT,
                articles      TEXT
            )
        """)
        conn.commit()
    log.debug("database ready")


def insert_campaign(data: dict) -> int:
    """Insert a campaign row and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO campaigns (
                run_id, timestamp, status, user_prompt,
                content, hashtags, full_post, verdict,
                issues, denial_reason, platform,
                sources, articles
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("run_id",        ""),
            data.get("timestamp",     datetime.now().isoformat()),
            data.get("status",        ""),
            data.get("user_prompt",   ""),
            data.get("content",       ""),
            json.dumps(data.get("hashtags",  [])),
            data.get("full_post",     ""),
            data.get("verdict",       ""),
            json.dumps(data.get("issues",    [])),
            data.get("denial_reason", ""),
            data.get("platform",      ""),
            data.get("sources",       "") if isinstance(data.get("sources"), str)
                else json.dumps(data.get("sources", [])),
            json.dumps(data.get("articles",  [])) if isinstance(data.get("articles"), list)
                else json.dumps([]),
        ))
        conn.commit()
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("insert_campaign failed to obtain row id")
        return row_id


def get_all_campaigns(limit: int = 50) -> list[dict]:
    """Return all campaigns newest first."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM campaigns
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_campaign_by_id(campaign_id: int) -> dict | None:
    """Return one campaign by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM campaigns WHERE id = ?",
            (campaign_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_all_campaigns_raw() -> list[dict]:
    """Return every campaign for FAISS index building."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM campaigns").fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for field in ("hashtags", "issues", "articles"):
        if isinstance(d.get(field), str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = []
    return d