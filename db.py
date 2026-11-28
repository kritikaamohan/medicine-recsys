"""Accounts and saved history for MediGuide (SQLite, standard library only).

Passwords are never stored. Only a random salt and a PBKDF2-HMAC-SHA256 hash are saved.
Note: on Streamlit Community Cloud the local file can reset when the app restarts, so
this is demo-level persistence. A hosted PostgreSQL database (e.g. Supabase) would be
the production upgrade.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("MEDIGUIDE_DB",
                              Path(__file__).resolve().parent / "mediguide.db"))
ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")


def _connect(path=None):
    conn = sqlite3.connect(str(path or DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(path=None):
    with closing(_connect(path)) as conn, conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                salt TEXT NOT NULL,
                pw_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
        """)


def _hash(password, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                               bytes.fromhex(salt_hex), ITERATIONS).hex()


def register(username, password, path=None):
    """Create an account. Returns (ok, message)."""
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers or underscore."
    if len(password or "") < 8:
        return False, "Password must be at least 8 characters."
    salt = secrets.token_hex(16)
    try:
        with closing(_connect(path)) as conn, conn:
            conn.execute("INSERT INTO users (username, salt, pw_hash, created_at) "
                         "VALUES (?, ?, ?, ?)",
                         (username, salt, _hash(password, salt),
                          datetime.now().isoformat(timespec="seconds")))
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created."


def login(username, password, path=None):
    """Return {'id', 'username'} if the credentials are right, otherwise None."""
    with closing(_connect(path)) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?",
                           ((username or "").strip(),)).fetchone()
    if row is None:
        _hash(password or "", secrets.token_hex(16))   # keep timing similar
        return None
    if hmac.compare_digest(_hash(password or "", row["salt"]), row["pw_hash"]):
        return {"id": row["id"], "username": row["username"]}
    return None


def save_check(user_id, res, path=None):
    with closing(_connect(path)) as conn, conn:
        conn.execute("INSERT INTO checks (user_id, created_at, result_json) VALUES (?, ?, ?)",
                     (user_id, datetime.now().isoformat(timespec="seconds"), json.dumps(res)))


def list_checks(user_id, path=None):
    """Newest first. Each item: {'id', 'created_at', 'result'}."""
    with closing(_connect(path)) as conn:
        rows = conn.execute("SELECT id, created_at, result_json FROM checks "
                            "WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()
    return [{"id": r["id"], "created_at": r["created_at"],
             "result": json.loads(r["result_json"])} for r in rows]


def delete_checks(user_id, path=None):
    with closing(_connect(path)) as conn, conn:
        conn.execute("DELETE FROM checks WHERE user_id = ?", (user_id,))
