"""SQLite storage: users, face embeddings, attendance."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np

import config


@contextmanager
def _conn():
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                vector BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE(user_id, date)
            );
            """
        )


# ---------- users ----------
def emp_id_exists(emp_id):
    with _conn() as con:
        return con.execute("SELECT 1 FROM users WHERE emp_id=?", (emp_id,)).fetchone() is not None


def add_user(name, emp_id, department, embeddings):
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO users(emp_id, name, department, created_at) VALUES(?,?,?,?)",
            (emp_id, name, department, datetime.now().strftime("%Y-%m-%d %H:%M")),
        )
        uid = cur.lastrowid
        con.executemany(
            "INSERT INTO embeddings(user_id, vector) VALUES(?,?)",
            [(uid, np.asarray(e, dtype=np.float32).tobytes()) for e in embeddings],
        )
    return uid


def delete_user(user_id):
    with _conn() as con:
        con.execute("DELETE FROM users WHERE id=?", (user_id,))


def get_users():
    with _conn() as con:
        rows = con.execute(
            """SELECT u.id, u.emp_id, u.name, u.department, u.created_at,
                      (SELECT COUNT(*) FROM embeddings e WHERE e.user_id=u.id) AS samples
               FROM users u ORDER BY u.name"""
        ).fetchall()
    return [dict(r) for r in rows]


def load_gallery():
    """Returns (users_dict, [(user_id, embedding_array), ...])."""
    with _conn() as con:
        users = {
            r["id"]: {"name": r["name"], "emp_id": r["emp_id"], "department": r["department"]}
            for r in con.execute("SELECT * FROM users")
        }
        rows = [
            (r["user_id"], np.frombuffer(r["vector"], dtype=np.float32))
            for r in con.execute("SELECT user_id, vector FROM embeddings")
        ]
    return users, rows


# ---------- attendance ----------
def marked_user_ids(day):
    with _conn() as con:
        return {r["user_id"] for r in con.execute("SELECT user_id FROM attendance WHERE date=?", (day,))}


def mark_attendance(user_id):
    """Marks today's attendance once. Returns {'time','status'} or None if already marked."""
    now = datetime.now()
    status = "Late" if now.strftime("%H:%M") > config.LATE_AFTER else "Present"
    try:
        with _conn() as con:
            con.execute(
                "INSERT INTO attendance(user_id, date, time, status) VALUES(?,?,?,?)",
                (user_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), status),
            )
        return {"time": now.strftime("%H:%M:%S"), "status": status}
    except sqlite3.IntegrityError:
        return None


def get_attendance(day=None):
    """Attendance rows for one date (YYYY-MM-DD), or all dates if day is None."""
    q = """SELECT a.date, a.time, u.emp_id, u.name, u.department, a.status
           FROM attendance a JOIN users u ON u.id=a.user_id"""
    args = ()
    if day:
        q += " WHERE a.date=?"
        args = (day,)
    q += " ORDER BY a.date DESC, a.time DESC"
    with _conn() as con:
        return [dict(r) for r in con.execute(q, args).fetchall()]
