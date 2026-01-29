from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def _ensure_parent(path: str) -> None:
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)


def _connect(sqlite_path: str) -> sqlite3.Connection:
    _ensure_parent(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(cur: sqlite3.Cursor, table_name: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table_name});")
    return {row["name"] for row in cur.fetchall()}


def init_db(sqlite_path: str) -> None:
    conn = _connect(sqlite_path)
    cur = conn.cursor()

    cols = _table_columns(cur, "predictions")
    if not cols:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL,
                image_relpath TEXT NOT NULL
            );
            """
        )
        conn.commit()
        cols = _table_columns(cur, "predictions")

    # Migrate old schema that used prediction_id instead of id.
    if "id" not in cols and "prediction_id" in cols:
        cur.execute(
            """
            CREATE TABLE predictions_new (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL,
                image_relpath TEXT NOT NULL
            );
            """
        )

        if "image_relpath" in cols:
            cur.execute(
                """
                INSERT INTO predictions_new (id, filename, label, score, created_at, image_relpath)
                SELECT prediction_id, filename, label, score, created_at, image_relpath
                FROM predictions;
                """
            )
        else:
            cur.execute(
                """
                INSERT INTO predictions_new (id, filename, label, score, created_at, image_relpath)
                SELECT prediction_id, filename, label, score, created_at, ''
                FROM predictions;
                """
            )

        cur.execute("DROP TABLE predictions;")
        cur.execute("ALTER TABLE predictions_new RENAME TO predictions;")
        conn.commit()
        cols = _table_columns(cur, "predictions")

    # Defensive migration for missing image_relpath.
    if "image_relpath" not in cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN image_relpath TEXT DEFAULT '';")
        conn.commit()

    conn.close()


def insert_prediction(sqlite_path: str, item: Dict) -> None:
    conn = _connect(sqlite_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (id, filename, label, score, created_at, image_relpath)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            item["id"],
            item["filename"],
            item["label"],
            float(item["score"]),
            item["created_at"],
            item["image_relpath"],
        ),
    )
    conn.commit()
    conn.close()


def list_predictions(sqlite_path: str, limit: int, offset: int) -> Tuple[List[Dict], int]:
    conn = _connect(sqlite_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as c FROM predictions;")
    total = int(cur.fetchone()["c"])

    cur.execute(
        """
        SELECT id, filename, label, score, created_at, image_relpath
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )

    items = []
    for row in cur.fetchall():
        pred_id = row["id"]
        items.append(
            {
                "id": pred_id,
                "filename": row["filename"],
                "label": row["label"],
                "score": float(row["score"]),
                "created_at": row["created_at"],
                "image_url": f"/api/predictions/{pred_id}/image",
                "image_relpath": row["image_relpath"],
            }
        )

    conn.close()
    return items, total


def get_prediction(sqlite_path: str, prediction_id: str) -> Dict | None:
    conn = _connect(sqlite_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, filename, label, score, created_at, image_relpath
        FROM predictions
        WHERE id = ?
        """,
        (prediction_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None

    pred_id = row["id"]
    return {
        "id": pred_id,
        "filename": row["filename"],
        "label": row["label"],
        "score": float(row["score"]),
        "created_at": row["created_at"],
        "image_url": f"/api/predictions/{pred_id}/image",
        "image_relpath": row["image_relpath"],
    }


def clear_predictions(sqlite_path: str) -> List[str]:
    """
    Remove do banco e retorna a lista de image_relpath (para apagar os arquivos).
    """
    conn = _connect(sqlite_path)
    cur = conn.cursor()

    cur.execute("SELECT image_relpath FROM predictions;")
    relpaths = [r["image_relpath"] for r in cur.fetchall() if r["image_relpath"]]

    cur.execute("DELETE FROM predictions;")
    conn.commit()
    conn.close()

    return relpaths
