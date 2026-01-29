import os
import sqlite3
from typing import List, Tuple, Optional
from datetime import datetime

from app.models import PredictionItem


def _connect(sqlite_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    return con


def init_db(sqlite_path: str) -> None:
    con = _connect(sqlite_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                score REAL NOT NULL
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);")
        con.commit()
    finally:
        con.close()


def insert_prediction(sqlite_path: str, prediction_id: str, filename: str, label: str, score: float) -> PredictionItem:
    created_at = datetime.utcnow().isoformat() + "Z"
    con = _connect(sqlite_path)
    try:
        con.execute(
            "INSERT INTO predictions (prediction_id, created_at, filename, label, score) VALUES (?, ?, ?, ?, ?)",
            (prediction_id, created_at, filename, label, float(score)),
        )
        con.commit()
        return PredictionItem(
            prediction_id=prediction_id,
            created_at=created_at,
            filename=filename,
            label=label,
            score=float(score),
        )
    finally:
        con.close()


def list_predictions(sqlite_path: str, limit: int, offset: int) -> Tuple[List[PredictionItem], int]:
    con = _connect(sqlite_path)
    try:
        total = con.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        rows = con.execute(
            """
            SELECT prediction_id, created_at, filename, label, score
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        items = [
            PredictionItem(
                prediction_id=r["prediction_id"],
                created_at=r["created_at"],
                filename=r["filename"],
                label=r["label"],
                score=float(r["score"]),
            )
            for r in rows
        ]
        return items, int(total)
    finally:
        con.close()


def get_prediction(sqlite_path: str, prediction_id: str) -> Optional[PredictionItem]:
    con = _connect(sqlite_path)
    try:
        row = con.execute(
            """
            SELECT prediction_id, created_at, filename, label, score
            FROM predictions
            WHERE prediction_id = ?
            """,
            (prediction_id,),
        ).fetchone()
        if not row:
            return None
        return PredictionItem(
            prediction_id=row["prediction_id"],
            created_at=row["created_at"],
            filename=row["filename"],
            label=row["label"],
            score=float(row["score"]),
        )
    finally:
        con.close()
