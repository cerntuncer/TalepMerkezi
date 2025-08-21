import os
import time
import json
import sqlite3
from typing import List, Optional, Tuple
from dataclasses import dataclass

import requests
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
except Exception:
    pass

HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL", "https://api-inference.huggingface.co/models/facebook/bart-large-mnli")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
SQLITE_PATH = os.getenv("SQLITE_PATH", "/workspace/TalepMerkezi/app.db")
LABELS_PATH = os.getenv("LABELS_PATH", os.path.join(os.path.dirname(__file__), "labels.tr.txt"))
CANDIDATE_LABELS = os.getenv("CANDIDATE_LABELS")
WORKER_POLL_SECONDS = int(os.getenv("WORKER_POLL_SECONDS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

class ZeroShotResponse(BaseModel):
    labels: List[str]
    scores: List[float]

@dataclass
class TalepRow:
    id: int
    text: str


def read_labels() -> List[str]:
    if CANDIDATE_LABELS:
        return [x.strip() for x in CANDIDATE_LABELS.split(",") if x.strip()]
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return ["Destek", "Ödeme", "Ürün", "Şikayet", "İade", "Öneri", "Bilgi Talebi"]


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def fetch_unlabeled(conn: sqlite3.Connection, limit: int) -> List[TalepRow]:
    cur = conn.execute(
        """
        SELECT Id, Text
        FROM Talepler
        WHERE PredictedLabel IS NULL OR PredictedLabel = ''
        ORDER BY Id ASC
        LIMIT ?
        """,
        (limit,),
    )
    return [TalepRow(id=row["Id"], text=row["Text"]) for row in cur.fetchall()]


def update_label(conn: sqlite3.Connection, talep_id: int, label: str) -> None:
    conn.execute(
        "UPDATE Talepler SET PredictedLabel = ?, Status = ? WHERE Id = ?",
        (label, "InProgress", talep_id),
    )


class ApiError(Exception):
    pass


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True,
       retry=retry_if_exception_type(ApiError))
def classify_text(text: str, labels: List[str]) -> str:
    headers = {"Accept": "application/json"}
    if HUGGINGFACE_API_TOKEN:
        headers["Authorization"] = f"Bearer {HUGGINGFACE_API_TOKEN}"
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": labels, "multi_label": False}
    }
    resp = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code == 503:
        # Model loading on first call; treat as transient
        raise ApiError("Model loading, retrying")
    if resp.status_code >= 400:
        raise ApiError(f"API error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    # HF can return either dict or list depending on route
    if isinstance(data, list) and data:
        data = data[0]
    labels_resp = data.get("labels")
    scores_resp = data.get("scores")
    if not isinstance(labels_resp, list) or not isinstance(scores_resp, list) or not labels_resp:
        # Fallback to first label to avoid blocking
        return labels[0]
    best_idx = max(range(len(scores_resp)), key=lambda i: scores_resp[i])
    return labels_resp[best_idx]


def process_batch(conn: sqlite3.Connection) -> Tuple[int, int]:
    labels = read_labels()
    rows = fetch_unlabeled(conn, BATCH_SIZE)
    updated = 0
    for row in rows:
        try:
            label = classify_text(row.text, labels)
        except Exception as e:
            # Skip this row on persistent errors
            print(f"Classification failed for Id={row.id}: {e}")
            continue
        update_label(conn, row.id, label)
        updated += 1
    if updated:
        conn.commit()
    return len(rows), updated


def run_once() -> None:
    with get_db() as conn:
        total, updated = process_batch(conn)
        print(f"Processed={total} Updated={updated}")


def run_watch() -> None:
    print("Worker started. Polling for new rows…")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"Error in loop: {e}")
        time.sleep(WORKER_POLL_SECONDS)


if __name__ == "__main__":
    mode = os.getenv("MODE", "watch").lower()
    if mode == "once":
        run_once()
    else:
        run_watch()
