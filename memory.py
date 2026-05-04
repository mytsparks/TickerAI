"""
memory.py — Persistent lesson store for the ReflectionAgent.

Store: JSON file, entries keyed by "{ticker}_{regime}_{trigger}".
Retrieval: cosine similarity on text-embedding-3-small embeddings.
Thread-safe: all file I/O protected by a threading.Lock.
"""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime
from pathlib import Path


class MemoryStore:
    def __init__(self, api_key: str, store_path: str = "memory_store.json",
                 base_url: str = "") -> None:
        from openai import OpenAI
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._use_embeddings = True
        self._path = Path(store_path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_lesson(
        self,
        ticker: str,
        regime: str,
        trigger: str,
        lesson: str,
    ) -> None:
        """Embed lesson and persist to the JSON store."""
        key = f"{ticker}_{regime}_{trigger}"
        embedding = self._embed(lesson)
        entry = {
            "lesson": lesson,
            "embedding": embedding,
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "regime": regime,
        }
        with self._lock:
            data = self._load()
            data[key] = entry
            self._save(data)

    def retrieve(self, ticker: str, query: str, top_k: int = 3) -> list[str]:
        """Return top-k lesson strings most similar to query, filtered to ticker."""
        query_emb = self._embed(query)
        with self._lock:
            data = self._load()

        candidates = [v for v in data.values() if v.get("ticker") == ticker]
        if not candidates:
            all_entries = list(data.values())
            if not all_entries:
                return []
            candidates = all_entries

        scored = [
            (self._cosine_similarity(query_emb, c["embedding"]), c["lesson"])
            for c in candidates
            if c.get("embedding")
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [lesson for _, lesson in scored[:top_k]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        if not self._use_embeddings:
            return []
        try:
            resp = self._client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000],
            )
            return resp.data[0].embedding
        except Exception:
            self._use_embeddings = False
            return []

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
