"""
rag.py — SEC EDGAR filing retrieval, chunking, embedding, and disk cache.

Flow per ticker (called once at backtest start or monthly):
  1. Look up CIK from SEC company_tickers.json
  2. Fetch most recent 10-Q and 10-K filing text via EDGAR full-text index
  3. Chunk at ~500 tokens, embed batch with text-embedding-3-small
  4. Pickle (chunks, embeddings) to .rag_cache/{ticker}_{YYYY-MM}.pkl

SEC requirement: User-Agent header must identify the app and contact email.
"""

from __future__ import annotations

import math
import pickle
import re
import time
from pathlib import Path

import requests

_HEADERS = {"User-Agent": "TickerAI mytspark03@gmail.com", "Accept-Encoding": "gzip, deflate"}
_EDGAR_BASE = "https://data.sec.gov"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CHUNK_TOKENS = 500
_CHARS_PER_TOKEN = 4  # rough approximation


class RAGStore:
    def __init__(self, api_key: str, cache_dir: str = ".rag_cache") -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(exist_ok=True)
        self._loaded: dict[str, tuple[list[str], list[list[float]]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_loaded(self, ticker: str, as_of_date: str) -> None:
        """Load from disk cache or fetch from EDGAR. as_of_date: 'YYYY-MM-DD'."""
        month_key = as_of_date[:7]  # YYYY-MM
        cache_key = f"{ticker}_{month_key}"
        if cache_key in self._loaded:
            return
        cache_path = self._cache_path(ticker, month_key)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self._loaded[cache_key] = pickle.load(f)
                return
            except Exception:
                pass
        self._fetch_and_index(ticker, cache_key, cache_path)

    def query(self, ticker: str, query_text: str, top_k: int = 3) -> list[str]:
        """Return top-k SEC filing chunks most similar to query_text."""
        cache_keys = [k for k in self._loaded if k.startswith(f"{ticker}_")]
        if not cache_keys:
            return []
        chunks, embeddings = self._loaded[cache_keys[-1]]
        if not chunks:
            return []
        query_emb = self._embed_single(query_text)
        scored = [
            (self._cosine_similarity(query_emb, emb), chunk)
            for emb, chunk in zip(embeddings, chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    # ------------------------------------------------------------------
    # Fetch and index
    # ------------------------------------------------------------------

    def _fetch_and_index(self, ticker: str, cache_key: str, cache_path: Path) -> None:
        try:
            cik = self._ticker_to_cik(ticker)
            if not cik:
                self._loaded[cache_key] = ([], [])
                return

            filing_texts = self._fetch_recent_filings(cik, forms=["10-Q", "10-K"], max_per_form=1)
            if not filing_texts:
                self._loaded[cache_key] = ([], [])
                return

            all_chunks: list[str] = []
            for text in filing_texts:
                all_chunks.extend(self._chunk_text(text))

            all_chunks = all_chunks[:200]

            if not all_chunks:
                self._loaded[cache_key] = ([], [])
                return

            embeddings = self._embed_batch(all_chunks)
            self._loaded[cache_key] = (all_chunks, embeddings)

            with open(cache_path, "wb") as f:
                pickle.dump((all_chunks, embeddings), f)
        except Exception as e:
            print(f"[RAGStore] Failed to index {ticker}: {e}")
            self._loaded[cache_key] = ([], [])

    def _ticker_to_cik(self, ticker: str) -> str | None:
        try:
            resp = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            ticker_upper = ticker.upper()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker_upper:
                    return str(entry["cik_str"]).zfill(10)
        except Exception as e:
            print(f"[RAGStore] CIK lookup failed: {e}")
        return None

    def _fetch_recent_filings(self, cik: str, forms: list[str], max_per_form: int = 1) -> list[str]:
        url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            sub = resp.json()
        except Exception as e:
            print(f"[RAGStore] Submissions fetch failed: {e}")
            return []

        filings = sub.get("filings", {}).get("recent", {})
        form_types = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        texts: list[str] = []
        counts: dict[str, int] = {f: 0 for f in forms}

        for form, acc, doc in zip(form_types, accessions, primary_docs):
            if form not in forms:
                continue
            if counts[form] >= max_per_form:
                continue
            acc_clean = acc.replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
            text = self._fetch_filing_text(doc_url)
            if text:
                texts.append(text)
                counts[form] += 1
            time.sleep(0.15)

        return texts

    def _fetch_filing_text(self, url: str) -> str:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            raw = resp.text
            # Strip HTML tags if present
            if "<html" in raw.lower() or "<HTML" in raw:
                raw = re.sub(r"<[^>]+>", " ", raw)
                raw = re.sub(r"&[a-z]+;", " ", raw)
            # Collapse whitespace
            raw = re.sub(r"\s+", " ", raw).strip()
            return raw[:500_000]
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Chunking and embedding
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str, max_tokens: int = _CHUNK_TOKENS) -> list[str]:
        max_chars = max_tokens * _CHARS_PER_TOKEN
        paragraphs = re.split(r"\n{2,}|\. {2,}", text)
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 1 <= max_chars:
                current = (current + " " + para).strip()
            else:
                if current:
                    chunks.append(current)
                if len(para) > max_chars:
                    for i in range(0, len(para), max_chars):
                        chunks.append(para[i: i + max_chars])
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)
        return chunks

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            try:
                resp = self._client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[t[:8000] for t in batch],
                )
                results.extend([item.embedding for item in resp.data])
            except Exception as e:
                print(f"[RAGStore] Embedding batch failed: {e}")
                results.extend([[]] * len(batch))
            time.sleep(0.05)
        return results

    def _embed_single(self, text: str) -> list[float]:
        try:
            resp = self._client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000],
            )
            return resp.data[0].embedding
        except Exception:
            return []

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _cache_path(self, ticker: str, month_year: str) -> Path:
        return self._cache_dir / f"{ticker}_{month_year}.pkl"
