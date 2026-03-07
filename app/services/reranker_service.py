from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.config.settings import Settings
from app.models.schemas import SourceChunk

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class RerankerService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def candidate_pool_size(self, top_k: int, question: str) -> int:
        q_tokens = self._tokenize(question)
        query_factor = min(max(len(q_tokens), 3), 12)
        base = top_k * self._settings.rerank_candidate_multiplier
        adaptive = base + (query_factor * 2)
        return min(max(adaptive, self._settings.rerank_candidate_min), self._settings.rerank_candidate_max)

    def rerank(self, question: str, chunks: list[SourceChunk], final_k: int) -> list[SourceChunk]:
        if not chunks:
            return []
        if len(chunks) == 1:
            only = chunks[0]
            only.semantic_score = float(only.score)
            only.rerank_score = float(only.score)
            return [only]

        q_tokens = set(self._tokenize(question))
        sem_w, lex_w, seq_w = self._effective_weights(token_count=len(q_tokens))
        raw_scores = [float(c.score) for c in chunks]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        denom = (max_s - min_s) if (max_s - min_s) > 1e-9 else 1.0

        ranked: list[tuple[float, SourceChunk]] = []
        for chunk in chunks:
            semantic_norm = (float(chunk.score) - min_s) / denom
            lexical = self._lexical_overlap(q_tokens, chunk.text)
            sequence = SequenceMatcher(None, question.lower(), (chunk.text or "").lower()[:1000]).ratio()
            rerank_score = (
                (semantic_norm * sem_w)
                + (lexical * lex_w)
                + (sequence * seq_w)
            )
            # Penalize semantic-only hits when there is no lexical evidence for focused queries.
            if lexical == 0.0 and len(q_tokens) >= 3:
                rerank_score *= 0.85
            chunk.semantic_score = float(chunk.score)
            chunk.rerank_score = float(rerank_score)
            chunk.score = float(rerank_score)
            ranked.append((rerank_score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:final_k]]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in TOKEN_RE.findall(text or "")]

    def _lexical_overlap(self, query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        chunk_tokens = set(self._tokenize(text))
        if not chunk_tokens:
            return 0.0
        intersection = len(query_tokens.intersection(chunk_tokens))
        return intersection / len(query_tokens)

    def _effective_weights(self, token_count: int) -> tuple[float, float, float]:
        if token_count <= 6:
            return 0.5, 0.4, 0.1
        if token_count <= 12:
            return 0.58, 0.32, 0.1
        return (
            self._settings.rerank_semantic_weight,
            self._settings.rerank_lexical_weight,
            self._settings.rerank_sequence_weight,
        )
