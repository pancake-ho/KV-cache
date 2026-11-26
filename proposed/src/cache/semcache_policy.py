"""Adapt SemCache admission and eviction scores to native KV prefix entries."""

from collections import Counter, deque
from dataclasses import dataclass

from semcache.cache.admission import AdmissionPolicy
from semcache.cache.cache_metrics import CacheMetrics, normalize
from semcache.cache.eviction import EvictionPolicy


@dataclass
class Residency:
    last_reuse: int
    hits: int = 0


class SemCachePrefixPolicy:
    """Score whole prefixes by frequency, age and stored bytes.

    The OPT row-level attention signal is unavailable in Qwen's native path.
    Renormalize the remaining upstream weights. The query clock advances on
    lookup, while a reuse is counted only after the request executes.
    """

    name = "semcache_frequency_age_size"

    def __init__(self, history_requests: int = 100):
        if history_requests < 1:
            raise ValueError("SemCache history must contain at least one request")
        self.admission = AdmissionPolicy(alpha=5 / 7, beta=0, delta=2 / 7)
        self.eviction = EvictionPolicy(alpha=4 / 7, beta=0, gamma=2 / 7, delta=1 / 7)
        self.history_requests = history_requests
        self.clock = 0
        self.history = deque()
        self.frequency = Counter()
        self.resident: dict[str, Residency] = {}

    @staticmethod
    def _key(request, adapter_sha256):
        return (request.model_id, request.adapter_id, adapter_sha256,
                request.share_group, request.token_ids)

    def observe(self, request, adapter_sha256):
        self.clock += 1
        key = self._key(request, adapter_sha256)
        self.history.append((self.clock, key))
        self.frequency[key] += 1
        while self.history and self.history[0][0] <= self.clock - self.history_requests:
            _, old = self.history.popleft()
            self.frequency[old] -= 1
            if self.frequency[old] == 0:
                del self.frequency[old]

    def admit(self, request, adapter_sha256, size_bytes, entries):
        def metrics(req, fingerprint, size):
            return CacheMetrics(self.frequency[self._key(req, fingerprint)], 0, 0, size)

        candidate = metrics(request, adapter_sha256, size_bytes)
        population = [metrics(e.request, e.adapter_sha256, e.size_bytes) for e in entries.values()]
        return self.admission.admit(normalize(candidate, [*population, candidate]))

    def eviction_order(self, entries, candidate):
        """Rank the candidate and residents once for a single admission."""
        metrics = {}
        for key, entry in entries.items():
            state = self.resident[key]
            metrics[key] = CacheMetrics(state.hits, 0, self.clock - state.last_reuse, entry.size_bytes)
        metrics[candidate.request.request_id] = CacheMetrics(0, 0, 0, candidate.size_bytes)
        population = metrics.values()
        maxima = CacheMetrics(max(m.frequency for m in population), 0,
                              max(m.age for m in population), max(m.size for m in population))
        scores = {key: self.eviction.score(normalize(value, (maxima,)))
                  for key, value in metrics.items()}
        return sorted(scores, key=lambda key: (-scores[key], key))

    def insert(self, request_id: str):
        self.resident[request_id] = Residency(last_reuse=self.clock)

    def remove(self, request_id: str):
        del self.resident[request_id]

    def reuse(self, entry):
        state = self.resident[entry.request.request_id]
        state.hits += 1
        state.last_reuse = self.clock
        key = self._key(entry.request, entry.adapter_sha256)
        self.history.append((self.clock, key))
        self.frequency[key] += 1
