"""Möbius-Rail: persistent route memory that proposes *trees* of continuations; the model decides.

- Every token that passes through the model (prompts and the model's own answers) is appended to the rail
  log. The rail is therefore written by the model itself: its end flows back into its start (Möbius).
- Rails are joined by shared contexts, not by fixed sequences: an index maps every n-gram (n = 1..NMAX) to the
  positions that followed it, so any matching context — in this conversation or any earlier session — is an
  entry point, and a shorter match is a switch point onto a different rail.
- Drafting builds a weighted trie of continuations from several entry points and context lengths, and returns
  the best-first top-B nodes as a tree. Nothing is ever emitted from the rail: the full model verifies the
  tree in one ANE pass and only tokens the model itself would produce are accepted. Where the model leaves
  every branch, it writes a new rail.
"""
from __future__ import annotations

import heapq
from pathlib import Path
import numpy as np


class DraftRail:
    def __init__(self, path: str | Path | None = None, nmax: int = 6, max_entries: int = 24, depth: int = 24):
        self.path = Path(path) if path else None
        self.nmax, self.max_entries, self.depth = nmax, max_entries, depth
        self.log: list[int] = []
        self.idx: list[dict] = [dict() for _ in range(nmax + 1)]
        # thought rails: SimHash of the model's final hidden state at a log position -> continuation after it
        self.planes = np.random.default_rng(1234).standard_normal((24, 1536)).astype(np.float32)
        self.sidx: list[dict] = [dict(), dict()]               # 24-bit exact state, 12-bit coarse state
        self.state_w = 6.0 ** 3
        if self.path and self.path.exists():
            self.add(np.fromfile(self.path, dtype=np.uint32).tolist())
        self._saved = len(self.log)

    def add(self, tokens):
        log, idx = self.log, self.idx
        for t in tokens:
            log.append(int(t))
            L = len(log)
            for n in range(1, min(self.nmax, L) + 1):
                key = tuple(log[L - n:L])
                idx[n].setdefault(key, []).append(L)          # continuation starts at L

    def _hash(self, hidden):
        bits = (self.planes @ np.asarray(hidden, np.float32)) > 0
        full = int(np.packbits(bits).view(">u4")[0] >> 8) if False else int("".join("1" if b else "0" for b in bits), 2)
        return full, full >> 12

    def add_state(self, log_index: int, hidden):
        """The model's hidden state after reading log[log_index] (it predicts log[log_index + 1])."""
        if hidden is None or not np.isfinite(hidden).all():
            return
        full, coarse = self._hash(hidden)
        self.sidx[0].setdefault(full, []).append(log_index + 1)
        self.sidx[1].setdefault(coarse, []).append(log_index + 1)

    def save(self):
        if not self.path or len(self.log) == self._saved:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as f:
            np.asarray(self.log[self._saved:], dtype=np.uint32).tofile(f)
        self._saved = len(self.log)

    def tree(self, ctx: list[int], budget: int, state=None) -> list[tuple[int, int]]:
        """Nodes (token, parent) of a draft tree below the root ctx[-1]; parent -1 = root.
        state: hidden vector that predicted ctx[-1] -> thought-rail entries whose continuation starts with ctx[-1]."""
        log, L = self.log, len(self.log)
        trie: dict = {}                                        # node -> [weight, children dict]
        root = [0.0, {}]
        if state is not None:
            full, coarse = self._hash(state)
            for level, key, w in ((0, full, self.state_w), (1, coarse, self.state_w / 8)):
                occ = self.sidx[level].get(key)
                if not occ:
                    continue
                used = 0
                for pos in reversed(occ):
                    if pos >= L or log[pos] != ctx[-1]:       # continuation must start with the current token
                        continue
                    node, decay = root, 1.0
                    for t in log[pos + 1:pos + 1 + self.depth]:
                        child = node[1].get(t)
                        if child is None:
                            child = node[1][t] = [0.0, {}]
                        child[0] += w * decay
                        node, decay = child, decay * 0.92
                    used += 1
                    if used >= self.max_entries:
                        break
        for n in range(min(self.nmax, len(ctx)), 0, -1):
            occ = self.idx[n].get(tuple(ctx[-n:]))
            if not occ:
                continue
            w_n = 4.0 ** n
            used = 0
            for pos in reversed(occ):
                if pos >= L:
                    continue
                cont = log[pos:pos + self.depth]
                recency = 1.0 + pos / max(L, 1)
                node, decay = root, 1.0
                for t in cont:
                    child = node[1].get(t)
                    if child is None:
                        child = node[1][t] = [0.0, {}]
                    child[0] += w_n * recency * decay
                    node, decay = child, decay * 0.92
                used += 1
                if used >= self.max_entries:
                    break
        out: list[tuple[int, int]] = []
        heap = [(-c[0], t, id(c), -1, c) for t, c in root[1].items()]
        heapq.heapify(heap)
        while heap and len(out) < budget:
            negw, t, _, parent, c = heapq.heappop(heap)
            out.append((t, parent))
            me = len(out) - 1
            for t2, c2 in c[1].items():
                heapq.heappush(heap, (-c2[0], t2, id(c2), me, c2))
        return out
