"""mrail: append-only maps of exact model runs (reference reader/writer, see SPEC.md).

Monorail: the map lies over the model. Known track is driven, not computed.

HCGM has no KV cache: its whole memory is the fixed-size Holo-Ward state (28 layers x 2 KV heads x
64 frontier + 128 roots, ~5.5 MB, independent of context length). So the complete computational state
after ANY prefix is a small, constant object. The monorail stores it.

  - Session  = every token sequence the model has lived through (prompt + its own answer + the token it
               would emit next). Written by the model itself: the end of every ride extends the track.
  - Anchor   = the model's exact state after seq[:P] (all layers), at block boundaries, prompt ends,
               every ~32 answer tokens and at the end of each ride.
  - Riding   = longest common prefix of the new context with ANY session:
               * whole context on the track  -> the model's own continuation is already known (greedy is
                 deterministic in its state): emit it, restore the anchor, compute NOTHING
               * context leaves the track     -> restore the last anchor before the switch point and
                 compute only from there (the Möbius part): every layer below that point is skipped.
               * after the known track ends   -> normal tree-verified decoding with the Möbius-Rail
                 drafts; the new stretch becomes track for the next ride.

File format (.mrail, append-only, own format):
  8 bytes magic b"MRAIL\\x01\\0\\0"
  records: u8 kind | u32 payload length | payload
    kind 1 SESSION: u32 sid | u32 n | u32 gen_start | n x u32 tokens   (tokens[gen_start:] = the model's own)
    kind 2 ANCHOR : u32 sid | u32 P | 8-byte prefix hash | u32 json length | json | raw layer arrays
    kind 3 USE    : 8-byte prefix hash | u32 hits          (usage weight of a track position)
    kind 4 DELTA  : u32 sid | u32 P | 8-byte hash | 8-byte parent hash | u32 json length | json |
                    per layer: row fields as (packed changed-row bitmask + changed rows), small fields raw
                    (anchor = parent anchor + changed rows; 3.3x smaller on consecutive anchors)
"""
from __future__ import annotations

import hashlib, json, mmap, struct, time, zlib
from pathlib import Path
import numpy as np

from .drafts import DraftRail as MoebiusRail

MAGIC = b"MRAIL\x03\x00\x00"
K_SESSION, K_ANCHOR, K_USE, K_DELTA = 1, 2, 3, 4
ROW_FIELDS = ("_frontier_keys", "_frontier_values", "_root_keys", "_root_values")   # runtime-specific; override per model
MAX_CHAIN = 8


def chain(tokens, h0: bytes = b"\x00" * 8):
    """Prefix hashes: out[P] = hash(tokens[:P]), out[0] = h0."""
    out = [h0]
    h = h0
    for t in tokens:
        h = hashlib.blake2b(h + int(t).to_bytes(4, "little"), digest_size=8).digest()
        out.append(h)
    return out


class MapFile:
    def __init__(self, path, nmax_rail: int = 6, limit: int | None = None, extra=()):
        """limit: index only the first `limit` rides. extra: [(path, limit)] of foreign maps whose rides are
        loaded as draft track only (never emitted as the model's own continuation, no anchors)."""
        self.path = Path(path)
        self.sessions: list[list[int]] = []
        self.pos_hash: dict[bytes, tuple[int, int]] = {}      # prefix hash -> (sid of the longest ride, P)
        self.gen_hash: dict[bytes, int] = {}                   # prefix hash -> sid whose continuation there is model output
        self.gen_start: list[int] = []
        self.anchors: dict[bytes, tuple[int, int, int]] = {}  # prefix hash -> (file offset, length, P)
        self.use: dict[bytes, int] = {}
        self._cache: dict[bytes, tuple] = {}                  # decoded anchors (small LRU)
        self.rail = MoebiusRail(None, nmax=nmax_rail)         # n-gram drafts over all rides (Möbius-Rail)
        self._mm = None
        self.limit = limit
        for xp, xl in extra:
            self._load_foreign(Path(xp), xl)
        if self.path.exists() and self.path.stat().st_size > len(MAGIC):
            self._load()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(MAGIC)

    # ---------- file ----------
    def _load(self):
        with self.path.open("rb") as f:
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mm = self._mm
        assert mm[:8] == MAGIC, "not a .mrail file"
        off, n_own = 8, 0
        while off + 5 <= len(mm):
            kind, n = struct.unpack_from("<BI", mm, off)
            p = off + 5
            if p + n > len(mm):
                break                                          # torn tail: ignore
            if kind == K_SESSION:
                sid, cnt, gen = struct.unpack_from("<III", mm, p)
                if self.limit is None or n_own < self.limit:
                    toks = np.frombuffer(mm, np.uint32, cnt, p + 12).tolist()
                    self._index_session(toks, gen)
                n_own += 1
            elif kind == K_ANCHOR:
                sid, P = struct.unpack_from("<II", mm, p)
                h = bytes(mm[p + 8:p + 16])
                self.anchors.setdefault(h, (p + 16, n - 16, P, None))
            elif kind == K_DELTA:
                sid, P = struct.unpack_from("<II", mm, p)
                h = bytes(mm[p + 8:p + 16]); parent = bytes(mm[p + 16:p + 24])
                self.anchors.setdefault(h, (p + 24, n - 24, P, parent))
            elif kind == K_USE:
                h = bytes(mm[p:p + 8]); self.use[h] = self.use.get(h, 0) + struct.unpack_from("<I", mm, p + 8)[0]
            off = p + n

    def _load_foreign(self, path: Path, limit):
        with path.open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        assert mm[:8] == MAGIC
        off, k = 8, 0
        while off + 5 <= len(mm):
            kind, n = struct.unpack_from("<BI", mm, off)
            p = off + 5
            if kind == K_SESSION and (limit is None or k < limit):
                _, cnt, _ = struct.unpack_from("<III", mm, p)
                toks = np.frombuffer(mm, np.uint32, cnt, p + 12).tolist()
                self._index_session(toks, len(toks))              # foreign: draft track only
                k += 1
            off = p + n
        mm.close()

    def _append(self, kind: int, payload: bytes) -> int:
        with self.path.open("ab") as f:
            start = f.tell()
            f.write(struct.pack("<BI", kind, len(payload)))
            f.write(payload)
        self._mm = None                                        # reopen lazily
        return start + 5

    def _map(self):
        if self._mm is None:
            with self.path.open("rb") as f:
                self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    # ---------- sessions ----------
    def _index_session(self, toks, gen):
        sid = len(self.sessions)
        self.sessions.append(toks)
        self.gen_start.append(gen)
        hs = chain(toks)
        for P in range(1, len(toks) + 1):
            cur = self.pos_hash.get(hs[P])
            if cur is None or len(self.sessions[cur[0]]) < len(toks):
                self.pos_hash[hs[P]] = (sid, P)
            if P >= gen and P < len(toks):
                g = self.gen_hash.get(hs[P])
                if g is None or len(self.sessions[g]) < len(toks):
                    self.gen_hash[hs[P]] = sid
        self.rail.add(toks)
        return sid

    def add_session(self, toks, gen_start: int) -> int:
        toks = [int(t) for t in toks]
        sid = self._index_session(toks, gen_start)
        self._append(K_SESSION, struct.pack("<III", sid, len(toks), gen_start) + np.asarray(toks, np.uint32).tobytes())
        return sid

    # ---------- anchors (exact model state) ----------
    def has_anchor(self, h: bytes) -> bool:
        return h in self.anchors

    def add_anchor(self, sid: int, P: int, h: bytes, snap, parent: bytes | None = None):
        """snap = (meta, layers). Stored as delta to parent when parent is an anchor with chain < MAX_CHAIN."""
        if h in self.anchors:
            return 0
        meta, layers = snap
        meta = dict(meta)
        pl = None
        if parent is not None and parent in self.anchors:
            pmeta, pl = self.get_layers(parent)
            if pmeta.get("chain", 0) + 1 >= MAX_CHAIN:
                pl = None
        meta["z"] = 1                                          # zlib-1 over the payload (empty roots/zero rows)
        if pl is None:
            meta["chain"] = 0
            js = json.dumps(meta, separators=(",", ":")).encode()
            raw = zlib.compress(b"".join(np.ascontiguousarray(L[k]).tobytes() for L in layers for k, _, _ in meta["arrays"]), 1)
            payload = struct.pack("<II", sid, P) + h + struct.pack("<I", len(js)) + js + raw
            p = self._append(K_ANCHOR, payload)
            self.anchors[h] = (p + 16, len(payload) - 16, P, None)
        else:
            meta["chain"] = pmeta.get("chain", 0) + 1
            js = json.dumps(meta, separators=(",", ":")).encode()
            parts = []
            for L, Lp in zip(layers, pl):
                for k, _, _ in meta["arrays"]:
                    a = np.ascontiguousarray(L[k])
                    if k in ROW_FIELDS:
                        ch = (a != Lp[k]).any(-1)                      # [heads, rows]
                        parts.append(np.packbits(ch.ravel()).tobytes())
                        parts.append(a[ch].tobytes())
                    else:
                        parts.append(a.tobytes())
            payload = struct.pack("<II", sid, P) + h + parent + struct.pack("<I", len(js)) + js + zlib.compress(b"".join(parts), 1)
            p = self._append(K_DELTA, payload)
            self.anchors[h] = (p + 24, len(payload) - 24, P, parent)
        self._cache[h] = (meta, layers)
        self._trim()
        return len(payload)

    def _trim(self):
        while len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))

    def get_layers(self, h: bytes):
        """-> (meta, [per-layer dict name -> array]) for anchor h (resolving delta chains)."""
        if h in self._cache:
            v = self._cache.pop(h); self._cache[h] = v
            return v
        off, n, P, parent = self.anchors[h]
        mm = self._map()
        (jl,) = struct.unpack_from("<I", mm, off)
        meta = json.loads(bytes(mm[off + 4:off + 4 + jl]))
        buf = memoryview(mm)[off + 4 + jl:off + n]
        if meta.get("z"):
            buf = memoryview(zlib.decompress(buf))
        layers, o = [], 0
        base = self.get_layers(parent)[1] if parent is not None else None
        for li in range(len(meta["scalars"])):
            L = {}
            for k, dt, shape in meta["arrays"]:
                item = np.dtype(dt).itemsize
                if base is not None and k in ROW_FIELDS:
                    nrows = shape[0] * shape[1]
                    nb = (nrows + 7) // 8
                    ch = np.unpackbits(np.frombuffer(buf[o:o + nb], np.uint8))[:nrows].astype(bool).reshape(shape[:2]); o += nb
                    arr = base[li][k].copy()
                    cnt = int(ch.sum()) * shape[2] * item
                    arr[ch] = np.frombuffer(buf[o:o + cnt], dtype=dt).reshape(-1, shape[2]); o += cnt
                else:
                    cnt = int(np.prod(shape)) * item
                    arr = np.frombuffer(buf[o:o + cnt], dtype=dt).reshape(shape).copy(); o += cnt
                L[k] = arr
            layers.append(L)
        self._cache[h] = (meta, layers)
        self._trim()
        return meta, layers

    def bump(self, h: bytes, hits: int = 1):
        self.use[h] = self.use.get(h, 0) + hits
        self._append(K_USE, h + struct.pack("<I", hits))

    # ---------- the ride ----------
    def locate(self, ctx):
        """-> (lcp, sid, best_anchor_hash, best_anchor_P, hashes of ctx)"""
        hs = chain(ctx)
        lcp, sid = 0, -1
        lo, hi = 0, len(ctx)                                   # prefix property is monotone: binary search
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if hs[mid] in self.pos_hash:
                lo = mid
            else:
                hi = mid - 1
        lcp = lo
        if lcp == len(ctx):
            sid = self.gen_hash.get(hs[lcp], -1)               # only a ride where the model itself spoke on from here
        best = 0
        for P in range(lcp, 0, -1):
            if hs[P] in self.anchors:
                best = P
                break
        return lcp, sid, (hs[best] if best else None), best, hs

    def size_mb(self) -> float:
        return self.path.stat().st_size / 1e6


# ---------- generic state helpers ----------
def snapshot_from_layers(layers, scalars=None):
    """layers: [ {name: np.ndarray} per layer ] -> (meta, layers) accepted by Monorail.add_anchor."""
    names = list(layers[0])
    meta = {"arrays": [[k, str(layers[0][k].dtype), list(layers[0][k].shape)] for k in names],
            "scalars": scalars or [{} for _ in layers]}
    return meta, layers
