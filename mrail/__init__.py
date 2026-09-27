"""mrail — append-only maps of exact language-model runs (.mrail v3). See SPEC.md."""
from .core import MapFile, chain, snapshot_from_layers, MAGIC
from .drafts import DraftRail

Monorail = MapFile
__all__ = ["MapFile", "Monorail", "DraftRail", "chain", "snapshot_from_layers", "MAGIC"]
__version__ = "0.3.0"


def exact_answer(m: MapFile, ctx, max_new: int, stop=()):
    """The model's own continuation of ctx if ctx lies on a track, else None."""
    lcp, sid, *_ = m.locate(list(ctx))
    if lcp != len(ctx) or sid < 0:
        return None
    out = []
    for t in m.sessions[sid][len(ctx):len(ctx) + max_new]:
        out.append(t)
        if t in stop:
            break
    return out or None
