"""python -m mrail info MAP  |  python -m mrail merge OUT MAP [MAP ...]"""
import sys
from pathlib import Path
from . import MapFile


def main(argv):
    if len(argv) >= 2 and argv[0] == "info":
        m = MapFile(argv[1])
        n_full = sum(1 for v in m.anchors.values() if v[3] is None)
        print(f"{argv[1]}: {len(m.sessions)} runs, {sum(len(s) for s in m.sessions)} tokens, "
              f"{len(m.anchors)} anchors ({n_full} full, {len(m.anchors) - n_full} delta), "
              f"{sum(m.use.values())} uses, {m.size_mb():.2f} MB")
    elif len(argv) >= 3 and argv[0] == "merge":
        out = MapFile(argv[1])
        for p in argv[2:]:
            src = MapFile(p)
            for toks, g in zip(src.sessions, src.gen_start):
                out.add_session(toks, g)
        print(f"{argv[1]}: {len(out.sessions)} runs (anchors are not copied: they belong to the exact model state)")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
