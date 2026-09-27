import numpy as np, tempfile, os
from pathlib import Path
import mrail


def test_roundtrip_exact_answer_and_anchor():
    d = Path(tempfile.mkdtemp()); p = d / "t.mrail"
    m = mrail.MapFile(p)
    run = [1, 5, 7, 9, 11, 13, 2]
    m.add_session(run, gen_start=3)                              # prompt [1,5,7], answer [9,11,13,2]
    layers = [{"_frontier_keys": np.arange(2 * 4 * 8, dtype=np.uint16).reshape(2, 4, 8),
               "_frontier_values": np.ones((2, 4, 8), np.uint16), "_root_keys": np.zeros((2, 4, 8), np.uint16),
               "_root_values": np.zeros((2, 4, 8), np.uint16), "_counts": np.arange(3)}]
    h3 = mrail.chain(run[:3])[-1]
    m.add_anchor(0, 3, h3, mrail.snapshot_from_layers(layers))
    layers2 = [{k: v.copy() for k, v in layers[0].items()}]; layers2[0]["_frontier_keys"][1, 2] = 77
    h5 = mrail.chain(run[:5])[-1]
    m.add_anchor(0, 5, h5, mrail.snapshot_from_layers(layers2), parent=h3)
    m2 = mrail.MapFile(p)                                          # reopen
    assert mrail.exact_answer(m2, [1, 5, 7], 10, stop={2}) == [9, 11, 13, 2]
    assert mrail.exact_answer(m2, [1, 5], 10) is None             # prompt part is not model output
    meta, L = m2.get_layers(h5)
    assert (L[0]["_frontier_keys"] == layers2[0]["_frontier_keys"]).all()
    assert m2.anchors[h5][3] == h3                                # stored as delta
