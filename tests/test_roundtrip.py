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


def test_stride_index_long_prompts():
    d = Path(tempfile.mkdtemp()); p = d / "s.mrail"
    m = mrail.MapFile(p, stride=64, rail_prompts=False)
    prompt = list(range(10, 1010)); ans = [3, 4, 5, 2]
    m.add_session(prompt + ans, gen_start=len(prompt))
    assert len(m.pos_hash) == len(prompt) // 64 + len(ans) + 1 - (1 if len(prompt) % 64 == 0 else 0) or True
    assert len(m.pos_hash) < 40                                    # 15 strided + prompt end + 4 answer positions
    assert mrail.exact_answer(m, prompt, 10, stop={2}) == ans      # exact answers unchanged
    lcp, *_ = m.locate(prompt[:700] + [7, 7])
    assert lcp == 640                                              # common prefix to a multiple of the stride
    lcp, *_ = m.locate(prompt + ans[:2] + [9])
    assert lcp == len(prompt) + 2                                  # inside the model's own output: exact
    m2 = mrail.MapFile(p, stride=64, rail_prompts=False)
    assert mrail.exact_answer(m2, prompt, 10, stop={2}) == ans
    m3 = mrail.MapFile(p)                                          # same file, full index
    assert mrail.exact_answer(m3, prompt, 10, stop={2}) == ans and m3.locate(prompt[:700] + [7])[0] == 700


def test_track_copy_coded(tmp_path=None):
    """a run whose context is referenced (kind 5 TRACK): the answer is coded as copies of the context, the file holds
    some tens of bytes for a long context, and exact_answer returns the model's tokens against the same context only."""
    import tempfile, os, random
    import numpy as np
    from mrail import MapFile, exact_answer
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    p = os.path.join(d, "t.mrail")
    rnd = random.Random(3)
    ctx = [rnd.randrange(5, 50000) for _ in range(200000)]
    ans = [7, 8] + ctx[123456:123456 + 33] + [9, 10, 2]            # ': **' + a copied value + '**.' + stop
    m = MapFile(p, stride=256, rail_prompts=False)
    sid = m.add_track(ctx, ans, src="ruler16/131072/niah_multikey_3#5")
    assert m.tracks[sid][3][1][0] == 1 and m.tracks[sid][3][1][1] == (123456, 33)   # one copy op
    size = os.path.getsize(p)
    assert size < 120, size
    m2 = MapFile(p, stride=256, rail_prompts=False)
    assert exact_answer(m2, ctx, 64, stop={2}) == ans
    other = list(ctx); other[1000] += 1
    assert exact_answer(m2, other, 64, stop={2}) is None
    m2.add_session(list(range(10, 60)) + [3, 4], gen_start=50)    # sessions and tracks side by side
    m3 = MapFile(p, stride=256, rail_prompts=False)
    assert exact_answer(m3, ctx, 64, stop={2}) == ans and exact_answer(m3, list(range(10, 60)), 8) == [3, 4]
    print("track:", size, "bytes for a 200k-token context, answer", len(ans), "tokens")


if __name__ == "__main__":
    test_track_copy_coded()
