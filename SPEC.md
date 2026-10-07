# The `.mrail` format, version 3

A `.mrail` file is an append-only map of **exact runs of one language model**. It stores what the model has already
computed, so that a runtime can answer known contexts without running the model, restore exact states at shared
prefixes, and draft continuations for new contexts. It is designed for models whose state after any prefix has a
constant size (state-space models, linear attention, gated delta networks, fixed-state attention such as Ward recall),
where the state is a pure function of the token prefix.

All integers are little-endian.

## Layout

```
magic      8 bytes   "MRAIL" 0x03 0x00 0x00
record*    kind:u8  length:u32  payload[length]
```

Records are only ever appended. A reader stops at the first record whose declared length runs past the end of the file
(a torn write) and ignores it.

### kind 1: SESSION (a run)

```
sid:u32  n:u32  gen_start:u32  tokens:u32[n]
```

`tokens[:gen_start]` is the context the model was given (prompt, earlier turns). `tokens[gen_start:]` is the model's
own greedy output, including the token it would emit next. A runtime may serve `tokens[P:]` as the model's answer
to a context `tokens[:P]` only when `P >= gen_start`. `sid` is the index of the session in the file (informational).

### kind 2: ANCHOR (an exact state, full)

```
sid:u32  P:u32  hash:8  json_len:u32  json[json_len]  payload
```

The state of the model after `tokens[:P]` of a run whose prefix hash is `hash`. `json` describes the payload:
`{"arrays": [[name, dtype, shape], ...], "scalars": [{...} per layer], "chain": 0, "z": 1}`. The payload is the
concatenation, layer by layer and array by array in the listed order, of the raw arrays; if `"z": 1` it is
zlib-compressed as a whole.

### kind 4: DELTA (an exact state relative to a parent anchor)

```
sid:u32  P:u32  hash:8  parent_hash:8  json_len:u32  json  payload
```

As ANCHOR, but for every array listed in `row_fields` of the runtime (by default the frontier and root keys and
values, shape `[heads, rows, dim]`) the payload holds a packed bitmask of changed `[heads, rows]` entries followed by
the changed rows only; all other arrays are stored whole. `"chain"` counts deltas since the last full anchor; writers
store a full anchor once the chain would reach 8.

### kind 5: TRACK (a run whose context lives elsewhere)

```
ctx_len:u32  ctx_hash:8  src_len:u16  src:utf8[src_len]  n_ops:u32  op*
op = 0 LIT   count:u16  tokens:u32[count]
     1 COPY  pos:u32  len:u16
```

A run whose context is not stored: `ctx_hash` is the prefix hash `h_{ctx_len}` of the whole context, `src` names where
the context lives (for example an eval item `ruler16/1048576/niah_multikey_3#5`, a document id), and the model's own
output is coded against the context: `COPY` repeats `len` tokens of the context from `pos`, `LIT` gives tokens. A
runtime serves the decoded output as the model's answer only to a context whose own hash chain ends in `ctx_hash`;
it decodes the copies against that context. A run over 10^6 context tokens with a short answer costs some tens of
bytes instead of 4 MB; maps of benchmark runs, of documents held elsewhere, of repositories, are a few KB. Tracks
answer the whole context exactly (semantics 1); a runtime that needs prefixes of the context (restarts, drafts)
resolves `src` and checks it against `ctx_hash`. Readers that do not know kind 5 skip it.

### kind 3: USE

```
hash:8  hits:u32
```

A usage counter for a prefix (for example an answer served from the map). Readers sum the hits per hash.

## Prefix hashes

`h_0 = 0x00 * 8`, `h_P = blake2b(h_{P-1} || u32le(tokens[P-1]), digest_size=8)`.
Because the hash of a prefix is a chain, the longest common prefix of a new context with all runs of a map is found
by a binary search over the context's own hash chain against a set of all prefix hashes of all runs.

## Semantics

1. **Exact answer.** If a context `x[:P]` equals `tokens[:P]` of a run with `P >= gen_start`, the model's greedy
   continuation of `x` is `tokens[P:]` up to the end of the run (the model's state is a function of `x`).
2. **Exact restart.** The state after the longest prefix that has an ANCHOR or DELTA can be restored; the rest of the
   context is computed. Without anchors, any state is rebuilt from the tokens.
3. **Drafts.** For unseen contexts, runs are a source of draft continuations (for example through an n-gram index);
   drafts must be verified by the model. Sessions loaded from a foreign map are draft-only.
4. A map belongs to one exact model (weights, quantisation, runtime numerics up to ties). Anchors additionally assume
   the same first token for all runs when the model's state depends on it (Ward recall's gauge center).
