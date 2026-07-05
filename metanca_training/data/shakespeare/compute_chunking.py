# metanca_training/data/shakespeare/compute_chunking.py
"""chunk_bytes is chosen by searching descending byte sizes for the largest chunk whose
P95 token count (over the single fixed vocab-10000 BPE tokenizer -- the only tokenizer in
the fixed-vocab grid) is <= CONTEXT - 1 = 63 (one extra token beyond the scored window
comes from the x/y shift)."""
import json
import numpy as np
import sentencepiece as spm

TEXT = open("input.txt", "rb").read().decode("utf-8")
VOCABS = [10000]
CONTEXT = 64

sps = {v: spm.SentencePieceProcessor(model_file=f"shakespeare_{v}_bpe.model") for v in VOCABS}

def p95_tokens(chunk_bytes, sp):
    chunks = [TEXT[i:i + chunk_bytes] for i in range(0, len(TEXT), chunk_bytes)]
    return float(np.percentile([len(sp.encode(c)) for c in chunks], 95))

finest = sps[VOCABS[0]]  # single fixed tokenizer: 10k BPE
chunk_bytes = None
for cb in range(400, 100 - 1, -8):
    if p95_tokens(cb, finest) <= CONTEXT - 1:
        chunk_bytes = cb
        break
if chunk_bytes is None:
    raise RuntimeError("no chunk_bytes in [100, 400] (step 8) satisfies p95 <= 63")

out = {"chunk_bytes": chunk_bytes, "context_length": CONTEXT, "vocabs": VOCABS,
       "p95_tokens": {str(v): p95_tokens(chunk_bytes, sps[v]) for v in VOCABS}}
json.dump(out, open("chunking.json", "w"), indent=2)
print(out)
