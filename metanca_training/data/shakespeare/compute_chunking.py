# metanca_training/data/shakespeare/compute_chunking.py
"""Pick chunk_bytes: largest byte span whose vocab-512 tokenization fits 63 tokens
for >=95% of chunks; record per-vocab P95 token counts."""
import json
import numpy as np
import sentencepiece as spm

TEXT = open("input.txt", "rb").read().decode("utf-8")
VOCABS = [512, 1024, 2048, 4096]
CONTEXT = 64

sps = {v: spm.SentencePieceProcessor(model_file=f"shakespeare_{v}_bpe.model") for v in VOCABS}

def p95_tokens(chunk_bytes, sp):
    chunks = [TEXT[i:i + chunk_bytes] for i in range(0, len(TEXT), chunk_bytes)]
    return float(np.percentile([len(sp.encode(c)) for c in chunks], 95))

chunk_bytes = next(b for b in range(200, 40, -8) if p95_tokens(b, sps[512]) <= CONTEXT - 1)
out = {"chunk_bytes": chunk_bytes, "context_length": CONTEXT, "vocabs": VOCABS,
       "p95_tokens": {str(v): p95_tokens(chunk_bytes, sps[v]) for v in VOCABS}}
json.dump(out, open("chunking.json", "w"), indent=2)
print(out)
