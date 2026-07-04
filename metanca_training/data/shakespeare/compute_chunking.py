# metanca_training/data/shakespeare/compute_chunking.py
"""chunk_bytes is pinned by the FINEST tokenizer: vocab 258 is byte-level, where
tokens-per-chunk == chunk_bytes exactly, so chunk_bytes = CONTEXT - 1 = 63 (one extra
token beyond the scored window comes from the x/y shift). Record per-vocab P95 token
counts for the BPE vocabs (well under 63)."""
import json
import numpy as np
import sentencepiece as spm

TEXT = open("input.txt", "rb").read().decode("utf-8")
VOCABS = [258, 512, 1024]
BYTE_VOCAB = 258  # 0-255 raw bytes, 256 eos, 257 pad
CONTEXT = 64


class _ByteTokenizer:
    def encode(self, s):
        return list(s.encode("utf-8"))


sps = {v: (_ByteTokenizer() if v == BYTE_VOCAB else
           spm.SentencePieceProcessor(model_file=f"shakespeare_{v}_bpe.model"))
       for v in VOCABS}

def p95_tokens(chunk_bytes, sp):
    chunks = [TEXT[i:i + chunk_bytes] for i in range(0, len(TEXT), chunk_bytes)]
    return float(np.percentile([len(sp.encode(c)) for c in chunks], 95))

chunk_bytes = CONTEXT - 1  # byte-level tokens == bytes: 63 tokens fill the shifted window
out = {"chunk_bytes": chunk_bytes, "context_length": CONTEXT, "vocabs": VOCABS,
       "p95_tokens": {str(v): p95_tokens(chunk_bytes, sps[v]) for v in VOCABS}}
json.dump(out, open("chunking.json", "w"), indent=2)
print(out)
