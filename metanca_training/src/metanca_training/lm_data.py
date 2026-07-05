"""Multi-tokenizer collate pipeline for the LLM scaling study.

The same text chunks are tokenized by every vocab's tokenizer (a collate over raw text),
so all architectures see identical underlying text per step. Shakespeare is small, so the
collate runs eagerly at load time into static-shape arrays. (A streaming corpus would use
grain with this collate as a per-batch transform.)"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


BYTE_VOCAB = 258  # byte-level "tokenizer": 0-255 raw bytes, 256 eos, 257 pad


class _ByteTokenizer:
    """Byte-level tokenizer (monorepo byte_tokenize_text convention): ids 0-255 are raw
    bytes, 256 = eos (unused by our chunk windowing), 257 = pad. Duck-types the subset of
    the SentencePiece API the collate pipeline uses."""

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def pad_id(self) -> int:
        return 257


class MultiVocabLM(NamedTuple):
    train: dict[int, tuple]
    val: dict[int, tuple]
    factors: dict[int, dict[str, float]]
    context_length: int
    vocabs: tuple[int, ...]


def collate_chunks(chunks, sps: dict[int, Any], context: int, pad_ids: dict[int, int]):
    out = {}
    for v, sp in sps.items():
        pad = pad_ids[v]
        X = np.full((len(chunks), context), pad, dtype=np.int32)
        Y = np.full((len(chunks), context), pad, dtype=np.int32)
        M = np.zeros((len(chunks), context, 1), dtype=bool)
        for i, c in enumerate(chunks):
            ids = sp.encode(c)
            x, y = ids[:-1][:context], ids[1:][:context]
            n = min(len(x), len(y))
            X[i, :n] = x[:n]; Y[i, :n] = y[:n]; M[i, :n, 0] = True
        out[v] = (X, Y, M)
    return out


def _batchify(arr: np.ndarray, batch_size: int) -> np.ndarray:
    """Reshape to [n_batches, batch_size, ...], PADDING the ragged tail with zero rows.

    No data is ever dropped (protocol decision, 2026-07-05): padded rows carry an all-False
    mask (M pads to False; X/Y pad values are never scored), so they contribute nothing to
    losses or metrics while every real chunk is consumed at any batch size.
    """
    rem = (-len(arr)) % batch_size
    if rem:
        pad_block = np.zeros((rem, *arr.shape[1:]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad_block], axis=0)
    return arr.reshape(-1, batch_size, *arr.shape[1:])


def load_multi_vocab_shakespeare(data_dir: str, batch_size: int = 8, val_split: float = 0.1):
    import sentencepiece as spm
    d = Path(data_dir)
    cfg = json.loads((d / "chunking.json").read_text())
    context, vocabs = int(cfg["context_length"]), tuple(int(v) for v in cfg["vocabs"])
    text = (d / "input.txt").read_bytes().decode("utf-8")
    # chunking slices by character offset while chunk_bytes is a byte count; only
    # equivalent for pure-ASCII text (each char == 1 byte).
    assert text.isascii(), "chunking assumes 1 byte == 1 char; recalibrate chunking.json for non-ASCII corpora"
    cb = int(cfg["chunk_bytes"])
    # include the final partial chunk — no data dropped anywhere in this pipeline
    chunks = [text[i:i + cb] for i in range(0, len(text), cb)]
    n_train = int(len(chunks) * (1.0 - val_split))
    sps = {v: (_ByteTokenizer() if v == BYTE_VOCAB else
               spm.SentencePieceProcessor(model_file=str(d / f"shakespeare_{v}_bpe.model")))
           for v in vocabs}
    pad_ids = {v: sps[v].pad_id() for v in vocabs}

    train, val, factors = {}, {}, {}
    for split_name, split_chunks, store in (("train", chunks[:n_train], train),
                                            ("val", chunks[n_train:], val)):
        # ALL split chunks are collated and consumed; _batchify pads (never drops) the
        # ragged tail with fully-masked rows, so factors computed over the collated
        # population exactly match what training/eval scores.
        col = collate_chunks(split_chunks, sps, context, pad_ids)
        n_bytes = sum(len(c.encode("utf-8")) for c in split_chunks)
        for v, (X, Y, M) in col.items():
            store[v] = (_batchify(X, batch_size), _batchify(Y, batch_size),
                        _batchify(M, batch_size))
            factors.setdefault(v, {})[split_name] = float(M.sum()) / n_bytes
    return MultiVocabLM(train=train, val=val, factors=factors,
                        context_length=context, vocabs=vocabs)
