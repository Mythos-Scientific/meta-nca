"""Multi-tokenizer collate pipeline for the LLM scaling study.

The same text chunks are tokenized by every vocab's tokenizer (a collate over raw text),
so all architectures see identical underlying text per step. Shakespeare is small, so the
collate runs eagerly at load time into static-shape arrays. (A streaming corpus would use
grain with this collate as a per-batch transform.)"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


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
            x, y = ids[:-1][:context], ids[1:][:context + 1][:context]
            n = min(len(x), len(y))
            X[i, :n] = x[:n]; Y[i, :n] = y[:n]; M[i, :n, 0] = True
        out[v] = (X, Y, M)
    return out


def _batchify(arr: np.ndarray, batch_size: int) -> np.ndarray:
    n = (len(arr) // batch_size) * batch_size          # drop ragged tail (shared across vocabs)
    return arr[:n].reshape(-1, batch_size, *arr.shape[1:])


def load_multi_vocab_shakespeare(data_dir: str, batch_size: int = 8, val_split: float = 0.1):
    import sentencepiece as spm
    d = Path(data_dir)
    cfg = json.loads((d / "chunking.json").read_text())
    context, vocabs = int(cfg["context_length"]), tuple(int(v) for v in cfg["vocabs"])
    text = (d / "input.txt").read_bytes().decode("utf-8")
    cb = int(cfg["chunk_bytes"])
    chunks = [text[i:i + cb] for i in range(0, len(text) - cb + 1, cb)]
    n_train = int(len(chunks) * (1.0 - val_split))
    sps = {v: spm.SentencePieceProcessor(model_file=str(d / f"shakespeare_{v}_bpe.model"))
           for v in vocabs}
    pad_ids = {v: sps[v].pad_id() for v in vocabs}

    train, val, factors = {}, {}, {}
    for split_name, split_chunks, store in (("train", chunks[:n_train], train),
                                            ("val", chunks[n_train:], val)):
        col = collate_chunks(split_chunks, sps, context, pad_ids)
        n_bytes = sum(len(c.encode("utf-8")) for c in split_chunks)
        for v, (X, Y, M) in col.items():
            store[v] = (_batchify(X, batch_size), _batchify(Y, batch_size),
                        _batchify(M, batch_size))
            factors.setdefault(v, {})[split_name] = float(M.sum()) / n_bytes
    return MultiVocabLM(train=train, val=val, factors=factors,
                        context_length=context, vocabs=vocabs)
