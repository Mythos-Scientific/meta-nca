"""LM baseline data utilities, ported from the monorepo's data_utils (commit dc3e154).

Covers the two baseline corpora:
- text-file corpora (Shakespeare): byte-level or SentencePiece tokenization into
  fixed-context causal-LM batches (get_lm_text_datasets)
- FineWeb pre-tokenized shards (streaming batch source; get_fineweb_datasets)
"""

import logging
import math
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Iterator, Sequence

import jax
import numpy as np

logger = logging.getLogger(__name__)

BYTE_VOCAB_SIZE = 256
BYTE_EOS_TOKEN_ID = 256
BYTE_PAD_TOKEN_ID = 257
FINEWEB_VOCAB_SIZE = 1024
FINEWEB_SHARD_MAGIC = 20240520
FINEWEB_SHARD_VERSION = 1
FINEWEB_HEADER_SIZE = 256
FINEWEB_HEADER_DTYPE = np.dtype("<i4")
FINEWEB_TOKEN_DTYPE = np.dtype("<u2")
FINEWEB_HEADER_BYTES = FINEWEB_HEADER_SIZE * FINEWEB_HEADER_DTYPE.itemsize

def byte_tokenize_text(text: str, add_eos: bool = True) -> np.ndarray:
    token_ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    if add_eos:
        token_ids = np.concatenate([token_ids, np.array([BYTE_EOS_TOKEN_ID], dtype=np.int32)])
    return token_ids


def prepare_causal_lm_batches(
    token_ids: np.ndarray,
    batch_size: int,
    context_length: int,
    pad_token_id: int = BYTE_PAD_TOKEN_ID,
    eos_token_id: int = BYTE_EOS_TOKEN_ID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if context_length <= 0:
        raise ValueError("context_length must be > 0")

    token_ids = np.asarray(token_ids, dtype=np.int32)
    if token_ids.size == 0 or token_ids[-1] != eos_token_id:
        token_ids = np.concatenate([token_ids, np.array([eos_token_id], dtype=np.int32)])

    inputs = token_ids[:-1]
    targets = token_ids[1:]

    n_windows = max(1, math.ceil(inputs.shape[0] / context_length))
    total_sequences = int(math.ceil(n_windows / batch_size) * batch_size)
    n_batches = total_sequences // batch_size

    batch_inputs = np.full((n_batches, batch_size, context_length), pad_token_id, dtype=np.int32)
    batch_targets = np.full((n_batches, batch_size, context_length), pad_token_id, dtype=np.int32)
    batch_mask = np.zeros((n_batches, batch_size, context_length, 1), dtype=bool)

    for seq_idx in range(n_windows):
        start = seq_idx * context_length
        end = min(start + context_length, inputs.shape[0])
        batch_idx = seq_idx // batch_size
        batch_pos = seq_idx % batch_size
        length = end - start

        batch_inputs[batch_idx, batch_pos, :length] = inputs[start:end]
        batch_targets[batch_idx, batch_pos, :length] = targets[start:end]
        batch_mask[batch_idx, batch_pos, :length, 0] = True

    return batch_inputs, batch_targets, batch_mask


def sp_tokenize_text(
    text: str, tokenizer_model: str, add_eos: bool = True
) -> tuple[np.ndarray, int, int]:
    """Tokenize ``text`` with a trained SentencePiece model.

    Returns ``(token_ids, eos_token_id, pad_token_id)`` so callers can pad and
    terminate sequences with the tokenizer's own control ids (rather than the
    byte-level EOS=256 / PAD=257).
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model)
    token_ids = np.asarray(sp.EncodeAsIds(text), dtype=np.int32)
    eos_id, pad_id = sp.eos_id(), sp.pad_id()
    if pad_id < 0:
        raise ValueError(
            f"Tokenizer {tokenizer_model} has no pad id; train it with pad_id set so "
            "padded positions have an in-vocabulary label."
        )
    if add_eos:
        token_ids = np.concatenate([token_ids, np.array([eos_id], dtype=np.int32)])
    return token_ids, eos_id, pad_id


def get_lm_text_datasets(
    rand_key: jax.random.PRNGKey,
    text_path: str,
    context_length: int,
    batch_size: int,
    val_split: float = 0.1,
    tokenizer_model: str | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    del rand_key
    text = Path(text_path).read_text(encoding="utf-8")
    if tokenizer_model:
        token_ids, eos_token_id, pad_token_id = sp_tokenize_text(
            text, tokenizer_model, add_eos=True
        )
    else:
        token_ids = byte_tokenize_text(text, add_eos=True)
        eos_token_id, pad_token_id = BYTE_EOS_TOKEN_ID, BYTE_PAD_TOKEN_ID

    split_idx = int((1.0 - val_split) * token_ids.shape[0])
    split_idx = min(max(split_idx, 1), token_ids.shape[0] - 1)

    train_token_ids = token_ids[:split_idx]
    val_token_ids = token_ids[split_idx:]

    train_batches = prepare_causal_lm_batches(
        token_ids=train_token_ids,
        batch_size=batch_size,
        context_length=context_length,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    val_batches = prepare_causal_lm_batches(
        token_ids=val_token_ids,
        batch_size=batch_size,
        context_length=context_length,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    return train_batches, val_batches


def _read_fineweb_header(shard_path: str | Path) -> np.ndarray:
    shard_path = Path(shard_path)
    header = np.fromfile(shard_path, dtype=FINEWEB_HEADER_DTYPE, count=FINEWEB_HEADER_SIZE)
    if header.size != FINEWEB_HEADER_SIZE:
        raise ValueError(f"Short FineWeb header for {shard_path}")
    if int(header[0]) != FINEWEB_SHARD_MAGIC or int(header[1]) != FINEWEB_SHARD_VERSION:
        raise ValueError(f"Unexpected FineWeb shard header for {shard_path}")
    return header


def load_fineweb_shard_tokens(shard_path: str | Path) -> np.memmap:
    shard_path = Path(shard_path)
    header = _read_fineweb_header(shard_path)

    num_tokens = int(header[2])
    expected_size = FINEWEB_HEADER_BYTES + num_tokens * FINEWEB_TOKEN_DTYPE.itemsize
    actual_size = shard_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"FineWeb shard size mismatch for {shard_path}: "
            f"expected {expected_size} bytes, got {actual_size}"
        )

    return np.memmap(
        shard_path,
        dtype=FINEWEB_TOKEN_DTYPE,
        mode="r",
        offset=FINEWEB_HEADER_BYTES,
        shape=(num_tokens,),
    )


def _resolve_fineweb_shard_files(shard_glob: str) -> tuple[Path, ...]:
    files = tuple(Path(path) for path in sorted(glob(shard_glob)))
    if not files:
        raise FileNotFoundError(f"No FineWeb shards found for glob: {shard_glob}")
    return files


def _count_fineweb_split_tokens(files: Sequence[Path]) -> int:
    return sum(int(_read_fineweb_header(path)[2]) for path in files)


def _iter_fineweb_token_chunks(
    files: Sequence[Path],
    *,
    tokens_per_chunk: int,
    n_chunks: int,
) -> Iterator[np.ndarray]:
    if tokens_per_chunk <= 0:
        raise ValueError("tokens_per_chunk must be > 0")
    if n_chunks <= 0:
        return

    file_iter = iter(files)
    current_tokens = load_fineweb_shard_tokens(next(file_iter))
    current_pos = 0

    for _ in range(n_chunks):
        token_chunk = np.empty((tokens_per_chunk,), dtype=np.int32)
        filled = 0
        while filled < tokens_per_chunk:
            available = current_tokens.shape[0] - current_pos
            if available <= 0:
                try:
                    current_tokens = load_fineweb_shard_tokens(next(file_iter))
                except StopIteration as exc:
                    raise RuntimeError(
                        "FineWeb split ended before expected full-pass batch count"
                    ) from exc
                current_pos = 0
                continue

            take_now = min(tokens_per_chunk - filled, available)
            token_chunk[filled : filled + take_now] = current_tokens[
                current_pos : current_pos + take_now
            ]
            current_pos += take_now
            filled += take_now

        yield token_chunk


@dataclass(frozen=True)
class FineWebBatchSplit:
    """Reusable finite causal-LM batch iterator over one FineWeb split."""

    shard_glob: str
    batch_size: int
    context_length: int
    files: tuple[Path, ...] = field(init=False, repr=False)
    total_tokens: int = field(init=False)
    tokens_per_batch: int = field(init=False)
    num_batches: int = field(init=False)
    sample_batch: tuple[np.ndarray, np.ndarray, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0")

        files = _resolve_fineweb_shard_files(self.shard_glob)
        total_tokens = _count_fineweb_split_tokens(files)
        tokens_per_batch = self.batch_size * self.context_length + 1
        num_batches = total_tokens // tokens_per_batch
        if num_batches <= 0:
            raise ValueError(
                f"FineWeb split {self.shard_glob!r} does not contain enough tokens for one batch "
                f"(need at least {tokens_per_batch}, found {total_tokens})"
            )

        object.__setattr__(self, "files", files)
        object.__setattr__(self, "total_tokens", total_tokens)
        object.__setattr__(self, "tokens_per_batch", tokens_per_batch)
        object.__setattr__(self, "num_batches", num_batches)
        object.__setattr__(self, "sample_batch", next(iter(self)))

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        batch_mask = np.ones((self.batch_size, self.context_length, 1), dtype=bool)
        for token_chunk in _iter_fineweb_token_chunks(
            self.files,
            tokens_per_chunk=self.tokens_per_batch,
            n_chunks=self.num_batches,
        ):
            batch_inputs = token_chunk[:-1].reshape(self.batch_size, self.context_length)
            batch_targets = token_chunk[1:].reshape(self.batch_size, self.context_length)
            yield batch_inputs, batch_targets, batch_mask.copy()


def batch_source_num_batches(
    batches: tuple[np.ndarray, np.ndarray, np.ndarray] | FineWebBatchSplit,
) -> int:
    if isinstance(batches, FineWebBatchSplit):
        return batches.num_batches
    return int(batches[0].shape[0])


def is_streaming_batch_source(
    batches: tuple[np.ndarray, np.ndarray, np.ndarray] | FineWebBatchSplit,
) -> bool:
    return isinstance(batches, FineWebBatchSplit)


def batch_source_sample_batch(
    batches: tuple[np.ndarray, np.ndarray, np.ndarray] | FineWebBatchSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(batches, FineWebBatchSplit):
        return batches.sample_batch
    return batches[0][0], batches[1][0], batches[2][0]


def iterate_batches(
    batches: tuple[np.ndarray, np.ndarray, np.ndarray] | FineWebBatchSplit,
    batch_indices: Sequence[int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if isinstance(batches, FineWebBatchSplit):
        if batch_indices is not None:
            raise ValueError("FineWebBatchSplit does not support indexed iteration")
        yield from batches
        return

    if batch_indices is None:
        batch_indices = range(int(batches[0].shape[0]))

    for batch_idx in batch_indices:
        batch_idx = int(batch_idx)
        yield batches[0][batch_idx], batches[1][batch_idx], batches[2][batch_idx]


def get_fineweb_datasets(
    train_glob: str,
    val_glob: str,
    *,
    context_length: int,
    batch_size: int,
) -> tuple[FineWebBatchSplit, FineWebBatchSplit]:
    return (
        FineWebBatchSplit(
            shard_glob=train_glob,
            batch_size=batch_size,
            context_length=context_length,
        ),
        FineWebBatchSplit(
            shard_glob=val_glob,
            batch_size=batch_size,
            context_length=context_length,
        ),
    )


