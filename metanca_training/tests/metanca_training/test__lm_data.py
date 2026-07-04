import numpy as np
from metanca_training.lm_data import collate_chunks


class FakeSP:
    """chars -> byte values; deterministic, no external model."""
    def encode(self, s): return [min(b, 250) for b in s.encode("utf-8")]


def test_collate_shapes_and_alignment():
    chunks = ["hello world", "hi"]
    out = collate_chunks(chunks, {300: FakeSP()}, context=8, pad_ids={300: 251})
    X, Y, M = out[300]
    assert X.shape == (2, 8) and Y.shape == (2, 8) and M.shape == (2, 8, 1)
    ids = FakeSP().encode("hello world")
    assert list(X[0]) == ids[:-1][:8]           # inputs = ids[:-1], truncated
    assert list(Y[0]) == ids[1:][:8]            # targets = ids[1:]
    n_valid = len(FakeSP().encode("hi")) - 1    # "hi" -> 2 ids -> 1 scored position
    assert M[1, :, 0].sum() == n_valid
    assert (X[1, n_valid:] == 251).all()        # padded with pad_id


def test_collate_same_chunks_all_vocabs():
    out = collate_chunks(["abc", "de"], {1: FakeSP(), 2: FakeSP()}, context=4,
                         pad_ids={1: 251, 2: 251})
    assert set(out) == {1, 2}
    assert out[1][0].shape == out[2][0].shape   # same n_chunks x context
