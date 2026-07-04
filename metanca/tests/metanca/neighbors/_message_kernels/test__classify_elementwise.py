import pytest

from metanca.neighbors._message_kernels._elementwise_nonbarrier_op import (
    _classify_elementwise,
)


def test_rank1_equal_shapes_are_pointwise():
    assert _classify_elementwise((128,), (128,), "add") == "equal_shape"


def test_bias_broadcast():
    assert _classify_elementwise((128, 512), (512,), "add") == "broadcast_vector"


def test_residual_channel_pair():
    # embedding[V,D] + out_proj[D,D] both write to residual channel D
    assert _classify_elementwise((128, 128), (512, 128), "add") == "shared_trailing_channel"


def test_equal_shape_matrices_are_residual_not_pointwise():
    """mlp_dim == vocab_size makes mlp_out[M,D] shape-equal to embedding[V,D];
    the pair still shares only the trailing residual channel. Classifying it as
    a pointwise ("equal_shape") pairing sliced the neighbor along the wrong
    axis and crashed the NCA update (concat shape mismatch)."""
    assert _classify_elementwise((512, 128), (512, 128), "add") == "shared_trailing_channel"


def test_incompatible_raises():
    with pytest.raises(TypeError):
        _classify_elementwise((128, 512), (300,), "add")
