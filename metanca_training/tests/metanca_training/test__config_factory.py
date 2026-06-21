"""Tests for derived config helper utilities."""

from metanca_training._config_factory import compute_hidden_state_dim, compute_local_rule_arch


def test_compute_hidden_state_dim_and_local_rule_arch():
    hidden_state_dim = compute_hidden_state_dim(
        input_shape=(28, 28, 1),
        d_layer=4,
        d_neuron=3,
        d_spatial=2,
    )

    local_rule_arch = compute_local_rule_arch(hidden_state_dim, [5, 6])

    assert hidden_state_dim == 14
    assert local_rule_arch == [45, 5, 6, 15]
