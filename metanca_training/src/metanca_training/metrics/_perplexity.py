"""Perplexity metric."""

import chex
import jax.numpy as jnp


def compute_perplexity(*, loss: chex.Numeric | None = None, **_: object) -> dict[str, chex.Numeric]:
    """Return perplexity from an already-computed mean cross-entropy loss."""
    if loss is None:
        raise ValueError("perplexity requires the batch loss")
    return {"perplexity": jnp.exp(loss)}
