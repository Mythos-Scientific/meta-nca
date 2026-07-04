"""Bits-per-byte / bits-per-token style metric derived from mean loss in nats."""

import chex
import jax.numpy as jnp


def compute_bpb(*, loss: chex.Numeric | None = None, **_: object) -> dict[str, chex.Numeric]:
    """Return bits per symbol from an already-computed mean cross-entropy loss."""
    if loss is None:
        raise ValueError("bpb requires the batch loss")
    return {"bpb": loss / jnp.log(2.0)}
