from typing import Optional

import jax
import jax.numpy as jnp


def check_assertions(
    neighbors: jax.Array, expected_neighbors: jax.Array, prefix: str = "", debug: bool = False
) -> Optional[Exception]:
    neighbors_reshape = neighbors.reshape(-1)
    expected_neighbors_reshape = expected_neighbors.reshape(-1)

    assertions = [lambda p1, p2: len(p1) == len(p2), lambda p1, p2: jnp.allclose(p1, p2)]

    error_messages = [
        f"{prefix}# of neighbors =/= expected # of neighbors",
        f"{prefix}Incorrect neighbors",
    ]

    for assertion, err in zip(assertions, error_messages):
        result = assertion(neighbors_reshape, expected_neighbors_reshape)

        if not result:
            exc = AssertionError(err)
            if debug:
                return exc
            else:
                raise exc

    return None
