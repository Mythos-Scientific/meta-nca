import jax


def flatten_params(params: dict) -> list[tuple[str, jax.Array]]:
    """
    Flattens a parameter dictionary, returning the attribute path and array
    associated with each parameter.
    """
    return _flatten_params_inner(params, [])


def _flatten_params_inner(
    params: dict, param_list: list[tuple[str, jax.Array]], prefix: str = ""
) -> list[tuple[str, jax.Array]]:
    for key, value in params.items():
        if isinstance(value, jax.Array):
            param_list.append((f"{prefix}.{key}", value))
        else:
            param_list.extend(
                _flatten_params_inner(value, [], prefix=key if not prefix else f"{prefix}.{key}")
            )

    return param_list
