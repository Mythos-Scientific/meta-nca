from typing import Any, Callable

from metanca.typing import Strategy

_STRATEGIES: dict[str, dict[str, Strategy]] = {}


def register_strategy(
    strategy_realm: str, strategy_name: str, overwrite: bool = False, **kwargs
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        if strategy_realm not in _STRATEGIES:
            _STRATEGIES[strategy_realm] = {strategy_name: {"fn": fn, "metadata": dict(kwargs)}}

        elif strategy_name not in _STRATEGIES[strategy_realm] or overwrite:
            strategy = _STRATEGIES[strategy_realm].get(strategy_name, {"metadata": {}})
            strategy["metadata"].update(dict(kwargs))
            strategy["fn"] = fn
            _STRATEGIES[strategy_realm][strategy_name] = strategy

        else:
            raise KeyError(f"{strategy_name} already exists in {strategy_realm}.")

        return fn

    return decorator


def list_strategy_realms() -> list[str]:
    return [k for k in _STRATEGIES]


def list_strategies(realm: str) -> list[str]:
    return [k for k in _STRATEGIES[realm]]


def get_strategy_attribute(realm: str, name: str, attribute: str) -> Any:
    return _STRATEGIES[realm][name]["metadata"][attribute]


def filter_strategies(realm: str, filter_fn: Callable[[Strategy], bool]) -> list[str]:
    return [name for name in _STRATEGIES[realm] if filter_fn(_STRATEGIES[realm][name])]


def get_strategy_fn(realm: str, strategy_name: str) -> Callable:
    return _STRATEGIES[realm][strategy_name]["fn"]
