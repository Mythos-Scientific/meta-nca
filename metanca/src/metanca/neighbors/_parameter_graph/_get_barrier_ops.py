from metanca.strategies import ARGUMENTS_REALM, filter_strategies


def get_barrier_ops() -> list[str]:
    return filter_strategies(ARGUMENTS_REALM, lambda strategy: strategy["metadata"]["barrier"])


def get_nonbarrier_ops() -> list[str]:
    return filter_strategies(ARGUMENTS_REALM, lambda strategy: not strategy["metadata"]["barrier"])
