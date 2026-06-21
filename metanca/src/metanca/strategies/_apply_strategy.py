from ._register import get_strategy_fn


def apply_strategy(realm: str, strategy_name: str, *args, **kwargs):
    fn = get_strategy_fn(realm, strategy_name)
    return fn(*args, **kwargs)
