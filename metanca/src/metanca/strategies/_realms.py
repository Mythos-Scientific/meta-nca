from functools import partial

from ._apply_strategy import apply_strategy
from ._register import register_strategy

ARGUMENTS_REALM = "arguments"
FORWARD_SLICE_REALM = "forward_slice_info"
BACKWARD_SLICE_REALM = "backward_slice_info"

register_argument_strategy = partial(register_strategy, strategy_realm=ARGUMENTS_REALM)
register_forward_slice_strategy = partial(register_strategy, strategy_realm=FORWARD_SLICE_REALM)
register_backward_slice_strategy = partial(register_strategy, strategy_realm=BACKWARD_SLICE_REALM)


def apply_argument_strategy(strategy_name: str, *args, **kwargs):
    return apply_strategy(ARGUMENTS_REALM, strategy_name, *args, **kwargs)


def apply_forward_slice_strategy(strategy_name: str, *args, **kwargs):
    return apply_strategy(FORWARD_SLICE_REALM, strategy_name, *args, **kwargs)


def apply_backward_slice_strategy(strategy_name: str, *args, **kwargs):
    return apply_strategy(BACKWARD_SLICE_REALM, strategy_name, *args, **kwargs)
