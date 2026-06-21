from typing import Callable, ParamSpec, TypeVar

import flax.linen as nn

ApplyReturn = TypeVar("ApplyReturn")
ApplyParams = ParamSpec("ApplyParams")
ApplyLike = Callable[ApplyParams, ApplyReturn]


def forward_factory(model: nn.Module, params: dict) -> ApplyLike:
    def fwd(*args, **kwargs):
        return model.apply(params, *args, **kwargs)

    return fwd
