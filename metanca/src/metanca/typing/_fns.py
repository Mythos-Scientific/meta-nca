from typing import Callable, Literal, Sequence, TypeVar

import jax
from frozendict import frozendict

from ._neuron_view import NeighborNeuronViewDict, NeuronView
from ._prim import Prim

PrimitiveContext = Literal["direct-direct", "indirect-direct", "direct-indirect"]
FactoryFn = TypeVar("FactoryFn", bound=Callable)
ArgFn = TypeVar("ArgFn", bound=Callable[[jax.Array, jax.Array, Prim, PrimitiveContext], frozendict])
NeighborGetFn = Callable[
    [str], dict[Literal["fwd", "bwd"], tuple[NeuronView, list[NeighborNeuronViewDict]]]
]
RandomIdxFn = Callable[[jax.random.PRNGKey, Sequence[int]], jax.Array]
