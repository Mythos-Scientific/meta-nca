from typing import Literal, TypedDict

import jax

from ._slice_info import MsgSliceInfo


class NeuronView(TypedDict):
    """Leading axis is the neuron-based view"""

    weights: jax.Array
    hidden_states: jax.Array
    positional_encoding: jax.Array


class NeighborNeuronViewDict(TypedDict):
    direction: Literal["fwd", "bwd"]
    neighbor_name: str
    focus: NeuronView
    neighbor: NeuronView
    slice_info: MsgSliceInfo
