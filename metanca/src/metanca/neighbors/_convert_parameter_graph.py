import logging
from typing import Literal

import jax
import jax.extend.core as core
import networkx as nx
from frozendict import frozendict

from metanca.strategies import (
    apply_argument_strategy,
    apply_backward_slice_strategy,
    apply_forward_slice_strategy,
)
from metanca.typing import AdjacencyDict, Prim

logger = logging.getLogger(__name__)


def convert_parameter_graph(
    parameter_graph: nx.DiGraph,
    flattened_parameters: list[tuple[str, core.Var, jax.Array]],
    slice_strategy=apply_forward_slice_strategy,
) -> AdjacencyDict:
    """
    Convert parameter graph to pytrees.

    ``slice_strategy`` (forward by default) is consulted to validate that the
    edge's primitive can actually slice this neighbor pair. Edges that fail
    validation (e.g. residual-add between two non-bias param shapes) are
    dropped — keeping them would crash later inside ``get_neighbors``.
    """
    var2data = {var: (name, param) for name, var, param in flattened_parameters}
    primitives: dict[
        tuple[core.Var, core.Var],
        tuple[Prim, Literal["direct-direct", "direct-indirect", "indirect-direct"]],
    ] = nx.get_edge_attributes(parameter_graph, "primitive")

    # Some models (e.g. token LMs that do work in ``apply``) leave extra
    # non-parameter constvars in the jaxpr. ``build_compute_graph`` adds those
    # as PARAMETER nodes and ``build_parameter_neighbor_graphs`` may reference
    # them; skip vars that don't correspond to real parameters.
    adjacency: dict[str, list[tuple[str, str, frozendict]]] = {
        var2data[var][0]: [] for var in parameter_graph if var in var2data
    }

    for edge in parameter_graph.edges():
        neighbor_var, focus_var = edge
        if neighbor_var not in var2data or focus_var not in var2data:
            continue
        neighbor_name, neighbor_param = var2data[neighbor_var]
        focus_name, focus_param = var2data[focus_var]

        primitive, context = primitives[edge]
        factory_arguments = apply_argument_strategy(
            primitive.name, focus_param, neighbor_param, primitive, context
        )

        try:
            slice_strategy(primitive.name, **factory_arguments)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            logger.debug(
                "Dropping incompatible %s edge %s -> %s: %s",
                primitive.name,
                neighbor_name,
                focus_name,
                exc,
            )
            continue

        adjacency[focus_name].append((neighbor_name, primitive.name, factory_arguments))

    return {k: tuple(v) for k, v in adjacency.items()}


def convert_parameter_graph_backward(
    parameter_graph: nx.DiGraph,
    flattened_parameters: list[tuple[str, core.Var, jax.Array]],
) -> AdjacencyDict:
    return convert_parameter_graph(
        parameter_graph,
        flattened_parameters,
        slice_strategy=apply_backward_slice_strategy,
    )
