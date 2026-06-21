from typing import Literal

import jax
import jax.extend.core as core
import networkx as nx
from frozendict import frozendict

from metanca.strategies import apply_argument_strategy
from metanca.typing import AdjacencyDict, Prim


def convert_parameter_graph(
    parameter_graph: nx.DiGraph,
    flattened_parameters: list[tuple[str, core.Var, jax.Array]],
) -> AdjacencyDict:
    """
    Convert parameter graph to pytrees
    """
    var2data = {var: (name, param) for name, var, param in flattened_parameters}
    primitives: dict[
        tuple[core.Var, core.Var],
        tuple[Prim, Literal["direct-direct", "direct-indirect", "indirect-direct"]],
    ] = nx.get_edge_attributes(parameter_graph, "primitive")

    adjacency: dict[str, list[tuple[str, str, frozendict]]] = {
        var2data[var][0]: [] for var in parameter_graph
    }

    for edge in parameter_graph.edges():

        neighbor_var, focus_var = edge
        neighbor_name, neighbor_param = var2data[neighbor_var]
        focus_name, focus_param = var2data[focus_var]

        primitive, context = primitives[edge]
        factory_arguments = apply_argument_strategy(
            primitive.name, focus_param, neighbor_param, primitive, context
        )

        adjacency[focus_name].append((neighbor_name, primitive.name, factory_arguments))

    return {k: tuple(v) for k, v in adjacency.items()}
