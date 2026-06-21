import itertools
from collections import defaultdict, deque
from typing import Optional

import jax.extend.core as core
import networkx as nx

from ._get_barrier_ops import get_barrier_ops, get_nonbarrier_ops
from ._op_names import PARAMETER_OP_NAME
from ._opnode import OpNode

ERR_MSG_AMBIG_MSG_OP = (
    "The indirect parameter's message passing op neither a barrier op nor a non-barrier op."
)
ERR_MSG_NO_REVISED_MSG_OP_FOUND = """None of the immediate predecessors of the
indirect parameter were barrier operations.
Probably this means we need to go into multi-hop preceding neighborhoods.
""".replace("\n", " ")


def _find_nearest_preceding_barrier_op(
    compute_graph: nx.DiGraph, source: OpNode, barrier_ops: set[str]
) -> Optional[OpNode]:
    visited: set[OpNode] = set()
    queue: deque[OpNode] = deque(compute_graph.predecessors(source))

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if node.primitive.name in barrier_ops:
            return node

        for predecessor in compute_graph.predecessors(node):
            if predecessor not in visited:
                queue.append(predecessor)

    return None


def _get_reachable_paths_recursive(
    compute_graph: nx.DiGraph,
    visited: set[OpNode],
    source_node: OpNode,
    current_path: list[OpNode],
    reachable_paths: dict[OpNode, list[OpNode]],
    barrier_ops: set[str],
) -> dict[OpNode, list[OpNode]]:

    for neighbor in compute_graph.successors(source_node):
        # extended_path = current_path + [neighbor]

        if neighbor not in visited:
            visited.add(neighbor)
            reachable_paths[neighbor] = current_path + [neighbor]

            if neighbor.primitive.name not in barrier_ops or len(current_path) < 1:
                _get_reachable_paths_recursive(
                    compute_graph,
                    visited,
                    neighbor,
                    current_path + [neighbor],
                    reachable_paths,
                    barrier_ops,
                )

    return reachable_paths


def _get_reachable_paths(
    compute_graph: nx.DiGraph, source_node: OpNode, barrier_ops: set[str]
) -> dict[OpNode, list[OpNode]]:
    return dict(
        _get_reachable_paths_recursive(compute_graph, set(), source_node, [], {}, barrier_ops)
    )


def build_parameter_neighbor_graphs(
    compute_graph: nx.DiGraph,
    barrier_ops: Optional[set[str]] = None,
    nonbarrier_ops: Optional[set[str]] = None,
) -> tuple[nx.DiGraph, nx.DiGraph]:

    if barrier_ops is None:
        barrier_ops = set(get_barrier_ops())

    if nonbarrier_ops is None:
        nonbarrier_ops = set(get_nonbarrier_ops())

    parameter_ops = set(
        opnode for opnode in compute_graph if opnode.primitive.name == PARAMETER_OP_NAME
    )
    parameter_vars = set(op.outvars[0] for op in parameter_ops)

    parameter_op2reachable_paths: dict[core.Var, dict[OpNode, list[OpNode]]] = {}
    intermediate_op2reachable_params = defaultdict(set)

    param_var2message_op: dict[core.Var, OpNode] = {}

    barrier_intermediates = set()
    nonbarrier_intermediates = set()

    for parameter_op in parameter_ops:
        parameter_op2reachable_paths[parameter_op.outvars[0]] = reachable_paths = (
            _get_reachable_paths(compute_graph, parameter_op, barrier_ops)
        )

        message_op = min(
            filter(lambda u: u.primitive.name in barrier_ops | nonbarrier_ops, reachable_paths),
            key=lambda u: len(reachable_paths[u]),
        )
        param_var2message_op[parameter_op.outvars[0]] = message_op

        for target_op in reachable_paths:
            if target_op.primitive.name in barrier_ops:
                barrier_intermediates.add(target_op)
                intermediate_op2reachable_params[target_op].add(parameter_op.outvars[0])

            elif target_op.primitive.name in nonbarrier_ops:
                nonbarrier_intermediates.add(target_op)
                intermediate_op2reachable_params[target_op].add(parameter_op.outvars[0])

    forward_graph = nx.DiGraph()
    backward_graph = nx.DiGraph()

    for intermediate_op in nonbarrier_intermediates:
        reachables = intermediate_op2reachable_params[intermediate_op]
        for p1, p2 in itertools.combinations(reachables, 2):
            p1_mo, p2_mo = map(param_var2message_op.get, (p1, p2))
            forward_graph.add_edge(p1, p2, primitive=(p1_mo.primitive, "direct-direct"))
            forward_graph.add_edge(p2, p1, primitive=(p2_mo.primitive, "direct-direct"))

    for intermediate_op in barrier_intermediates:
        reachables = set(intermediate_op2reachable_params[intermediate_op])
        dps = set(filter(lambda var: var in parameter_vars, intermediate_op.invars))
        for dp, ip in itertools.product(dps, reachables - dps):

            # the direct parameter is a forward neighbor of the indirect parameter.
            # the indirect parameter is a backward neighbor of direct parameter.

            # the message passing is via the intermediate operation we are considering.
            # consider: (xW1 + b1)W2
            # - the quantity xW1 + b1 is being influenced by W2.
            # - the means by which the parameters of W1 and b1 are communicating
            #   with W2 are via the multiplication "dot with W2".

            # on a layer boundary, such as flatten(conv(x, W1) + b1) dot W2 though,
            # it's important to carry forward the convolution information since this tells
            # us which dimensions convolved...
            # how else would we be able to understand (generally)
            # how to flatten the convolution filters?
            # for biases, it's not as important.
            dp_mo = param_var2message_op[dp]
            proposed_ip_mo = param_var2message_op[ip]
            forward_graph.add_edge(dp, ip, primitive=(dp_mo.primitive, "direct-indirect"))

            if proposed_ip_mo.primitive.name in nonbarrier_ops:
                # we need to walk back to the most recent barrier op
                # to understand the information transfer
                revised_ip_mo = _find_nearest_preceding_barrier_op(
                    compute_graph, proposed_ip_mo, barrier_ops
                )

                backward_graph.add_edge(
                    ip,
                    dp,
                    primitive=(
                        (revised_ip_mo or proposed_ip_mo).primitive,
                        "indirect-direct",
                    ),
                )

            elif proposed_ip_mo.primitive.name in barrier_ops:
                backward_graph.add_edge(
                    ip, dp, primitive=(proposed_ip_mo.primitive, "indirect-direct")
                )
            else:
                raise ValueError(ERR_MSG_AMBIG_MSG_OP)

    return forward_graph, backward_graph
