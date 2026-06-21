import itertools

import jax.extend.core as core
import networkx as nx

from metanca.typing import Prim

from ._op_names import INPUT_OP_NAME, OUTPUT_OP_NAME, PARAMETER_OP_NAME
from ._opnode import OpNode


def build_compute_graph(
    jaxpr: core.Jaxpr,
) -> nx.DiGraph:

    outvar2opnode: dict[core.Var, OpNode] = {}

    parameter_ops = zip(jaxpr.constvars, itertools.repeat(PARAMETER_OP_NAME))
    invars = zip(jaxpr.invars, itertools.repeat(INPUT_OP_NAME))
    outvars = zip(jaxpr.outvars, itertools.repeat(OUTPUT_OP_NAME))

    for var, var_type in itertools.chain(parameter_ops, invars, outvars):
        outvar2opnode[var] = OpNode(
            invars=(), outvars=(var,), primitive=Prim.build(name=var_type, params={})
        )

    compute_graph = nx.DiGraph()
    for i, eqn in enumerate(jaxpr.eqns):
        opnode = OpNode(
            invars=eqn.invars,
            outvars=eqn.outvars,
            primitive=Prim.build(name=eqn.primitive.name, params=eqn.params),
            eqn_id=i,
        )
        for outvar in opnode.outvars:
            outvar2opnode[outvar] = opnode

        for invar in opnode.invars:
            # JAX equations can carry literals (e.g. reduce axes) that are not graph nodes.
            if isinstance(invar, core.Literal):
                continue
            compute_graph.add_edge(outvar2opnode[invar], opnode)

    return compute_graph
