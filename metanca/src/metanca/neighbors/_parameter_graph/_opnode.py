from dataclasses import dataclass
from typing import Optional

from metanca.typing import Prim


@dataclass
class OpNode:
    """
    Represents an operation in a compute graph.
    """

    invars: list
    outvars: list
    primitive: Prim
    eqn_id: Optional[int] = None

    def is_unary(self) -> bool:
        return len(self.invars) == 1

    # def wrap(self, op: "OpNode") -> "OpNode":
    #    """
    #    compose an operation with another one.
    #    op1.wrap(op2) effectively should symbolize op1(op2(x))
    #    """
    #    primitive = Prim(
    #        name=f"({self.primitive} ○ {op.primitive.name})",
    #        params={"outer": self.primitive, "inner": op.primitive},
    #    )
    #    return OpNode(invars=op.invars, outvars=self.outvars, primitive=primitive)

    def __str__(self) -> str:
        return f"{self.outvars}={self.primitive}({self.invars})"

    def __hash__(self) -> int:
        return hash(str(self))
