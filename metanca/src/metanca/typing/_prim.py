from dataclasses import dataclass

from frozendict import frozendict


@dataclass(frozen=True)
class Prim:
    """Represents a primitive operation"""

    name: str
    params: frozendict

    @classmethod
    def build(cls, name: str, params: dict) -> "Prim":
        return Prim(name, params=frozendict(params))

    def __str__(self) -> str:
        return f"{self.name}[{self.params}]"
