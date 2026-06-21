import functools
from typing import Any

from frozendict import frozendict

_primitives = (str, int, float, bool, type(None), bytes)
_mutable_containers = (dict, list, set)
_immutable_containers = (frozendict, tuple, frozenset)

_types = _primitives + _mutable_containers + _immutable_containers

_container_types = _mutable_containers + _immutable_containers
_map_types = (dict, frozendict)


@functools.cache
def _get_mapping(freeze: bool) -> dict:
    primitives = dict(zip(_primitives, _primitives))
    if freeze:
        across = dict(zip(_mutable_containers, _immutable_containers))
        reflexive = dict(zip(_immutable_containers, _immutable_containers))
    else:
        across = dict(zip(_immutable_containers, _mutable_containers))
        reflexive = dict(zip(_mutable_containers, _mutable_containers))

    return primitives | reflexive | across


def toggle_statics(obj: Any, freeze: bool = True):
    """
    Recursively convert Python primitives and containers into immutable equivalents:
    - dict -> frozendict
    - list -> tuple
    - tuple -> tuple
    - set -> frozenset
    - primitives -> unchanged
    """
    equiv = _get_mapping(freeze)

    if isinstance(obj, _types):
        if isinstance(obj, _primitives):
            return obj
        elif isinstance(obj, _map_types):
            return equiv[type(obj)](
                {
                    toggle_statics(k, freeze=freeze): toggle_statics(v, freeze=freeze)
                    for k, v in obj.items()
                }
            )
        elif isinstance(obj, _container_types):
            return equiv[type(obj)]([toggle_statics(u, freeze=freeze) for u in obj])
    else:
        prefix = "un" if not freeze else ""
        raise TypeError(f"Unsupported type for {prefix}freezing: {type(obj)}")
