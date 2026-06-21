from typing import TypedDict

import jax

from metanca.typing import MsgSliceInfo

Indices = jax.Array
Parameter = jax.Array
Neighbors = jax.Array


class Case(TypedDict):
    strategy: str
    realm: str
    focus_shape: jax.Array
    neighbor_shape: jax.Array
    factory_kwargs: dict
    indices: jax.Array
    slice_info: MsgSliceInfo

    focus: jax.Array
    neighbor: jax.Array

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def create(
        cls,
        key: jax.random.PRNGKey,
        strategy: str,
        realm: str,
        focus_shape: tuple[int, ...],
        neighbor_shape: tuple[int, ...],
        indices: jax.Array,
        slice_info: MsgSliceInfo,
        factory_kwargs: dict,
    ) -> tuple[jax.random.PRNGKey, "Case"]:
        test_case = {
            "strategy": strategy,
            "realm": realm,
            "focus_shape": focus_shape,
            "neighbor_shape": neighbor_shape,
            "indices": indices,
            "slice_info": slice_info,
            "factory_kwargs": factory_kwargs,
        }

        key, focus_key, neighbor_key = jax.random.split(key, 3)

        test_case["focus"] = jax.random.normal(focus_key, test_case["focus_shape"])
        test_case["neighbor"] = jax.random.normal(neighbor_key, test_case["neighbor_shape"])

        return key, Case(**test_case)
