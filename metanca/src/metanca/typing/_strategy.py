from typing import Callable, TypedDict


class Strategy(TypedDict):
    fn: Callable
    metadata: dict
