import functools
import operator
from typing import Sequence


def numel(shape: Sequence[int]) -> int:
    return functools.reduce(operator.mul, shape)
