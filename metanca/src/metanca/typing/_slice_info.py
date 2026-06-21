from typing import Any, Callable, TypedDict


class SliceInfo(TypedDict):
    index_dim: int
    stride: int
    slice_size: int


class MsgSliceInfo(TypedDict):
    focus: SliceInfo
    neighbor: SliceInfo


SliceInfoFn = Callable[[Any], MsgSliceInfo]
