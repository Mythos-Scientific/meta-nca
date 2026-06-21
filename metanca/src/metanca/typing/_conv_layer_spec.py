from typing import Literal

NumChannels = int
KernelSize = tuple[int, ...]
PoolSize = tuple[int, ...]
PoolStrides = tuple[int, ...]

ConvLayerSpec = tuple[NumChannels, KernelSize, PoolSize, PoolStrides]
Padding = Literal["SAME", "VALID", "CIRCULAR"]
