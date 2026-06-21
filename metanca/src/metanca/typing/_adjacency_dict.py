from frozendict import frozendict

NeighborName = str
MessageFn = str
MessageFnParams = frozendict
AdjacencyDict = frozendict[str, tuple[tuple[NeighborName, MessageFn, MessageFnParams]]]
