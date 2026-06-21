from typing import TypeVar

T = TypeVar("T")


def nested_get(path: str | list[str], mapping: dict[str, T], delimiter: str = ".") -> T:
    if not isinstance(path, str):
        level = path.pop(0)

        if len(path) == 0:
            return mapping[level]
        else:
            return nested_get(path, mapping[level], delimiter=delimiter)
    else:
        return nested_get(path.split(delimiter), mapping, delimiter=delimiter)


def nested_contains(path: str | list[str], mapping: dict[str, T], delimiter: str = ".") -> bool:
    if not isinstance(path, str):
        level = path.pop(0)

        if len(path) == 0:
            return level in mapping
        else:
            return nested_contains(path, mapping[level], delimiter=delimiter)
    else:
        return nested_contains(path.split(delimiter), mapping, delimiter=delimiter)


def nested_set(path: str | list[str], value: T, mapping: dict[str, T], delimiter: str = "."):
    if not isinstance(path, str):
        level = path.pop(0)
        if level not in mapping:
            mapping[level] = {}

        if len(path) == 0:
            mapping[level] = value
            return
        else:
            return nested_set(path, value, mapping[level], delimiter=delimiter)
    else:
        return nested_set(path.split(delimiter), value, mapping, delimiter=delimiter)
