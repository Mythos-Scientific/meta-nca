from typing import Any, Callable, Type


def raise_with_err(
    msg: str, exception_type: Type[Exception] = NotImplementedError
) -> Callable[[Any], None]:
    def raise_fn(*args):
        raise exception_type(msg)

    return raise_fn
