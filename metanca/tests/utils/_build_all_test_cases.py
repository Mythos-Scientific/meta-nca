import importlib
import json
from typing import TypeVar

import pytest
import tests.resources

T = TypeVar("T", bound=dict)


def build_all_test_cases(filename: str) -> list:
    resource_path = importlib.resources.files(tests.resources).joinpath(filename)
    params = []
    with open(resource_path, "r") as f:
        all_test_cases = json.loads(f.read())

    for test_case in all_test_cases:
        realm = test_case.pop("strategy_realm")
        strategy = test_case.pop("strategy_name")
        id = test_case.pop("id")
        test = test_case["test"]
        test.update({"strategy": strategy, "realm": realm})
        params.append(
            pytest.param(
                test,
                id=id,
            )
        )

    return params
