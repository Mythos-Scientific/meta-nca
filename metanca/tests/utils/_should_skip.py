def should_skip(case_info: dict) -> bool:
    return any(v is NotImplemented for v in case_info.values())
