from typing import Any


def _empty_decorator(f):
    return f


register = _empty_decorator


def is_tagged(f: Any) -> bool:
    ...


def is_tagged_with_all(f: Any, *tags: str) -> bool:
    ...


def is_tagged_with_any(f: Any, *tags: str) -> bool:
    ...


def is_tagged_with_any_startswith(f: Any, start: str) -> bool:
    ...


def is_tagged_with_any_startswith_any(f: Any, *starts: str) -> bool:
    ...


def is_tagged_with_any_endswith(f: Any, end: str) -> bool:
    ...


def is_tagged_with_any_endswith_any(f: Any, *ends: str) -> bool:
    ...
