from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

def register(*tags: str, wrap: bool = ...) -> Callable[[_F], _F]: ...


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
