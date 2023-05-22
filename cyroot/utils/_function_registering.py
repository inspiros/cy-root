from functools import wraps
from typing import Callable, Any

__all__ = [
    'register',
    'is_tagged',
    'is_tagged_with_all',
    'is_tagged_with_any_startswith',
    'is_tagged_with_any_startswith_any',
    'is_tagged_with_any_endswith',
    'is_tagged_with_any_endswith_any',
]


def register(*tags: str, wrap: bool = False):
    """This decorator is used to add tag to functions."""

    def decorator(f: Callable):
        if wrap:
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
        else:
            wrapper = f
        if not hasattr(wrapper, '__tags__'):
            wrapper.__tags__ = set()
        for tag in tags:
            wrapper.__tags__.add(tag)

        return wrapper

    return decorator


def is_tagged(f: Any) -> bool:
    return hasattr(f, '__tags__')


def is_tagged_with_all(f: Any, *tags: str) -> bool:
    return hasattr(f, '__tags__') and all(tag in f.__tags__ for tag in tags)


def is_tagged_with_any(f: Any, *tags: str) -> bool:
    return hasattr(f, '__tags__') and any(tag in f.__tags__ for tag in tags)


def is_tagged_with_any_startswith(f: Any, start: str) -> bool:
    return hasattr(f, '__tags__') and any(tag.startswith(start) for tag in f.__tags__)


def is_tagged_with_any_startswith_any(f: Any, *starts: str) -> bool:
    return hasattr(f, '__tags__') and any(any(tag.startswith(_ for _ in starts))
                                          for tag in f.__tags__)


def is_tagged_with_any_endswith(f: Any, end: str) -> bool:
    return hasattr(f, '__tags__') and any(tag.endswith(end) for tag in f.__tags__)


def is_tagged_with_any_endswith_any(f: Any, *ends: str) -> bool:
    return hasattr(f, '__tags__') and any(any(tag.endswith(_ for _ in ends))
                                          for tag in f.__tags__)
