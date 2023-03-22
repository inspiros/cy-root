__all__ = [
    'tag',
    'has_tag',
]


def tag(*tags, wrap=False):
    """This decorator is used to add tag to functions."""

    def decorator(f):
        if wrap:
            @wrap(f)
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


def has_tag(f, **tags):
    return hasattr(f, '__tags__') and all(tag in f.__tags__ for tag in tags)
