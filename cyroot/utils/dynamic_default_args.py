import inspect
import string
from functools import wraps
from typing import Optional, Any

from .event import Event
from .format_dict import format_dict
from .setter_property import SetterProperty

__all__ = [
    'default',
    'named_default',
    'dynamic_default_args',
]

_empty = __import__('inspect')._empty


class _default(Event):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __repr__(self):
        return repr(self.value)

    @SetterProperty
    def value(self, value):
        self.__dict__['value'] = value
        self.emit(value)


class _NamedDefaultMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if len(args) == 2 and args[0] is None:
            args = ()
        from_args = len(args) in (1, 2)
        from_kwargs = len(kwargs) == 1
        if not from_args ^ from_kwargs:
            raise ValueError('Define named default with one string and one value, '
                             'either by two positional arguments or a single keyword '
                             'argument. If only name given, value will be set to None.')
        if from_args:
            name, value = args[0], args[1] if len(args) == 2 else None
        elif from_kwargs:
            name, value = next(iter(kwargs.items()))
        if not isinstance(name, str):
            raise ValueError(f'Name must be string. Got {type(name)}.')
        if name not in cls._instances:
            cls._instances[name] = super(_NamedDefaultMeta, cls).__call__(name, value)
        return cls._instances[name]


class _named_default(_default, metaclass=_NamedDefaultMeta):
    def __init__(self, name=None, value=None, **kwargs):
        super().__init__(value)
        self.name = name


def default(value: Any) -> Any:
    """
    Create a default object that holds a default value.
    """
    return _default(value)


def named_default(name: Optional[str] = None,
                  value: Optional[Any] = None,
                  **kwargs) -> Any:
    """Create a named default object that holds a default value
    for arguments, which can be dynamically changed later.

    This function accpets passing two positional arguments name
    and value. If value is not provided and the name hasn't been
    registered, value will be set to ``None``.

    >>> def foo(x=named_default('x', 0.5)):
    >>>    ...

    Ortherwise, use a single keyword argument. The keyword will
    be used as name.

    >>> def foo(x=named_default(x=1)):
    >>>    ...

    For modifying the default values everywhere, call this function
    with only the name of defined variable. Any value provided will
    have no effect.

    >>> default_x = named_default('x')
    >>> default_x.value = 2.0
    """
    return _named_default(name, value, **kwargs)


def dynamic_default_args(format_doc=True, force_wrap=False):
    """
    A decorator for substituting function with dynamic default
    arguments with minimal overhead.

    It can also will modify the function's docstring with
    format keys automatically when any of the default args changes.

    >>> @dynamic_default_args(format_doc=True)
    >>> def foo(x=named_default(x=5))
    >>>     \"\"\"An exmaple function with docstring.
    >>>
    >>>     Args:
    >>>         x: Argument dynamically defaults to {x}.
    >>>     \"\"\"
    >>>     ...

    When the default value is changed later, both the default and
    the function's docstring will be updated accordingly.

    >>> named_default('x').value = 10
    >>> foo()
    10
    >>> help(foo)

    Args:
        format_doc: Automatically format the docstring of the
         decorated function or not. Defaults to ``True``.
        force_wrap: Wrap the decorated function even if there
         is no dynamic default argument or not.
         Defaults to ``False``.
    """

    def decorator(func):
        params = inspect.signature(func).parameters
        n_params = len(params)

        names = list(params.keys())
        defaults = [v.default for v in params.values()]
        kinds = [v.kind for v in params.values()]
        default_mask = [True if v.default is not _empty else False
                        for v in params.values()]
        dynamic_default_mask = [True if isinstance(v.default, _default) else False
                                for v in params.values()]
        del params

        has_dynamic_defaults = any(dynamic_default_mask)
        if force_wrap or has_dynamic_defaults:
            func_alias = 'func'
            wrapper_alias = 'wrapper'
            default_alias = 'default'
            while func_alias in names:
                func_alias = '_' + func_alias
            while wrapper_alias in names:
                wrapper_alias = '_' + wrapper_alias
            while default_alias in names:
                default_alias = '_' + default_alias
            context = {
                default_alias: _default,
                func_alias: func
            }

            expr = f'def {wrapper_alias}('
            for i, (name, kind, default_val) in enumerate(zip(names, kinds, defaults)):
                if default_mask[i]:
                    context[f'{name}_'] = default_val
                expr += '{}{}'.format('*' if kind == 2 else '**' if kind == 4 else '', name)
                if default_val is not _empty:
                    expr += f'={name}_'
                if i < n_params - 1:
                    expr += ', '
            expr += f'):\n\treturn {func_alias}('
            for i, (name, kind) in enumerate(zip(names, kinds)):
                if kind == 2:
                    expr += '*'
                elif kind == 4:
                    expr += '**'
                expr += name
                if kind == 3:
                    expr += f'={name}'
                if dynamic_default_mask[i]:
                    expr += f'.value if isinstance({name}, {default_alias}) else {name}'
                if i < n_params - 1:
                    expr += ', '
            expr += ')\n'
            exec_locals = {}
            exec(compile(expr, f'<{func.__name__}_wrapper>', 'exec'), context, exec_locals)
            wrapper = wraps(func)(exec_locals[wrapper_alias])
        else:  # no wrapping
            wrapper = func

        if format_doc and wrapper.__doc__ is not None:
            format_keys = set(_[1] for _ in string.Formatter().parse(wrapper.__doc__)
                              if _[1] is not None)
            format_keys = format_keys.intersection(names)
            if len(format_keys):
                # format docstring
                wrapper.__default_doc__ = wrapper.__doc__
                format_keys_ids = [i for i in range(n_params) if names[i] in format_keys]

                def update_docstring(*args, **kwargs):
                    wrapper.__doc__ = wrapper.__default_doc__.format_map(format_dict({
                        names[i]: defaults[i].value if dynamic_default_mask[i]
                        else defaults[i] for i in format_keys_ids
                        if defaults[i] is not _empty}))

                update_docstring()
                # automatic update later
                for i in format_keys_ids:
                    if dynamic_default_mask[i]:
                        defaults[i].connect(update_docstring)

        return wrapper

    return decorator
