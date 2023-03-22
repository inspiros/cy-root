import inspect
import string
from functools import wraps
from types import FunctionType

import numpy as np

from .event import Event
from .format_dict import format_dict
from .setter_property import SetterProperty

__all__ = [
    'default',
    'named_default',
    'dynamic_default_args',
]

_empty = __import__('inspect')._empty


class default(Event):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f'_default({self.value})'

    @SetterProperty
    def value(self, value):
        self.__dict__['value'] = value
        self.emit(value)


class _NamedDefaultMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        from_args = len(args) in [1, 2]
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


class named_default(default, metaclass=_NamedDefaultMeta):
    def __init__(self, name=None, value=None, **kwargs):
        super().__init__(value)
        self.name = name

    def __repr__(self):
        return f'_default({self.name}={self.value})'


class _DynamicDefaultWrapperFactory:
    # container for compiled wrappers
    _COMPILED_WRAPPERS = {}

    @classmethod
    def get_compiled_wrapper(cls,
                             has_pso_dd=False,
                             has_psokw_dd=False,
                             has_kwo_dd=False):
        key = (has_pso_dd, has_psokw_dd, has_kwo_dd)
        if key not in cls._COMPILED_WRAPPERS:
            expr = """\
def wrapper(*args, **kwargs):
    {}
    {}""".format('n_args = len(args)' if has_pso_dd or has_psokw_dd else '',
                 'psokw_dd_keys = []' if has_psokw_dd else '')
            if has_pso_dd:
                expr += """
    extra_args = [None] * (pso_dd_inds[-1] + 1 - n_args)
    if len(extra_args):
        for i in range(n_args, pso_dd_inds[-1] + 1):
            if default_mask[i]:
                extra_args[i - n_args] = (defaults[i].value if dynamic_default_mask[i]
                                          else defaults[i])
            else: break"""
            if has_psokw_dd:
                expr += """
    for i in psokw_dd_inds:
        if i >= n_args:
            psokw_dd_keys.append(i)
    for i in psokw_dd_keys:
        param_name = names[i]
        if param_name not in kwargs:
            kwargs[param_name] = defaults[i].value"""
            if has_kwo_dd:
                expr += """
    for i in kwo_dd_inds:
        param_name = names[i]
        if param_name not in kwargs:
            kwargs[param_name] = defaults[i].value"""
            expr += """
    return func(*args, {}**kwargs)""".format('*extra_args, ' if has_pso_dd else '')

            cls._COMPILED_WRAPPERS[key] = compile(expr, f'<{"_".join(map(str, key))}_default_wrapper>', 'exec')
        return cls._COMPILED_WRAPPERS[key]


def dynamic_default_args():
    def decorator(func):
        params = inspect.signature(func).parameters
        n_params = len(params)

        names = [None] * n_params
        defaults = [None] * n_params
        kinds = np.empty(n_params, dtype=np.int64)
        default_mask = np.zeros(n_params, dtype=np.bool_)
        dynamic_default_mask = np.zeros(n_params, dtype=np.bool_)

        for i, (k, v) in enumerate(params.items()):
            names[i] = k
            kinds[i] = v.kind
            defaults[i] = v.default
            default_mask[i] = v.default is not _empty
            dynamic_default_mask[i] = isinstance(v.default, default)

        pso_mask = kinds == 0  # POSITIONAL_ONLY
        psokw_mask = kinds == 1  # POSITIONAL_OR_KEYWORD
        var_mask = kinds == 2  # VAR_POSITIONAL
        kwo_mask = kinds == 3  # KEYWORD_ONLY

        pso_start, psokw_start, var_start, kwo_start, _ = np.cumsum(np.pad(
            np.stack([pso_mask, psokw_mask, var_mask, kwo_mask]).sum(1), [(1, 0)]))

        dd_inds = np.where(dynamic_default_mask)[0]
        pso_dd_inds, psokw_dd_inds, kwo_dd_inds = \
            [np.where(type_mask * dynamic_default_mask)[0]
             for type_mask in [pso_mask, psokw_mask, kwo_mask]]

        compiled_wrapper = _DynamicDefaultWrapperFactory.get_compiled_wrapper(
            len(pso_dd_inds) > 0, len(psokw_dd_inds) > 0, len(kwo_dd_inds) > 0)
        context = dict(
            len=len,
            range=range,
            func=func,
            names=names,
            defaults=defaults,
            default_mask=default_mask,
            dynamic_default_mask=dynamic_default_mask,
            psokw_start=psokw_start,
            pso_dd_inds=pso_dd_inds,
            psokw_dd_inds=psokw_dd_inds,
            kwo_dd_inds=kwo_dd_inds,
        )
        wrapper = wraps(func)(FunctionType(compiled_wrapper.co_consts[0], context))

        if wrapper.__doc__ is not None:
            format_keys = set(_[1] for _ in string.Formatter().parse(wrapper.__doc__)
                              if _[1] is not None)
            if len(format_keys.intersection(names)):
                # format docstring
                wrapper.__default_doc__ = wrapper.__doc__
                format_keys_ids = np.array([i for i in range(n_params)
                                            if names[i] in format_keys])

                def update_docstring(*args, **kwargs):
                    wrapper.__doc__ = wrapper.__default_doc__.format_map(format_dict(
                        {names[i]: defaults[i].value for i in format_keys_ids}))

                update_docstring()
                # automatic update later
                for i in dd_inds:
                    defaults[i].connect(update_docstring)

        return wrapper

    return decorator
