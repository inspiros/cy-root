import inspect
from collections import OrderedDict
from functools import wraps
from types import FunctionType

import numpy as np

__all__ = [
    'CYROOT_APIS',
    '_default',
    '_named_default',
    'cyroot_api',
    'is_cyroot_api',
]

CYROOT_APIS = OrderedDict()


class _default:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'_default({self.value})'


class _NamedDefaultMeta(type):
    _instances = {}

    def __call__(cls, name, value=None):
        # print(cls, name, value)
        if name not in cls._instances:
            cls._instances[name] = super(_NamedDefaultMeta, cls).__call__(name, value)
        return cls._instances[name]


class _named_default(_default, metaclass=_NamedDefaultMeta):
    def __init__(self, name, value=None):
        super().__init__(value)
        self.name = name

    @classmethod
    def get(cls, name):
        return cls._instances[name]

    @classmethod
    def set(cls, name, value):
        cls._instances[name].value = value


def cyroot_api(docstring_args=(), docstring_kwargs=None):
    if docstring_kwargs is None:
        docstring_kwargs = {}

    def decorator(func):
        if not inspect.isbuiltin(func):
            _empty = __import__('inspect')._empty
            params = inspect.signature(func).parameters
            n_params = len(params)

            names = np.empty(n_params, dtype=np.str_)
            defaults = [None] * n_params
            kinds = np.empty(n_params, dtype=np.int64)
            default_mask = np.zeros(n_params, dtype=np.bool_)
            dynamic_default_mask = np.zeros(n_params, dtype=np.bool_)

            for i, (k, v) in enumerate(params.items()):
                names[i] = k
                kinds[i] = v.kind
                defaults[i] = v.default
                default_mask[i] = v.default is not _empty
                dynamic_default_mask[i] = isinstance(v.default, _default)

            pso_mask = kinds == 0  # POSITIONAL_ONLY
            psokw_mask = kinds == 1  # POSITIONAL_OR_KEYWORD
            var_mask = kinds == 2  # VAR_POSITIONAL
            kwo_mask = kinds == 3  # KEYWORD_ONLY

            pso_start, psokw_start, var_start, kwo_start, _ = np.cumsum(np.pad(
                np.stack([pso_mask, psokw_mask, var_mask, kwo_mask]).sum(1), [(1, 0)]))

            # dd_inds = np.where(dynamic_default_mask)[0]
            dd_pso_inds, dd_psokw_inds, dd_kwo_inds = \
                [np.where(type_mask * dynamic_default_mask)[0]
                 for type_mask in [pso_mask, psokw_mask, kwo_mask]]

            expr = """\
def wrapper(*args, **kwargs):
    {}
    {}""".format(
                'n_args = len(args)' if len(dd_pso_inds) + len(dd_psokw_inds) else '',
                'psokw_dd_keys = []' if len(dd_psokw_inds) else '')

            if len(dd_pso_inds):
                expr += """
    extra_args = [None] * (dd_pso_inds[-1] + 1 - n_args)
    if len(extra_args):
        for i in range(n_args, dd_pso_inds[-1] + 1):
            if default_mask[i]:
                extra_args[i - n_args] = (defaults[i].value if dynamic_default_mask[i]
                                          else defaults[i])
            else: break"""
            if len(dd_psokw_inds):
                expr += """
    for i in dd_psokw_inds:
        if i >= n_args:
            psokw_dd_keys.append(i)
    for i in psokw_dd_keys:
        param_name = names[i]
        if param_name not in kwargs:
            kwargs[param_name] = defaults[i].value"""
            if len(dd_kwo_inds):
                expr += """
    for i in dd_kwo_inds:
        param_name = names[i]
        if param_name not in kwargs:
            kwargs[param_name] = defaults[i].value"""
            expr += """
    return func(*args, {}**kwargs)""".format('*extra_args, ' if len(dd_pso_inds) else '')

            compiled = compile(expr, f'<{func.__name__}_default_wrapper>', 'exec')
            context = dict(
                len=len,
                range=range,
                func=func,
                names=names,
                defaults=defaults,
                default_mask=default_mask,
                dynamic_default_mask=dynamic_default_mask,
                psokw_start=psokw_start,
                dd_pso_inds=dd_pso_inds,
                dd_psokw_inds=dd_psokw_inds,
                dd_kwo_inds=dd_kwo_inds,
            )
            wrapper = wraps(func)(FunctionType(compiled.co_consts[0], context))

        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
        # add secret identifier
        wrapper.__cyroot__ = None
        # format docstring
        wrapper.__default_doc__ = wrapper.__doc__
        wrapper.__doc__ = wrapper.__doc__.format(*docstring_args, **docstring_kwargs)
        CYROOT_APIS[wrapper.__name__] = wrapper
        return wrapper

    return decorator


def is_cyroot_api(func):
    return isinstance(func, FunctionType) and hasattr(func, '__cyroot__')
