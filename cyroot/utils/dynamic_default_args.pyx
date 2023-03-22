# distutils: language=c++
from cpython cimport array
import array
import inspect
import string
from functools import wraps
from types import FunctionType
from typing import Optional, Any

cimport numpy as cnp
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


class _default(Event):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f'default({self.value})'

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

    def __repr__(self):
        return f'default({self.name}={self.value})'


def default(value: Any) -> Any:
    return _default(value)


def named_default(name: Optional[str] = None,
                  value: Optional[Any] = None,
                  **kwargs) -> Any:
    return _named_default(name, value, **kwargs)


# container for compiled wrappers
_COMPILED_WRAPPERS: dict = {}

cdef _get_compiled_wrapper(context: dict = None):
    if context is None:
        context = {}
    cdef tuple key = tuple([len(context['pso_dd_inds']) > 0,
                            len(context['psokw_dd_inds']) > 0,
                            len(context['kwo_dd_inds']) > 0])
    if key not in _COMPILED_WRAPPERS:
        expr = """\
def wrapper(*args, **kwargs):"""
        if key[0] or key[1]:
            expr += """
    n_args = len(args)"""
        if key[0]:
            expr += """
    extra_args = [None] * (pso_dd_inds[-1] + 1 - n_args)"""
        if key[1]:
            expr += """
    psokw_dd_keys = array('i')"""
        if key[0]:
            expr += """
    extra_args = [None] * (pso_dd_inds[-1] + 1 - n_args)
    if len(extra_args):
        for i in range(n_args, pso_dd_inds[-1] + 1):
            if default_mask[i]:
                extra_args[i - n_args] = defaults[i].value
            else: break"""
        if key[1]:
            expr += """
    for i in psokw_dd_inds:
        if i >= n_args:
            psokw_dd_keys.append(i)
    for i in psokw_dd_keys:
        param_name = names[i]
        if param_name not in kwargs:
            kwargs[param_name] = defaults[i].value"""
        if key[2]:
            expr += """
    for i in kwo_dd_inds:
        param_name = names[i]
        if param_name not in kwargs:
            kwargs[param_name] = defaults[i].value"""
        expr += """
    return func(*args, {}**kwargs)""".format('*extra_args, ' if key[0] else '')

        _COMPILED_WRAPPERS[key] = compile(expr, '<dynamic_default_function_wrapper>', 'exec')
    return FunctionType(_COMPILED_WRAPPERS[key].co_consts[0], globals=context)

cdef _get_cython_wrapper(context: dict = None):
    if context is None:
        context = {}
    cdef:
        object func = context['func']
        list names = context['names']
        list defaults = context['defaults']
        cnp.npy_bool[:] default_mask = context['default_mask']
        int psokw_start = context['psokw_start']
        cnp.int64_t[:] pso_dd_inds = context['pso_dd_inds']
        cnp.int64_t[:] psokw_dd_inds = context['psokw_dd_inds']
        cnp.int64_t[:] kwo_dd_inds = context['kwo_dd_inds']

        bint[3] key = [len(pso_dd_inds) > 0,
                       len(psokw_dd_inds) > 0,
                       len(kwo_dd_inds) > 0]

    if key == [0, 0, 0]:
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
    elif key == [0, 0, 1]:
        def wrapper(*args, **kwargs):
            cdef:
                int i
                str param_name
            for i in kwo_dd_inds:
                param_name = names[i]
                if param_name not in kwargs:
                    kwargs[param_name] = defaults[i].value
            return func(*args, **kwargs)
    elif key == [0, 1, 0]:
        def wrapper(*args, **kwargs):
            cdef:
                int i, n_args = len(args)
                str param_name
                array.array psokw_dd_keys = array.array('i')
            for i in psokw_dd_inds:
                if i >= n_args:
                    psokw_dd_keys.append(i)
            for i in psokw_dd_keys:
                param_name = names[i]
                if param_name not in kwargs:
                    kwargs[param_name] = defaults[i].value
            return func(*args, **kwargs)
    elif key == [0, 1, 1]:
        def wrapper(*args, **kwargs):
            cdef:
                int i, n_args = len(args)
                str param_name
                array.array psokw_dd_keys = array.array('i')
            for i in psokw_dd_inds:
                if i >= n_args:
                    psokw_dd_keys.append(i)
            for i in psokw_dd_keys:
                param_name = names[i]
                if param_name not in kwargs:
                    kwargs[param_name] = defaults[i].value
            for i in kwo_dd_inds:
                param_name = names[i]
                if param_name not in kwargs:
                    kwargs[param_name] = defaults[i].value
            return func(*args, **kwargs)
    elif key == [1, 0, 0]:
        def wrapper(*args, **kwargs):
            cdef:
                int i, n_args = len(args)
                list extra_args = [None] * (pso_dd_inds[-1] + 1 - n_args)
            if len(extra_args):
                for i in range(n_args, pso_dd_inds[-1] + 1):
                    if default_mask[i]:
                        extra_args[i - n_args] = defaults[i].value
                    else:
                        break
            return func(*args, *extra_args, **kwargs)
    elif key == [1, 0, 1]:
        def wrapper(*args, **kwargs):
            cdef:
                int i, n_args = len(args)
                str param_name
                list extra_args = [None] * (pso_dd_inds[-1] + 1 - n_args)
            if len(extra_args):
                for i in range(n_args, pso_dd_inds[-1] + 1):
                    if default_mask[i]:
                        extra_args[i - n_args] = defaults[i].value
                    else:
                        break
            for i in kwo_dd_inds:
                param_name = names[i]
                if param_name not in kwargs:
                    kwargs[param_name] = defaults[i].value
            return func(*args, *extra_args, **kwargs)
    elif key == [1, 1, 0]:
        def wrapper(*args, **kwargs):
            cdef:
                int i, n_args = len(args)
                str param_name
                list extra_args = [None] * (pso_dd_inds[-1] + 1 - n_args)
                array.array psokw_dd_keys = array.array('i')
            if len(extra_args):
                for i in range(n_args, pso_dd_inds[-1] + 1):
                    if default_mask[i]:
                        extra_args[i - n_args] = defaults[i].value
                    else:
                        break
            for i in psokw_dd_inds:
                if i >= n_args:
                    psokw_dd_keys.append(i)
            for i in psokw_dd_keys:
                param_name = names[i]
                if param_name not in kwargs:
                    kwargs[param_name] = defaults[i].value
            return func(*args, *extra_args, **kwargs)
    elif key == [1, 1, 1]:
        def wrapper(*args, **kwargs):
            cdef:
                int i, n_args = len(args), n_extra_args = pso_dd_inds[-1] + 1 - n_args
                str param_name
                list extra_args = [None] * n_extra_args
                array.array psokw_dd_keys = array.array('i')
            if n_extra_args:
                for i in range(n_args, pso_dd_inds[-1] + 1):
                    if default_mask[i]:
                        extra_args[i - n_args] = defaults[i].value
                    else:
                        break
            for i in psokw_dd_inds:
                if i >= n_args:
                    psokw_dd_keys.append(i)
            for i in psokw_dd_keys:
                param_name = names[i]
                if param_name not in kwargs:
                    kwargs[param_name] = defaults[i].value
            for i in kwo_dd_inds:
                param_name = names[i]
                if param_name not in kwargs:
                    kwargs[param_name] = defaults[i].value
            return func(*args, *extra_args, **kwargs)

    return wrapper

def dynamic_default_args(format_doc: bool = True,
                         force_wrap: bool = False,
                         cython_wrapper: bool = True):
    def decorator(func):
        params = inspect.signature(func).parameters
        cdef:
            unsigned int n_params = len(params)

            list names = [None] * n_params
            list defaults = [None] * n_params
            cnp.ndarray[cnp.int64_t, ndim=1] kinds = np.empty(n_params, dtype=np.int64)
            cnp.ndarray[cnp.npy_bool, ndim=1] default_mask = np.zeros(n_params, dtype=np.bool_)
            cnp.ndarray[cnp.npy_bool, ndim=1] dynamic_default_mask = np.zeros(n_params, dtype=np.bool_)

        cdef:
            int i
            str k
            object v
        for i, (k, v) in enumerate(params.items()):
            names[i] = k
            kinds[i] = v.kind
            default_mask[i] = v.default is not _empty
            dynamic_default_mask[i] = isinstance(v.default, _default)
            defaults[i] = v.default if dynamic_default_mask[i] else _default(v.default)
        cdef:
            cnp.ndarray[cnp.npy_bool, ndim=1] pso_mask = kinds == 0  # POSITIONAL_ONLY
            cnp.ndarray[cnp.npy_bool, ndim=1] psokw_mask = kinds == 1  # POSITIONAL_OR_KEYWORD
            cnp.ndarray[cnp.npy_bool, ndim=1] var_mask = kinds == 2  # VAR_POSITIONAL
            cnp.ndarray[cnp.npy_bool, ndim=1] kwo_mask = kinds == 3  # KEYWORD_ONLY
            int pso_start, psokw_start, var_start, kwo_start
        pso_start, psokw_start, var_start, kwo_start, _ = np.cumsum(np.pad(
            np.stack([pso_mask, psokw_mask, var_mask, kwo_mask]).sum(1), [(1, 0)]))

        cdef cnp.ndarray[cnp.int64_t, ndim=1] pso_dd_inds, psokw_dd_inds, kwo_dd_inds
        pso_dd_inds, psokw_dd_inds, kwo_dd_inds = \
            [np.where(type_mask * dynamic_default_mask)[0]
             for type_mask in [pso_mask, psokw_mask, kwo_mask]]

        cdef:
            bint has_pso_dd = pso_dd_inds.size > 0
            bint has_psokw_dd = psokw_dd_inds.size > 0
            bint has_kwo_dd = kwo_dd_inds.size > 0
        if force_wrap or has_pso_dd or has_psokw_dd or has_kwo_dd:
            if cython_wrapper:
                wrapper = wraps(func)(_get_cython_wrapper(context=dict(
                    func=func,
                    names=names,
                    defaults=defaults,
                    default_mask=default_mask,
                    psokw_start=psokw_start,
                    pso_dd_inds=pso_dd_inds,
                    psokw_dd_inds=psokw_dd_inds,
                    kwo_dd_inds=kwo_dd_inds,
                )))
            else:
                wrapper = wraps(func)(_get_compiled_wrapper(context=dict(
                    len=len,
                    range=range,
                    array=array.array,
                    func=func,
                    names=names,
                    defaults=defaults,
                    default_mask=default_mask,
                    psokw_start=psokw_start,
                    pso_dd_inds=pso_dd_inds,
                    psokw_dd_inds=psokw_dd_inds,
                    kwo_dd_inds=kwo_dd_inds,
                )))
        else:
            wrapper = func

        cdef:
            set format_keys
            list format_keys_ids
        if format_doc and wrapper.__doc__ is not None:
            format_keys = set(_[1] for _ in string.Formatter().parse(wrapper.__doc__)
                              if _[1] is not None)
            if len(format_keys.intersection(names)):
                # format docstring
                wrapper.__default_doc__ = wrapper.__doc__
                format_keys_ids = [i for i in range(n_params)
                                   if names[i] in format_keys]

                def update_docstring(*args, **kwargs):
                    wrapper.__doc__ = wrapper.__default_doc__.format_map(format_dict(
                        {names[i]: defaults[i].value for i in format_keys_ids}))

                update_docstring()
                # automatic update later
                for i in format_keys_ids:
                    defaults[i].connect(update_docstring)

        return wrapper

    return decorator
