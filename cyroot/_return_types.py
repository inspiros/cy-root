from .utils.namedtuple import namedtuple

__all__ = [
    'BracketingMethodsReturnType',
    'NewtonMethodReturnType',
    'QuasiNewtonMethodReturnType',
]


def _root_results_type(typename, field_names, *, defaults=None):
    if defaults is None:
        defaults = [None] * len(field_names)
    root_results_type = namedtuple(typename, field_names, defaults=defaults)

    if 'f_calls' in field_names:
        # assuming that f_calls is tracked outside the cython kernel
        f_calls_idx = field_names.index('f_calls')

        @classmethod
        def from_results(cls, results, f_calls):
            fields = list(results)
            fields.insert(f_calls_idx, f_calls)
            return cls(*fields)

        setattr(root_results_type, 'from_results', from_results)
    return root_results_type


BracketingMethodsReturnType = _root_results_type(
    'RootResults',
    ['root', 'f_root', 'iters', 'f_calls', 'a', 'b', 'f_a', 'f_b', 'precision', 'error', 'converged', 'optimal'],
    defaults=[None, None, 0, 0, None, None, None, None, None, None, False, False],
)

NewtonMethodReturnType = _root_results_type(
    'RootResults',
    ['root', 'f_root', 'df_root', 'iters', 'f_calls', 'precision', 'error', 'converged', 'optimal'],
    defaults=[None, None, None, 0, 0, None, None, False, False]
)

QuasiNewtonMethodReturnType = _root_results_type(
    'RootResults',
    ['root', 'f_root', 'iters', 'f_calls', 'precision', 'error', 'converged', 'optimal'],
    defaults=[None, None, 0, 0, None, None, False, False]
)
