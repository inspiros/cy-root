from .utils.namedtuple import namedtuple

__all__ = [
    'BracketingMethodsReturnType',
    'NewtonMethodReturnType',
    'QuasiNewtonMethodReturnType',
    'SplittingBracketingMethodReturnType',
    'SplittingNewtonMethodReturnType',
    'SplittingQuasiNewtonMethodReturnType',
]


def _root_results_type(typename, field_names, *, defaults=None):
    if defaults is None:
        defaults = [None] * len(field_names)
    root_results_type = namedtuple(typename, field_names, defaults=defaults)

    if 'f_calls' in field_names:
        # assuming that f_calls is tracked outside the cython kernel
        f_calls_idx = field_names.index('f_calls')

        @classmethod
        def from_results(cls, results, f_calls, **maps):
            fields = list(results)
            fields.insert(f_calls_idx, f_calls)
            for k, map in maps.items():
                field_id = k if isinstance(k, int) else cls._fields.index(k)
                fields[field_id] = map(fields[field_id])
            res = cls(*fields)
            return res

        setattr(root_results_type, 'from_results', from_results)
    return root_results_type


BracketingMethodsReturnType = _root_results_type(
    'RootResults',
    ['root', 'f_root', 'iters', 'f_calls', 'bracket', 'f_bracket',
     'precision', 'error', 'converged', 'optimal'],
    defaults=[None, None, 0, 0, None, None, None, None, False, False],
)

QuasiNewtonMethodReturnType = _root_results_type(
    'RootResults',
    ['root', 'f_root', 'iters', 'f_calls',
     'precision', 'error', 'converged', 'optimal'],
    defaults=[None, None, 0, 0, None, None, False, False]
)

NewtonMethodReturnType = _root_results_type(
    'RootResults',
    ['root', 'f_root', 'df_root', 'iters', 'f_calls',
     'precision', 'error', 'converged', 'optimal'],
    defaults=[None, None, None, 0, 0, None, None, False, False]
)

# SplittingBracketingMethodReturnType has additional field split_iters
SplittingBracketingMethodReturnType = _root_results_type(
    'RootResults',
    ['root', 'f_root', 'split_iters', 'iters', 'f_calls', 'bracket', 'f_bracket',
     'precision', 'error', 'converged', 'optimal'],
    defaults=[(), (), 0, (), (), (), (), (), (), ()],
)

SplittingQuasiNewtonMethodReturnType = _root_results_type(
    'RootResults',
    ['root', 'f_root', 'split_iters', 'iters', 'f_calls',
     'precision', 'error', 'converged', 'optimal'],
    defaults=[(), (), 0, (), (), (), (), (), ()]
)

SplittingNewtonMethodReturnType = _root_results_type(
    'RootResults',
    ['root', 'f_root', 'df_root', 'split_iters', 'iters', 'f_calls',
     'precision', 'error', 'converged', 'optimal'],
    defaults=[(), (), 0, (), (), (), (), (), (), ()]
)
