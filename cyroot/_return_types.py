from collections import namedtuple

__all__ = [
    'BracketingMethodsReturnType',
    'NewtonMethodReturnType',
    'QuasiNewtonMethodReturnType',
]

BracketingMethodsReturnType = namedtuple(
    'RootResults',
    ['root', 'f_root', 'iters', 'f_calls', 'a', 'b', 'f_a', 'f_b', 'precision', 'error', 'converged'],
    defaults=[None, None, 0, 0, None, None, None, None, None, None],
)

NewtonMethodReturnType = namedtuple(
    'RootResults',
    ['root', 'f_root', 'df_root', 'iters', 'f_calls', 'precision', 'error', 'converged'],
    defaults=[None, None, None, 0, 0, None, None, None]
)

QuasiNewtonMethodReturnType = namedtuple(
    'RootResults',
    ['root', 'f_root', 'iters', 'f_calls', 'precision', 'error', 'converged'],
    defaults=[None, None, 0, 0, None, None, None]
)
