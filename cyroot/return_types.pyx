# distutils: language=c++

from libc cimport math

__all__ = [
    'RootReturnType',
    'BracketingMethodsReturnType',
    'NewtonMethodsReturnType',
    'MultiRootsReturnType',
    'SplittingBracketingMethodsReturnType',
    'SplittingNewtonMethodsReturnType',
]

# noinspection DuplicatedCode
cdef class RootReturnType:
    def __init__(self,
                 root=None,
                 f_root=None,
                 iters=0,
                 f_calls=0,
                 precision=math.NAN,
                 error=math.NAN,
                 converged=False,
                 optimal=False):
        self.root = root
        self.f_root = f_root
        self.iters = iters
        self.f_calls = f_calls
        self.precision = precision
        self.error = error
        self.converged = converged
        self.optimal = optimal

    def __getitem__(self, i: int):
        if i < 0: i += 8
        if i == 0: return self.root
        if i == 1: return self.f_root
        if i == 2: return self.iters
        if i == 3: return self.f_calls
        if i == 4: return self.precision
        if i == 5: return self.error
        if i == 6: return self.converged
        if i == 7: return self.optimal
        raise IndexError('Index out of range.')

    def __repr__(self):
        return f'RootResults(root={self.root}, f_root={self.f_root}, ' \
               f'iters={self.iters}, f_calls={self.f_calls}, ' \
               f'precision={self.precision}, error={self.error}, ' \
               f'converged={self.converged}, optimal={self.optimal})'

# noinspection DuplicatedCode
cdef class BracketingMethodsReturnType(RootReturnType):
    def __init__(self,
                 root=None,
                 f_root=None,
                 iters=0,
                 f_calls=0,
                 bracket=(),
                 f_bracket=(),
                 precision=math.NAN,
                 error=math.NAN,
                 converged=False,
                 optimal=False):
        super().__init__(root, f_root, iters, f_calls, precision, error, converged, optimal)
        self.bracket = bracket
        self.f_bracket = f_bracket

    def __getitem__(self, i: int):
        if i < 0: i += 10
        if i == 0: return self.root
        if i == 1: return self.f_root
        if i == 2: return self.iters
        if i == 3: return self.f_calls
        if i == 4: return self.bracket
        if i == 5: return self.f_bracket
        if i == 6: return self.precision
        if i == 7: return self.error
        if i == 8: return self.converged
        if i == 9: return self.optimal
        raise IndexError('Index out of range.')

    def __repr__(self):
        return f'RootResults(root={self.root}, f_root={self.f_root}, ' \
               f'iters={self.iters}, f_calls={self.f_calls}, ' \
               f'bracket={self.bracket}, f_bracket={self.f_bracket}, ' \
               f'precision={self.precision}, error={self.error}, ' \
               f'converged={self.converged}, optimal={self.optimal})'

# noinspection DuplicatedCode
cdef class NewtonMethodsReturnType(RootReturnType):
    def __init__(self,
                 root=None,
                 f_root=None,
                 df_root=(),
                 iters=0,
                 f_calls=0,
                 precision=math.NAN,
                 error=math.NAN,
                 converged=False,
                 optimal=False):
        super().__init__(root, f_root, iters, f_calls, precision, error, converged, optimal)
        self.df_root = df_root

    def __getitem__(self, i: int):
        if i < 0: i += 9
        if i == 0: return self.root
        if i == 1: return self.f_root
        if i == 2: return self.df_root
        if i == 3: return self.iters
        if i == 4: return self.f_calls
        if i == 5: return self.precision
        if i == 6: return self.error
        if i == 7: return self.converged
        if i == 8: return self.optimal
        raise IndexError('Index out of range.')

    def __repr__(self):
        return f'RootResults(root={self.root}, f_root={self.f_root}, df_root={self.df_root}, ' \
               f'iters={self.iters}, f_calls={self.f_calls}, ' \
               f'precision={self.precision}, error={self.error}, ' \
               f'converged={self.converged}, optimal={self.optimal})'

# --------------------------------
# Splitting Methods
# --------------------------------
# noinspection DuplicatedCode
cdef class MultiRootsReturnType:
    def __init__(self,
                 root=(),
                 f_root=(),
                 split_iters=0,
                 iters=(),
                 f_calls=0,
                 precision=(),
                 error=(),
                 converged=(),
                 optimal=()):
        self.root = root
        self.f_root = f_root
        self.split_iters = split_iters
        self.iters = iters
        self.f_calls = f_calls
        self.precision = precision
        self.error = error
        self.converged = converged
        self.optimal = optimal

    def __getitem__(self, i: int):
        if i < 0: i += 9
        if i == 0: return self.root
        if i == 1: return self.f_root
        if i == 2: return self.split_iters
        if i == 3: return self.iters
        if i == 4: return self.f_calls
        if i == 5: return self.precision
        if i == 6: return self.error
        if i == 7: return self.converged
        if i == 8: return self.optimal
        raise IndexError('Index out of range.')

    def __repr__(self):
        return f'RootResults(root={self.root}, f_root={self.f_root}, ' \
               f'split_iters={self.split_iters}, iters={self.iters}, f_calls={self.f_calls}, ' \
               f'precision={self.precision}, error={self.error}, ' \
               f'converged={self.converged}, optimal={self.optimal})'

# noinspection DuplicatedCode
cdef class SplittingBracketingMethodsReturnType(MultiRootsReturnType):
    def __init__(self,
                 root=(),
                 f_root=(),
                 split_iters=0,
                 iters=(),
                 f_calls=0,
                 bracket=(),
                 f_bracket=(),
                 precision=(),
                 error=(),
                 converged=(),
                 optimal=()):
        super().__init__(root, f_root, split_iters, iters, f_calls, precision, error, converged, optimal)
        self.bracket = bracket
        self.f_bracket = f_bracket

    def __getitem__(self, i: int):
        if i < 0: i += 11
        if i == 0: return self.root
        if i == 1: return self.f_root
        if i == 2: return self.split_iters
        if i == 3: return self.iters
        if i == 4: return self.f_calls
        if i == 5: return self.bracket
        if i == 6: return self.f_bracket
        if i == 7: return self.precision
        if i == 8: return self.error
        if i == 9: return self.converged
        if i == 10: return self.optimal
        raise IndexError('Index out of range.')

    def __repr__(self):
        return f'RootResults(root={self.root}, f_root={self.f_root}, ' \
               f'split_iters={self.split_iters}, iters={self.iters}, f_calls={self.f_calls}, ' \
               f'bracket={self.bracket}, f_bracket={self.f_bracket}, ' \
               f'precision={self.precision}, error={self.error}, ' \
               f'converged={self.converged}, optimal={self.optimal})'

# noinspection DuplicatedCode
cdef class SplittingNewtonMethodsReturnType(MultiRootsReturnType):
    def __init__(self,
                 root=(),
                 f_root=(),
                 df_root=(),
                 split_iters=0,
                 iters=(),
                 f_calls=0,
                 precision=(),
                 error=(),
                 converged=(),
                 optimal=()):
        super().__init__(root, f_root, iters, f_calls, precision, error, converged, optimal)
        self.df_root = df_root

    def __getitem__(self, i: int):
        if i < 0: i += 10
        if i == 0: return self.root
        if i == 1: return self.f_root
        if i == 2: return self.df_root
        if i == 3: return self.split_iters
        if i == 4: return self.iters
        if i == 5: return self.f_calls
        if i == 6: return self.precision
        if i == 7: return self.error
        if i == 8: return self.converged
        if i == 9: return self.optimal
        raise IndexError('Index out of range.')

    def __repr__(self):
        return f'RootResults(root={self.root}, f_root={self.f_root}, df_root={self.df_root}, ' \
               f'iters={self.split_iters}, iters={self.iters}, f_calls={self.f_calls}, ' \
               f'precision={self.precision}, error={self.error}, ' \
               f'converged={self.converged}, optimal={self.optimal})'
