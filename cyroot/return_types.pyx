# distutils: language=c++

from libc cimport math
from operator import index
from .status import TerminationReason
from ._stopping cimport (
    RESIDUAL_TOLERANCE, BRACKET_TOLERANCE, STAGNATION, MAX_ITERATIONS,
    ZERO_DERIVATIVE, BREAKDOWN, NONFINITE, MAX_SPLIT_ITERATIONS, COMPLETED,
    UNSPECIFIED, CALLBACK_FAILURE,
)

_REASON_CODES = {
    RESIDUAL_TOLERANCE: TerminationReason.RESIDUAL_TOLERANCE,
    BRACKET_TOLERANCE: TerminationReason.BRACKET_TOLERANCE,
    STAGNATION: TerminationReason.STAGNATION,
    MAX_ITERATIONS: TerminationReason.MAX_ITERATIONS,
    ZERO_DERIVATIVE: TerminationReason.ZERO_DERIVATIVE,
    BREAKDOWN: TerminationReason.BREAKDOWN,
    NONFINITE: TerminationReason.NONFINITE,
    MAX_SPLIT_ITERATIONS: TerminationReason.MAX_SPLIT_ITERATIONS,
    COMPLETED: TerminationReason.COMPLETED,
    UNSPECIFIED: TerminationReason.UNSPECIFIED,
    CALLBACK_FAILURE: TerminationReason.CALLBACK_FAILURE,
}


def _normalize_reason(reason):
    if reason is None:
        return TerminationReason.UNSPECIFIED
    if isinstance(reason, str):
        return TerminationReason(reason)
    try:
        code = index(reason)
    except TypeError:
        raise ValueError('termination reason must be a known string or native status code') from None
    if isinstance(reason, bool) or code not in _REASON_CODES:
        raise ValueError('unknown termination reason')
    return _REASON_CODES[code]


def _successful(reason):
    return reason in (TerminationReason.RESIDUAL_TOLERANCE, TerminationReason.BRACKET_TOLERANCE)


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
                 optimal=False,
                 termination_reason=None):
        self.root = root
        self.f_root = f_root
        self.iters = iters
        self.f_calls = f_calls
        self.precision = precision
        self.error = error
        self.converged = converged
        self.optimal = optimal
        self.termination_reason = _normalize_reason(termination_reason)
        if termination_reason is not None:
            self.converged = _successful(self.termination_reason)
            self.optimal = self.termination_reason == TerminationReason.RESIDUAL_TOLERANCE

    @property
    def success(self):
        return self.converged

    def __iter__(self):
        i = 0
        while True:
            try:
                value = self[i]
            except (IndexError, StopIteration):
                return
            yield value
            i += 1

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
               f'converged={self.converged}, optimal={self.optimal}, ' \
               f'termination_reason={str(self.termination_reason)!r})'

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
                 optimal=False,
                 termination_reason=None):
        super().__init__(root, f_root, iters, f_calls, precision, error, converged, optimal, termination_reason)
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
               f'converged={self.converged}, optimal={self.optimal}, ' \
               f'termination_reason={str(self.termination_reason)!r})'

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
                 optimal=False,
                 termination_reason=None):
        super().__init__(root, f_root, iters, f_calls, precision, error, converged, optimal, termination_reason)
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
               f'converged={self.converged}, optimal={self.optimal}, ' \
               f'termination_reason={str(self.termination_reason)!r})'

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
                 optimal=(),
                 termination_reasons=None,
                 search_termination_reason=None):
        self.root = list(root)
        self.f_root = list(f_root)
        self.split_iters = split_iters
        self.iters = list(iters)
        self.f_calls = f_calls
        self.precision = list(precision)
        self.error = list(error)
        self.converged = list(converged)
        self.optimal = list(optimal)
        self.search_termination_reason = _normalize_reason(search_termination_reason)
        if termination_reasons is None:
            self.termination_reasons = [TerminationReason.UNSPECIFIED] * len(self.root)
        else:
            self.termination_reasons = [_normalize_reason(reason) for reason in termination_reasons]
            if len(self.termination_reasons) != len(self.root):
                raise ValueError('one termination reason is required per returned root')
            self.converged = [_successful(reason) for reason in self.termination_reasons]
            self.optimal = [reason == TerminationReason.RESIDUAL_TOLERANCE for reason in self.termination_reasons]

    @property
    def success(self):
        return (self.search_termination_reason == TerminationReason.COMPLETED
                and len(self.converged) == len(self.root) and all(self.converged))

    def __iter__(self):
        i = 0
        while True:
            try:
                value = self[i]
            except (IndexError, StopIteration):
                return
            yield value
            i += 1

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
               f'converged={self.converged}, optimal={self.optimal}, ' \
               f'termination_reasons={self.termination_reasons}, ' \
               f'search_termination_reason={str(self.search_termination_reason)!r})'

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
                 optimal=(),
                 termination_reasons=None,
                 search_termination_reason=None):
        super().__init__(root, f_root, split_iters, iters, f_calls, precision, error, converged, optimal,
                         termination_reasons, search_termination_reason)
        self.bracket = list(bracket)
        self.f_bracket = list(f_bracket)

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
               f'converged={self.converged}, optimal={self.optimal}, ' \
               f'termination_reasons={self.termination_reasons}, ' \
               f'search_termination_reason={str(self.search_termination_reason)!r})'

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
                 optimal=(),
                 termination_reasons=None,
                 search_termination_reason=None):
        super().__init__(root, f_root, split_iters, iters, f_calls, precision, error, converged, optimal,
                         termination_reasons, search_termination_reason)
        self.df_root = list(df_root)

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
               f'split_iters={self.split_iters}, iters={self.iters}, f_calls={self.f_calls}, ' \
               f'precision={self.precision}, error={self.error}, ' \
               f'converged={self.converged}, optimal={self.optimal}, ' \
               f'termination_reasons={self.termination_reasons}, ' \
               f'search_termination_reason={str(self.search_termination_reason)!r})'
