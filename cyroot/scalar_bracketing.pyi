from typing import Any, SupportsFloat, SupportsIndex

from numpy.typing import ArrayLike

from .return_types import RootReturnType, BracketingMethodsReturnType, NewtonMethodsReturnType, SplittingBracketingMethodsReturnType

__all__ = ['bisect', 'hybisect', 'regula_falsi', 'illinois', 'pegasus', 'anderson_bjorck', 'dekker', 'brent', 'chandrupatla', 'ridders', 'toms748', 'wu', 'itp']

def bisect(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, algo: SupportsIndex = ..., etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., **options: Any
) -> BracketingMethodsReturnType: ...

def hybisect(
    f: Any, interval_f: Any, a: SupportsFloat, b: SupportsFloat, *, etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., max_split_iter: SupportsIndex = ..., **options: Any
) -> SplittingBracketingMethodsReturnType: ...

def regula_falsi(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> BracketingMethodsReturnType: ...

def illinois(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> BracketingMethodsReturnType: ...

def pegasus(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> BracketingMethodsReturnType: ...

def anderson_bjorck(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> BracketingMethodsReturnType: ...

def dekker(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> BracketingMethodsReturnType: ...

def brent(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, interp_method: str = ..., sigma: SupportsFloat = ...,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., **options: Any
) -> BracketingMethodsReturnType: ...

def chandrupatla(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, sigma: SupportsFloat = ..., etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., **options: Any
) -> BracketingMethodsReturnType: ...

def ridders(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> BracketingMethodsReturnType: ...

def toms748(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, k: SupportsIndex = ..., mu: SupportsFloat = ...,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., **options: Any
) -> BracketingMethodsReturnType: ...

def wu(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> BracketingMethodsReturnType: ...

def itp(
    f: Any, a: SupportsFloat, b: SupportsFloat, f_a: SupportsFloat | None = ...,
    f_b: SupportsFloat | None = ..., *, k1: SupportsFloat | None = ..., k2: SupportsFloat = ...,
    n0: SupportsIndex = ..., etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> BracketingMethodsReturnType: ...
