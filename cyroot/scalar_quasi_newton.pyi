from typing import Any, SupportsComplex, SupportsFloat, SupportsIndex

from numpy.typing import ArrayLike

from .return_types import RootReturnType, BracketingMethodsReturnType, NewtonMethodsReturnType, SplittingBracketingMethodsReturnType

__all__ = ['secant', 'sidi', 'steffensen', 'inverse_quadratic_interp', 'hyperbolic_interp', 'muller']

def secant(
    f: Any, x0: SupportsFloat, x1: SupportsFloat, f_x0: SupportsFloat | None = ...,
    f_x1: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> RootReturnType: ...

def sidi(
    f: Any, x0s: ArrayLike, f_x0s: ArrayLike | None = ..., *, etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., **options: Any
) -> RootReturnType: ...

def steffensen(
    f: Any, x0: SupportsFloat, f_x0: SupportsFloat | None = ..., *, aitken: bool = ...,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., **options: Any
) -> RootReturnType: ...

def inverse_quadratic_interp(
    f: Any, x0: SupportsFloat, x1: SupportsFloat, x2: SupportsFloat,
    f_x0: SupportsFloat | None = ..., f_x1: SupportsFloat | None = ...,
    f_x2: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> RootReturnType: ...

def hyperbolic_interp(
    f: Any, x0: SupportsFloat, x1: SupportsFloat, x2: SupportsFloat,
    f_x0: SupportsFloat | None = ..., f_x1: SupportsFloat | None = ...,
    f_x2: SupportsFloat | None = ..., *, etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> RootReturnType: ...

def muller(
    f: Any, x0: SupportsFloat | SupportsComplex, x1: SupportsFloat | SupportsComplex, x2: SupportsFloat | SupportsComplex,
    f_x0: SupportsFloat | SupportsComplex | None = ..., f_x1: SupportsFloat | SupportsComplex | None = ...,
    f_x2: SupportsFloat | SupportsComplex | None = ..., *, etol: SupportsFloat | SupportsComplex = ..., ertol: SupportsFloat | SupportsComplex = ...,
    ptol: SupportsFloat | SupportsComplex = ..., prtol: SupportsFloat | SupportsComplex = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> RootReturnType: ...
