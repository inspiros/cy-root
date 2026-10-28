from typing import Any, SupportsFloat, SupportsIndex

from numpy.typing import ArrayLike

from .return_types import RootReturnType, BracketingMethodsReturnType, NewtonMethodsReturnType, SplittingBracketingMethodsReturnType

__all__ = ['newton', 'homeier', 'halley', 'super_halley', 'chebyshev', 'tangent_hyperbolas', 'householder']

def newton(
    f: Any, df: Any, x0: SupportsFloat, f_x0: SupportsFloat | None = ...,
    df_x0: SupportsFloat | None = ..., *, h: SupportsFloat | None = ..., etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., **options: Any
) -> NewtonMethodsReturnType: ...

def homeier(
    f: Any, df: Any, x0: SupportsFloat, f_x0: SupportsFloat | None = ...,
    df_x0: SupportsFloat | None = ..., *, h: SupportsFloat | None = ..., etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., **options: Any
) -> NewtonMethodsReturnType: ...

def halley(
    f: Any, df: Any, d2f: Any, x0: SupportsFloat, f_x0: SupportsFloat | None = ...,
    df_x0: SupportsFloat | None = ..., d2f_x0: SupportsFloat | None = ..., *,
    alpha: SupportsFloat | None = ..., h: SupportsFloat | None = ..., etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., **options: Any
) -> NewtonMethodsReturnType: ...

def super_halley(
    f: Any, df: Any, d2f: Any, x0: SupportsFloat, f_x0: SupportsFloat | None = ...,
    df_x0: SupportsFloat | None = ..., d2f_x0: SupportsFloat | None = ..., *,
    h: SupportsFloat | None = ..., etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> NewtonMethodsReturnType: ...

def chebyshev(
    f: Any, df: Any, d2f: Any, x0: SupportsFloat, f_x0: SupportsFloat | None = ...,
    df_x0: SupportsFloat | None = ..., d2f_x0: SupportsFloat | None = ..., *,
    h: SupportsFloat | None = ..., etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    **options: Any
) -> NewtonMethodsReturnType: ...

def tangent_hyperbolas(
    f: Any, df: Any, d2f: Any, x0: SupportsFloat, f_x0: SupportsFloat | None = ...,
    df_x0: SupportsFloat | None = ..., d2f_x0: SupportsFloat | None = ..., *,
    formula: SupportsIndex = ..., h: SupportsFloat | None = ..., etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., **options: Any
) -> NewtonMethodsReturnType: ...

def householder(
    f: Any, dfs: Any, x0: SupportsFloat, f_x0: SupportsFloat | None = ...,
    dfs_x0: ArrayLike | None = ..., *, d: SupportsIndex | None = ..., h: SupportsFloat | None = ...,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., **options: Any
) -> NewtonMethodsReturnType: ...

def ahmadi(
    f: Any, dfs: Any, x0: SupportsFloat, f_x0: SupportsFloat | None = ...,
    dfs_x0: ArrayLike | None = ..., d: SupportsIndex = ..., h: SupportsFloat = ...,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...
) -> NewtonMethodsReturnType: ...

AD3DFPtr: Any

AhmadiDeltaFuncFactory: Any
