from typing import Any, SupportsFloat, SupportsIndex

from numpy.typing import ArrayLike

from .return_types import RootReturnType, BracketingMethodsReturnType, NewtonMethodsReturnType, SplittingBracketingMethodsReturnType

__all__ = ['generalized_newton', 'generalized_halley', 'generalized_super_halley', 'generalized_chebyshev', 'generalized_tangent_hyperbolas']

def generalized_newton(
    F: Any, J: Any, x0: ArrayLike, F_x0: ArrayLike | None = ..., J_x0: ArrayLike | None = ..., *,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
    history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
    workspace_bytes: SupportsIndex = ...
) -> NewtonMethodsReturnType: ...

def generalized_halley(
    F: Any, J: Any, H: Any, x0: ArrayLike, F_x0: ArrayLike | None = ...,
    J_x0: ArrayLike | None = ..., H_x0: ArrayLike | None = ..., *, alpha: SupportsFloat | None = ...,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
    history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
    workspace_bytes: SupportsIndex = ...
) -> NewtonMethodsReturnType: ...

def generalized_super_halley(
    F: Any, J: Any, H: Any, x0: ArrayLike, F_x0: ArrayLike | None = ...,
    J_x0: ArrayLike | None = ..., H_x0: ArrayLike | None = ..., *, etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., history: SupportsIndex = ..., history_every: SupportsIndex = ...,
    history_bytes: SupportsIndex = ..., workspace_bytes: SupportsIndex = ...
) -> NewtonMethodsReturnType: ...

def generalized_chebyshev(
    F: Any, J: Any, H: Any, x0: ArrayLike, F_x0: ArrayLike | None = ...,
    J_x0: ArrayLike | None = ..., H_x0: ArrayLike | None = ..., *, etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., history: SupportsIndex = ..., history_every: SupportsIndex = ...,
    history_bytes: SupportsIndex = ..., workspace_bytes: SupportsIndex = ...
) -> NewtonMethodsReturnType: ...

def generalized_tangent_hyperbolas(
    F: Any, J: Any, H: Any, x0: ArrayLike, F_x0: ArrayLike | None = ...,
    J_x0: ArrayLike | None = ..., H_x0: ArrayLike | None = ..., *, etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., history: SupportsIndex = ..., history_every: SupportsIndex = ...,
    history_bytes: SupportsIndex = ..., workspace_bytes: SupportsIndex = ...
) -> NewtonMethodsReturnType: ...
