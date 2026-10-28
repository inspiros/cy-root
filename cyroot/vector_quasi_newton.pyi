from .native.broyden import BroydenResult
from .native.vector import WolfeBittnerResult
from typing import Any, SupportsFloat, SupportsIndex

from numpy.typing import ArrayLike

from .return_types import RootReturnType, BracketingMethodsReturnType, NewtonMethodsReturnType, SplittingBracketingMethodsReturnType

__all__ = ['wolfe_bittner', 'robinson', 'barnes', 'traub_steffensen', 'broyden', 'klement']

def wolfe_bittner(
    F: Any, x0s: ArrayLike, F_x0s: ArrayLike | None = ..., *, rcond: SupportsFloat = ...,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
    history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
    workspace_bytes: SupportsIndex = ...
) -> WolfeBittnerResult: ...

def robinson(
    F: Any, x0: ArrayLike, x1: ArrayLike, F_x0: ArrayLike | None = ...,
    F_x1: ArrayLike | None = ..., *, orth: str | int = ..., etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., history: SupportsIndex = ..., history_every: SupportsIndex = ...,
    history_bytes: SupportsIndex = ..., workspace_bytes: SupportsIndex = ...
) -> NewtonMethodsReturnType: ...

def barnes(
    F: Any, x0: ArrayLike, F_x0: ArrayLike | None = ..., J_x0: Any = ..., *,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
    history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
    workspace_bytes: SupportsIndex = ...
) -> BroydenResult: ...

def traub_steffensen(
    F: Any, x0: ArrayLike, F_x0: ArrayLike | None = ..., *, etol: SupportsFloat = ...,
    ertol: SupportsFloat = ..., ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
    max_iter: SupportsIndex = ..., history: SupportsIndex = ..., history_every: SupportsIndex = ...,
    history_bytes: SupportsIndex = ..., workspace_bytes: SupportsIndex = ...
) -> NewtonMethodsReturnType: ...

def broyden(
    F: Any, x0: ArrayLike, F_x0: ArrayLike | None = ..., J_x0: Any = ..., *,
    algo: int | str = ..., etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
    ptol: SupportsFloat = ..., prtol: SupportsFloat = ..., max_iter: SupportsIndex = ...,
    history: SupportsIndex = ..., history_every: SupportsIndex = ...,
    history_bytes: SupportsIndex = ..., workspace_bytes: SupportsIndex = ...
) -> BroydenResult: ...

def klement(
    F: Any, x0: ArrayLike, F_x0: ArrayLike | None = ..., J_x0: Any = ..., *,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
    history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...,
    workspace_bytes: SupportsIndex = ...
) -> BroydenResult: ...
