from typing import Any, SupportsFloat, SupportsIndex

from numpy.typing import ArrayLike

from .return_types import RootReturnType, BracketingMethodsReturnType, NewtonMethodsReturnType, SplittingBracketingMethodsReturnType

__all__ = ['vrahatis']

def vrahatis(
    F: Any, x0s: ArrayLike, F_x0s: ArrayLike | None = ..., *, h: ArrayLike | None = ...,
    etol: SupportsFloat = ..., ertol: SupportsFloat = ..., ptol: SupportsFloat = ...,
    prtol: SupportsFloat = ..., max_iter: SupportsIndex = ..., **options: Any
) -> BracketingMethodsReturnType: ...

def compute_admissible_n_polygon(
    F: Any, x: ArrayLike, h: ArrayLike | None = ..., eps: SupportsFloat = ..., **options: Any
) -> tuple[Any, Any]: ...

def get_M(
    n: SupportsIndex, sign: bool = ..., *, workspace_bytes: SupportsIndex = ...
) -> Any: ...

def sorted_by_vertices(
    *mats: Any, S: Any
) -> tuple[Any, ...]: ...
