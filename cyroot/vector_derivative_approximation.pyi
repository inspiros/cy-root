from typing import Any, SupportsIndex
import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ['VectorDerivativeApproximation', 'GeneralizedFiniteDifference', 'generalized_finite_difference']

class VectorDerivativeApproximation:
    n_f_calls: int
    def __init__(self, F: Any) -> None: ...
    def __call__(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def eval_with_f_val(self, x: NDArray[Any], F_x: NDArray[Any]) -> NDArray[np.float64]: ...

class GeneralizedFiniteDifference(VectorDerivativeApproximation):
    h: Any
    order: int
    kind: int
    def __init__(self, F: Any, h: ArrayLike = ..., order: SupportsIndex = ...,
                 kind: SupportsIndex | str = ...) -> None: ...

def generalized_finite_difference(F: Any, x: ArrayLike, F_x: ArrayLike | None = ...,
                                  h: ArrayLike = ..., order: SupportsIndex = ...,
                                  kind: SupportsIndex | str = ...) -> NDArray[np.float64]: ...
