from typing import Any, SupportsFloat, SupportsIndex

__all__ = ['DerivativeApproximation', 'FiniteDifference', 'finite_difference']

class DerivativeApproximation:
    n_f_calls: int
    def __init__(self, f: Any) -> None: ...
    def __call__(self, x: SupportsFloat) -> float: ...
    def eval_with_f_val(self, x: SupportsFloat, f_x: SupportsFloat) -> float: ...

class FiniteDifference(DerivativeApproximation):
    h: float
    order: int
    kind: int
    def __init__(self, f: Any, h: SupportsFloat = ..., order: SupportsIndex = ...,
                 kind: SupportsIndex | str = ...) -> None: ...

def finite_difference(f: Any, x: SupportsFloat, f_x: SupportsFloat | None = ...,
                      h: SupportsFloat = ..., order: SupportsIndex = ...,
                      kind: SupportsIndex | str = ...) -> float: ...
