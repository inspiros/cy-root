from collections.abc import Iterator
from typing import Any, NamedTuple, SupportsFloat, SupportsIndex
from ..status import TerminationReason
from ..return_types import NewtonMethodsReturnType

class HigherOrderStep(NamedTuple):
    root: float
    f_root: float
    df_root: float
    d2f_root: float
    iters: int
    f_calls: tuple[int, int, int]
    precision: float
    error: float
    termination_reason: TerminationReason | None
    callback_code: int
    callback_stage: str | None

class _ThirdOrderNewton:
    def step(self) -> HigherOrderStep: ...
    def run(self) -> NewtonMethodsReturnType: ...
    @property
    def result(self) -> NewtonMethodsReturnType: ...
    @property
    def history(self) -> tuple[HigherOrderStep, ...]: ...
    @property
    def done(self) -> bool: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def callback_code(self) -> int: ...
    @property
    def callback_stage(self) -> str | None: ...
    def __iter__(self) -> Iterator[HigherOrderStep]: ...
    def __next__(self) -> HigherOrderStep: ...

class Halley(_ThirdOrderNewton):
    def __init__(self, function: Any, derivative: Any, second_derivative: Any, x0: SupportsFloat,
                 f_x0: SupportsFloat | None = ..., df_x0: SupportsFloat | None = ...,
                 d2f_x0: SupportsFloat | None = ..., *,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...) -> None: ...

class Chebyshev(_ThirdOrderNewton):
    def __init__(self, function: Any, derivative: Any, second_derivative: Any, x0: SupportsFloat,
                 f_x0: SupportsFloat | None = ..., df_x0: SupportsFloat | None = ...,
                 d2f_x0: SupportsFloat | None = ..., *,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...) -> None: ...

class ModifiedHalley(_ThirdOrderNewton):
    def __init__(self, function: Any, derivative: Any, second_derivative: Any, x0: SupportsFloat,
                 f_x0: SupportsFloat | None = ..., df_x0: SupportsFloat | None = ...,
                 d2f_x0: SupportsFloat | None = ..., *,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ..., alpha: SupportsFloat = ...) -> None: ...

class SuperHalley(_ThirdOrderNewton):
    def __init__(self, function: Any, derivative: Any, second_derivative: Any, x0: SupportsFloat,
                 f_x0: SupportsFloat | None = ..., df_x0: SupportsFloat | None = ...,
                 d2f_x0: SupportsFloat | None = ..., *,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...) -> None: ...

class TangentHyperbolas(Halley): ...

def halley(function: Any, derivative: Any, second_derivative: Any, x0: SupportsFloat,
           **options: Any) -> NewtonMethodsReturnType: ...
def chebyshev(function: Any, derivative: Any, second_derivative: Any, x0: SupportsFloat,
              **options: Any) -> NewtonMethodsReturnType: ...
def modified_halley(function: Any, derivative: Any, second_derivative: Any, x0: SupportsFloat, *,
                    alpha: SupportsFloat = ..., **options: Any) -> NewtonMethodsReturnType: ...
def super_halley(function: Any, derivative: Any, second_derivative: Any, x0: SupportsFloat,
                 **options: Any) -> NewtonMethodsReturnType: ...
def tangent_hyperbolas(function: Any, derivative: Any, second_derivative: Any, x0: SupportsFloat,
                       **options: Any) -> NewtonMethodsReturnType: ...
