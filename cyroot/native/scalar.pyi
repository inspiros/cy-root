from collections.abc import Iterator
from typing import Any, NamedTuple, SupportsFloat, SupportsIndex
from ..status import TerminationReason
from ..return_types import BracketingMethodsReturnType, NewtonMethodsReturnType

class ScalarStep(NamedTuple):
    root: float
    f_root: float
    bracket: tuple[float, float]
    f_bracket: tuple[float, float]
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: TerminationReason | None
    callback_code: int

class NativeCallbackError(RuntimeError):
    native_status: int
    result: Any
    callback_stage: str | None
    def __init__(self, native_status: int, result: Any, callback_stage: str | None = ...) -> None: ...

class NewtonStep(NamedTuple):
    root: float
    f_root: float
    df_root: float
    iters: int
    f_calls: tuple[int, int]
    precision: float
    error: float
    termination_reason: TerminationReason | None
    callback_code: int
    callback_stage: str | None

class ScalarCallback:
    def __init__(self, function: Any) -> None: ...
    @classmethod
    def from_capsule(cls, capsule: Any, keepalive: Any = ...) -> ScalarCallback: ...
    @property
    def native(self) -> bool: ...

class Bisection:
    def __init__(self, function: Any, a: SupportsFloat, b: SupportsFloat,
                 f_a: SupportsFloat | None = ..., f_b: SupportsFloat | None = ...,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...) -> None: ...
    def step(self) -> ScalarStep: ...
    def run(self) -> BracketingMethodsReturnType: ...
    @property
    def result(self) -> BracketingMethodsReturnType: ...
    @property
    def done(self) -> bool: ...
    @property
    def history(self) -> tuple[ScalarStep, ...]: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def callback_code(self) -> int: ...
    def __iter__(self) -> Iterator[ScalarStep]: ...
    def __next__(self) -> ScalarStep: ...

class Newton:
    def __init__(self, function: Any, derivative: Any, x0: SupportsFloat,
                 f_x0: SupportsFloat | None = ..., df_x0: SupportsFloat | None = ...,
                 etol: SupportsFloat = ..., ertol: SupportsFloat = ...,
                 ptol: SupportsFloat = ..., prtol: SupportsFloat = ...,
                 max_iter: SupportsIndex = ..., history: SupportsIndex = ...,
                 history_every: SupportsIndex = ..., history_bytes: SupportsIndex = ...) -> None: ...
    def step(self) -> NewtonStep: ...
    def run(self) -> NewtonMethodsReturnType: ...
    @property
    def result(self) -> NewtonMethodsReturnType: ...
    @property
    def done(self) -> bool: ...
    @property
    def history(self) -> tuple[NewtonStep, ...]: ...
    @property
    def history_truncated(self) -> bool: ...
    @property
    def callback_code(self) -> int: ...
    @property
    def callback_stage(self) -> str | None: ...
    def __iter__(self) -> Iterator[NewtonStep]: ...
    def __next__(self) -> NewtonStep: ...

def bisect(function: Any, a: SupportsFloat, b: SupportsFloat, **options: Any) -> BracketingMethodsReturnType: ...
def newton(function: Any, derivative: Any, x0: SupportsFloat, **options: Any) -> NewtonMethodsReturnType: ...
