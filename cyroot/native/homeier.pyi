from typing import Any, SupportsFloat
from ..return_types import NewtonMethodsReturnType
from .scalar import Newton

class Homeier(Newton): ...

def homeier(function: Any, derivative: Any, x0: SupportsFloat, **options: Any) -> NewtonMethodsReturnType: ...
