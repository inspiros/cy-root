cy-root
========

A simple root-finding package written in Cython.

## Context

I had to find root of this beast of a function:

$$ f(x) = \frac{1}{\sum_{j=1}^{\inf}{\prod_{k=j}^{\inf}{\frac{1}{k * x + 1}}}} - p $$

## Requirements

- Python 3.8+
- Cython (if you want to build from `.pyx` files)
- numpy
- sympy
- A C/C++ compiler

## Installation

Make sure you have all the dependencies installed, then clone this repo and run:

```bash
git clone git://github.com/inspiros/cy-root.git
cd cy-root
pip install .
```

For uninstalling:

```bash
pip uninstall cy-root
```

## Supported algorithms

For more information about the listed algorithms, please use Google until I update the references.

### Scalar root:

- **Bracketing methods:** (methods that require lower and upper bounds)
    - Bisect
    - Regula Falsi (False Position)
    - Illinois
    - Pegasus
    - Anderson–Björck
    - Dekker
    - Brent
    - Chandrupatla
    - Ridders
    - TOMS748
    - ITP
- **Newton-like methods:** (methods that require derivative and higher order derivatives)
    - Newton-Raphson
    - Halley
    - Householder
- **Quasi-Newton methods:** (methods that approximate derivative or use interpolation)
    - Secant
    - Sidi
    - Steffensen
    - Inverse Quadratic Interpolation
    - Hyperbolic Interpolation
    - Muller

### Tensor root:

Not yet.

## Usage

Example:

```python
from cyroot import find_root_scalar

f = lambda x: x ** 2 - 612
result = find_root_scalar('bisect', f, a=-50, b=50)
print(result)
```

The returned `result` is a tuple with elements depend on the type of methods used:

- Common:
    - `root`: the solved root
    - `f_root`: value evaluated at root
    - `iters`: number of iterations
    - `f_calls`: number of function calls
    - `precision`: width of final bracket (for bracketing methods) or absolute difference of root with the last
      estimation
    - `error`: absolute value of f_root
    - `converged`: `True` if the stopping criterion is met, `False` if the procedure terminated early.
- Exclusive to bracketing methods:
    - `a`: final lower bound
    - `b`: final upper bound
    - `f_a`: value evaluated at final lower bound
    - `f_b`: value evaluated at final upper bound
- Exclusive to Newton-like methods:
    - `df_root`: derivative or tuple of derivatives (of increasing orders) evaluated at root

For more examples, please refer to the `examples` folder.
