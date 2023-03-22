cy-root ![Build wheels](https://github.com/inspiros/cy-root/actions/workflows/build_wheels.yml/badge.svg) ![PyPI](https://img.shields.io/pypi/v/cy-root) ![GitHub](https://img.shields.io/github/license/inspiros/cy-root)
========

<p align="center">
  <img src="https://www.austintexas.gov/sites/default/files/images/dsd/Community_Trees/tree-root-distruibution-1.jpg" width="300"/>
  <br/>
  <i><font size="1">(Not this root)</font></i>
  <br/>
</p>

A simple root-finding package written in Cython.
Many of the implemented methods can't be found in common Python libraries.

**Context:**

I had to find root of this beast of a function, which has no known bound.

$$ f(x) = \frac{1}{\sum\limits_{j=1}^{\infty} \left(\prod\limits_{k=j}^{\infty} \frac{1}{k \cdot x + 1} \right) } - p $$

Fortunately, Sidi's method came to the rescue.

## Requirements

- Python 3.6+
- Cython (if you want to build from `.pyx` files)
- numpy
- sympy
- A C/C++ compiler

## Installation

[cy-root](https://pypi.org/project/cy-root/) is now available on PyPI.

```bash
pip install cy-root
```

Alternatively, you can build from source.
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

**Note:**
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
    - Wu
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
    - Muller _(for complex root)_

### Vector root:

Not yet.

## Usage

#### Some examples:

Use `find_root_scalar` and pass method name as first argument.

```python
from cyroot import find_root_scalar

f = lambda x: x ** 2 - 612
result = find_root_scalar(method='itp', f=f, a=-10, b=50)
print(result)
```

Output:

> ```RootResults(root=24.73863375370596, f_root=-1.1368683772161603e-13, iters=8, f_calls=11, a=24.73863375370596, b=24.73863375370596, f_a=-1.1368683772161603e-13, f_b=-1.1368683772161603e-13, precision=6.353294779160024e-08, error=1.1368683772161603e-13, converged=True, optimal=True)```

A dictionary containing names and pointers to all the (scalar) methods are stored in `SCALAR_ROOT_FINDING_METHODS`.

Alternatively, import the function directly. You can also see the full list of input arguments of by using `help()` on
them.

```python
from cyroot import muller

# This function has no real root
f = lambda x: x ** 4 + 4 * x ** 2 + 5
# But Muller's method can be used to find complex root
result = muller(f, x0=0, x1=10, x2=20)
print(result)
```

Output:

> ```RootResults(root=(0.34356074972251255+1.4553466902253551j), f_root=(-8.881784197001252e-16-1.7763568394002505e-15j), iters=43, f_calls=43, precision=3.177770418807502e-08, error=1.9860273225978185e-15, converged=True, optimal=True)```

#### Output format:

The returned `result` is a namedtuple whose elements depend on the type of the method:

- Common:
    - `root`: the solved root.
    - `f_root`: value evaluated at root.
    - `iters`: number of iterations.
    - `f_calls`: number of function calls.
    - `precision`: width of final bracket (for bracketing methods) or absolute difference of root with the last
      estimation.
    - `error`: absolute value of `f_root`.
    - `converged`: `True` if the stopping criterion is met, `False` if the procedure terminated prematurely.
    - `optimal`: `True` only if the error tolerance is satisfied `abs(f_root) <= etol`.
- Exclusive to bracketing methods:
    - `a`: final lower bound.
    - `b`: final upper bound.
    - `f_a`: value evaluated at final lower bound.
    - `f_b`: value evaluated at final upper bound.
- Exclusive to Newton-like methods:
    - `df_root`: derivative or tuple of derivatives (of increasing orders) evaluated at root.

**Note**: `converged` might sometimes be `True` even if the solution is not optimal, which means the routine
stopped because the precision tolerance is satisfied.

**Configurations:**

The default values for stop condition arguments (i.e. `etol`, `ptol`, `max_iter`) are globally set to the values defined
in [`_defaults.py`](cyroot/_defaults.py). They can be modified dynamically, docstrings of all functions using them
will also be updated automatically.

```python
import cyroot

cyroot.set_default_stop_condition_args(
    etol=1e-7,
    ptol=0,  # disable precision tolerance
    max_iter=100)

help(cyroot.illinois)  # run to check the updated docstring
```

For more examples, please check the [`examples`](examples) folder.

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.

## References

https://en.wikipedia.org/wiki/Root-finding_algorithms
