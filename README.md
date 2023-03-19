## cy-root

<div align="center">
  <img src="https://www.austintexas.gov/sites/default/files/images/dsd/Community_Trees/tree-root-distruibution-1.jpg" width="300"/>
  <br>
  <i><span style="font-size: xx-small; ">(Not this root)</span></i>
</div>

A simple root-finding package written in Cython.
Not a serious one so please only use these codes as learning materials.
Many of the implemented methods here can't be found in common Python libraries.

**Context:**

I had to find root of this beast of a function, which has no known bound, and Sidi's method came to rescue:

```math
f(x) = \frac{1}{\sum\limits_{j=1}^{\infty} \left(\prod\limits_{k=j}^{\infty} \frac{1}{k \cdot x + 1} \right) } - p
```

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

**Note:** For more information about the listed algorithms, please use Google until I update the references.

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
    - Muller _(for complex root)_

### Vector root:

Not yet.

## Usage

**Some examples:**

Use `find_root_scalar` and pass method name as first argument.

```python
from cyroot import find_root_scalar

f = lambda x: x ** 2 - 612
result = find_root_scalar(method='itp', f=f, a=-50, b=50)
print(result)
```

Output:

```console
RootResults(root=24.738633753705507, f_root=-2.262368070660159e-11, iters=47, f_calls=48, a=24.73863375370501, b=24.738633753706008, f_a=-4.718003765447065e-11, f_b=2.1600499167107046e-12, precision=4.991562718714704e-13, error=2.262368070660159e-11, converged=True)
```

Alternatively, import the function directly. You can also see the full input arguments of by using `help()` on them.

```python
from cyroot import muller

# This function has no real root
f = lambda x: x ** 4 + 4 * x ** 2 + 5
# But Muller's method can be used to find complex root
result = muller(f, x0=0, x1=10, x2=20)
print(result)
```

Output:

```console
RootResults(root=(0.34356074972251255+1.4553466902253551j), f_root=(-8.881784197001252e-16-1.7763568394002505e-15j), iters=43, f_calls=43, precision=3.177770418807502e-08, error=1.9860273225978185e-15, converged=True)
```

The returned `result` is a namedtuple whose elements depend on the type of the method:

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

**Note**: `result.converged` might be `True` even if the solution is not optimal, which means the routine stopped
because the precision tolerance is satisfied. I will add another flag for optimality of the root later.

For more examples, please refer to the `examples` folder.
