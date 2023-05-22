![logo](https://raw.githubusercontent.com/inspiros/cy-root/master/resources/logo.png) cy-root [![Build wheels](https://github.com/inspiros/cy-root/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/inspiros/cy-root/actions) [![PyPI](https://img.shields.io/pypi/v/cy-root)](https://pypi.org/project/cy-root) [![PyPI - Downloads](https://img.shields.io/pypi/dm/cy-root)](https://pypi.org/project/cy-root) [![GitHub](https://img.shields.io/github/license/inspiros/cy-root)](https://github.com/inspiros/cy-root/blob/master/LICENSE.txt)
========

<p align="center">
  <img src="https://www.austintexas.gov/sites/default/files/images/dsd/Community_Trees/tree-root-distruibution-1.jpg" width="300"/>
  <br/>
  <i>(Not this root)</i>
  <br/>
</p>

A simple root-finding package written in Cython.
Many of the implemented methods can't be found in common Python libraries.

## News:

- **v1.0.3:** All methods now return a _partially_ typed `namedtuple`-like Cython Extension object instead of
  `namedtuple`.
- **v1.0.2:** Vector root-finding methods can now _(try to)_ solve systems of equations with number of inputs different
  from number of outputs.

## Requirements

- Python 3.6+
- dynamic-default-args
- numpy
- scipy
- sympy

#### For compilation:

- Cython (if you want to build from `.pyx` files)
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

## Supported algorithms

**Note:**
For more information about the listed algorithms, please use Google until I update the references.

### Scalar root:

- **Bracketing methods:** (methods that require lower and upper bounds)
    - ✅ Bisect
    - ✅ [Hybisect](https://dl.acm.org/doi/10.1145/3437120.3437324) _(bisection with interval analysis)_
    - ✅ Regula Falsi
    - ✅ Illinois
    - ✅ Pegasus
    - ✅ Anderson–Björck
    - ✅ Dekker
    - ✅ Brent _(with Inverse Quadratic Interpolation and Hyperbolic Interpolation)_
    - ✅ [Chandrupatla](https://dl.acm.org/doi/10.1016/S0965-9978%2896%2900051-8)
    - ✅ [Ridders](https://doi.org/10.1109/TCS.1979.1084580)
    - ✅ [TOMS748](https://dl.acm.org/doi/10.1145/210089.210111)
    - ✅ [Wu](https://doi.org/10.1016/j.amc.2004.04.120)
    - ✅ [ITP](https://dl.acm.org/doi/10.1145/3423597)
- **Newton-like methods:** (methods that require derivative and/or higher order derivatives)
    - ✅ Newton
    - ✅ Chebyshev
    - ✅ Halley
    - ✅ Super-Halley
    - ✅ Tangent Hyperbolas _(similar to Halley)_
    - ✅ Householder
- **Quasi-Newton methods:** (methods that approximate derivative, use interpolation, or successive iteration)
    - ✅ Secant
    - ✅ Sidi
    - ✅ Steffensen
    - ✅ Inverse Quadratic Interpolation
    - ✅ Hyperbolic Interpolation
    - ✅ [Muller](https://www.ams.org/journals/mcom/1956-10-056/S0025-5718-1956-0083822-0/) _(for complex root)_

### Vector root:

- **Bracketing methods:** (methods that require n-dimensional bracket)
    - ✅ [Vrahatis](https://doi.org/10.1007/BF01389620) _(generalized bisection using n-polygon)_

[//]: # (    - ⬜ [Eiger-Sikorski-Stenger]&#40;https://dl.acm.org/doi/10.1145/2701.2705&#41; _&#40;planned&#41;_)
- **Newton-like methods:** (methods that require Jacobian and/or Hessian)
    - ✅ Generalized Newton
    - ✅ Generalized Chebyshev
    - ✅ Generalized Halley
    - ✅ Generalized Super-Halley
    - ✅ Generalized Tangent Hyperbolas _(similar to Generalized Halley)_
- **Quasi-Newton methods:** (methods that approximate Jacobian, use interpolation, or successive iteration)
    - ✅ [Wolfe-Bittner](https://doi.org/10.1145/368518.368542)
    - ✅ [Robinson](https://epubs.siam.org/doi/abs/10.1137/0703057)
    - ✅ [Barnes](https://academic.oup.com/comjnl/article/8/1/66/489886)
    - ✅ Traub-Steffensen
    - ✅ [Broyden](https://doi.org/10.2307/2003941) _(Good and Bad)_
    - ✅ [Klement](https://jatm.com.br/jatm/article/view/373)

#### Derivative Approximation:

Methods that can be combined with any Newton-like root-finding methods to discard the need of analytical derivatives.

- ✅ Finite Difference _(for both scalar and vector functions, up to arbitrary order)_

## Usage

### Examples:

#### Example 1:

Use `find_scalar_root` or `find_vector_root` and pass method name as the first argument.
This example shows the use of `find_scalar_root` function with `itp` method.

```python
from cyroot import find_scalar_root

f = lambda x: x ** 2 - 612
result = find_scalar_root(method='itp', f=f, a=-10, b=50)
print(result)
```

Output:

```
RootResults(root=24.73863375370596, f_root=-1.1368683772161603e-13, iters=8, f_calls=10, bracket=(24.73863369031373, 24.738633753846678), f_bracket=(-3.1364744472739403e-06, 6.962181942071766e-09), precision=6.353294779160024e-08, error=1.1368683772161603e-13, converged=True, optimal=True)
```

The names and pointers to all implemented methods are stored in two dictionaries `SCALAR_ROOT_FINDING_METHODS` and
`VECTOR_ROOT_FINDING_METHODS`.

```python
from cyroot import SCALAR_ROOT_FINDING_METHODS, VECTOR_ROOT_FINDING_METHODS

print('scalar root methods:', SCALAR_ROOT_FINDING_METHODS.keys())
print('vector root methods:', VECTOR_ROOT_FINDING_METHODS.keys())
```

#### Example 2:

Alternatively, import the function directly.
You can also see the full list of input arguments of by using `help()` on them.

This example shows the use of `muller` method for finding complex root:

```python
from cyroot import muller

# This function has no real root
f = lambda x: x ** 4 + 4 * x ** 2 + 5
# But Muller's method can be used to find complex root
result = muller(f, x0=0, x1=10, x2=20)
print(result)
```

Output:

```
RootResults(root=(0.34356074972251255+1.4553466902253551j), f_root=(-8.881784197001252e-16-1.7763568394002505e-15j), iters=43, f_calls=43, precision=3.177770418807502e-08, error=1.9860273225978185e-15, converged=True, optimal=True)
```

#### Example 3:

Considering the parabola $f(x)=x^2-612$ in **Example 1** with initial bounds $(a,b)$ where $a=-b$, many bracketing
methods will fail to find a root as the values evaluated at initial bracket are identical.

In this example, we use the `hybisect` method which repeatedly bisects the search regions until the Bolzano criterion
holds, thus can find multiple roots:

```python
import math

from cyroot import hybisect

f = lambda x: x ** 2 - 612
# interval arithmetic function of f
interval_f = lambda x_l, x_h: ((min(abs(x_l), abs(x_h))
                                if math.copysign(1, x_l) * math.copysign(1, x_h) > 0
                                else 0) ** 2 - 612,
                               max(abs(x_l), abs(x_h)) ** 2 - 612)

result = hybisect(f, interval_f, -50, 50)
print(result)
```

Output:

```
RootResults(root=[-24.738633753707973, 24.738633753707973], f_root=[9.936229616869241e-11, 9.936229616869241e-11], split_iters=1, iters=[43, 43], f_calls=(92, 3), bracket=[(-24.738633753710815, -24.73863375370513), (24.73863375370513, 24.738633753710815)], f_bracket=[(nan, nan), (nan, nan)], precision=[5.6843418860808015e-12, 5.6843418860808015e-12], error=[9.936229616869241e-11, 9.936229616869241e-11], converged=[True, True], optimal=[True, True])
```

#### Example 4:

This example shows the use of the `halley` method with functions returning first and second order derivatives of `f`:

```python
from cyroot import halley

f = lambda x: x ** 3 - 5 * x ** 2 + 2 * x - 1
# first order derivative
df = lambda x: 3 * x ** 2 - 10 * x + 2
# second order derivative
d2f = lambda x: 6 * x - 10

result = halley(f, df, d2f, x0=1.5)
print(result)
```

Output:

```
RootResults(root=4.613470267581537, f_root=-3.623767952376511e-13, df_root=(19.7176210537612, 17.68082160548922), iters=11, f_calls=(12, 12, 12), precision=4.9625634836147965e-05, error=3.623767952376511e-13, converged=True, optimal=True)
```

The `householder` method supports an arbitrary number of higher order derivatives:

```python
from cyroot import householder

f = lambda x: x ** 3 - 5 * x ** 2 + 2 * x - 1
df = lambda x: 3 * x ** 2 - 10 * x + 2
d2f = lambda x: 6 * x - 10
d3f = lambda x: 6

result = householder(f, dfs=[df, d2f, d3f], x0=1.5)
print(result)  # same result
```

#### Example 5:

Similarly, to find roots of systems of equations with Newton-like methods, you have to define functions returning
**Jacobian** (and **Hessian**) of `F`.

This example shows the use of `generalized_super_halley` method:

```python
import numpy as np

from cyroot import generalized_super_halley

# all functions for vector root methods must take a numpy array
# as argument, and return an array-like object
F = lambda x: np.array([x[0] ** 2 + 2 * x[0] * np.sin(x[1]) - x[1],
                        4 * x[0] * x[1] ** 2 - x[1] ** 3 - 1])
# Jacobian
J = lambda x: np.array([
    [2 * x[0] + 2 * np.sin(x[1]), 2 * x[0] * np.cos(x[1]) - 1],
    [4 * x[1] ** 2, 8 * x[0] * x[1] - 3 * x[1] ** 2]
])
# Hessian
H = lambda x: np.array([
    [[2, 2 * np.cos(x[1])],
     [2 * np.cos(x[1]), -2 * x[0] * np.sin(x[1])]],
    [[0, 8 * x[1]],
     [8 * x[1], 8 * x[0] - 6 * x[1]]]
])

result = generalized_super_halley(F, J, H, x0=np.array([2., 2.]))
print(result)
```

Output: _(a bit messy)_

```
RootResults(root=array([0.48298601, 1.08951589]), f_root=array([-4.35123049e-11, -6.55444587e-11]), df_root=(array([[ 2.73877785, -0.55283751],
       [ 4.74817951,  0.6486328 ]]), array([[[ 2.        ,  0.92582907],
        [ 0.92582907, -0.85624041]],

       [[ 0.        ,  8.71612713],
        [ 8.71612713, -2.6732073 ]]])), iters=3, f_calls=(4, 4, 4), precision=0.0005808146393164461, error=6.554445874940029e-11, converged=True, optimal=True)
```

#### Example 6:

For vector bracketing root methods or vector root methods with multiple initial guesses, the input should be a 2D
`np.ndarray`.

This example shows the use of `vrahatis` method (a generalized bisection) with the example function in the original
paper:

```python
import numpy as np

from cyroot import vrahatis

F = lambda x: np.array([x[0] ** 2 - 4 * x[1],
                        -2 * x[0] + x[1] ** 2 + 4 * x[1]])

# If the initial points do not form an admissible n-polygon,
# an exception will be raised.
x0s = np.array([[-2., -0.25],
                [0.5, 0.25],
                [2, -0.25],
                [0.6, 0.25]])

result = vrahatis(F, x0s=x0s)
print(result)
```

Output:

```
RootResults(root=array([4.80212874e-11, 0.00000000e+00]), f_root=array([ 2.30604404e-21, -9.60425747e-11]), iters=34, f_calls=140, bracket=array([[ 2.29193750e-10,  2.91038305e-11],
       [-6.54727619e-12,  2.91038305e-11],
       [ 4.80212874e-11,  0.00000000e+00],
       [-6.98492260e-11,  0.00000000e+00]]), f_bracket=array([[-1.16415322e-10, -3.41972179e-10],
       [-1.16415322e-10,  1.29509874e-10],
       [ 2.30604404e-21, -9.60425747e-11],
       [ 4.87891437e-21,  1.39698452e-10]]), precision=2.9904297647806717e-10, error=9.604257471622717e-11, converged=True, optimal=True)
```

#### Example 7:

This example shows the use of `finite_difference` to approximate derivatives when analytical solutions are not
available:

```python
import math

from cyroot import finite_difference

f = lambda x: (math.sin(x) + 1) ** x
x = 3 * math.pi / 2
d3f_x = finite_difference(f, x,
                          h=1e-4,  # step
                          order=1,  # order
                          kind='forward')  # type: forward, backward, or central
# 7.611804179666343e-36
```

Similarly, `generalized_finite_difference` can compute vector derivative of arbitrary order
(`order=1` for **Jacobian**, `order=2` for **Hessian**), and `h` can be a number or a `np.ndarray` containing different
step sizes for each dimension:

```python
import numpy as np

from cyroot import generalized_finite_difference

F = lambda x: np.array([x[0] ** 3 - 3 * x[0] * x[1] + 5 * x[1] - 7,
                        x[0] ** 2 + x[0] * x[1] ** 2 - 4 * x[1] ** 2 + 3.5])
x = np.array([2., 3.])

# Derivatives of F will have shape (m, *([n] * order))
# where n is number of inputs, m is number of outputs
J_x = generalized_finite_difference(F, x, h=1e-4, order=1)  # Jacobian
# array([[  2.99985,  -1.00015],
#        [ 13.0003 , -11.9997 ]])
H_x = generalized_finite_difference(F, x, h=1e-3, order=2)  # Hessian
# array([[[12.   , -3.   ],
#         [-3.   ,  0.   ]],
#        [[ 2.   ,  6.001],
#         [ 6.001, -3.998]]])
K_x = generalized_finite_difference(F, x, h=1e-2, order=3)  # Kardashian, maybe
# array([[[[ 6.00000000e+00,  2.32830644e-10],
#          [ 2.32830644e-10,  2.32830644e-10]],
#         [[ 2.32830644e-10,  2.32830644e-10],
#          [ 2.32830644e-10,  1.11758709e-08]]],
#        [[[ 0.00000000e+00, -3.72529030e-09],
#          [-3.72529030e-09,  1.99999999e+00]],
#         [[-3.72529030e-09,  1.99999999e+00],
#          [ 1.99999999e+00, -1.67638063e-08]]]])
```

Conveniently, you can use the `FiniteDifference` and `GeneralizedFiniteDifference` classes to wrap our function and
pass them to any Newton-like methods.

This is actually the default behavior when derivative functions of all Newton-like methods or the initial Jacobian
guess of some vector quasi-Newton methods are not provided.

```python
from cyroot import GeneralizedFiniteDifference, generalized_halley

J = GeneralizedFiniteDifference(F, h=1e-4, order=1)
H = GeneralizedFiniteDifference(F, h=1e-3, order=2)

result = generalized_halley(F, J=J, H=H, x0=x)
print(result)
```

Output:

```
RootResults(root=array([2.16665878, 2.11415683]), f_root=array([-5.47455414e-11,  1.05089271e-11]), df_root=(array([[ 7.74141032, -1.49997634],
       [ 8.80307666, -7.75212506]]), array([[[ 1.30059527e+01, -3.00000000e+00],
        [-3.00000000e+00, -4.54747351e-13]],

       [[ 2.00000000e+00,  4.22931366e+00],
        [ 4.22931366e+00, -3.66668244e+00]]])), iters=4, f_calls=(5, 211, 211), precision=1.0327220168881081e-07, error=5.474554143347632e-11, converged=True, optimal=True)
```

#### Output format:

The returned `result` is a namedtuple-like object whose elements depend on the type of the method:

- Common:
    - `root`: the solved root.
    - `f_root`: value evaluated at root.
    - `iters`: number of iterations.
    - `f_calls`: number of function calls.
    - `precision`: width of final bracket (for bracketing methods), or absolute difference of root with the last
      estimation, or the span of the set of final estimations.
    - `error`: absolute value of `f_root`.
    - `converged`: `True` if the stopping criterion is met, `False` if the procedure terminated prematurely.
    - `optimal`: `True` only if the error tolerance is satisfied `abs(f_root) <= etol`.
- Exclusive to bracketing methods:
    - `bracket`: final bracket.
    - `f_bracket`: value evaluated at final bracket.
- Exclusive to Newton-like methods:
    - `df_root`: derivative or tuple of derivatives (of increasing orders) evaluated at root.

**Notes**:

- `converged` can be `True` even if the solution is not optimal, which means the routine stopped because the
  precision tolerance is satisfied.
- For `scipy.optimize.root` users, the stopping condition arguments `etol`, `ertol`, `ptol`, `prtol` are equivalent to
  `f_tol`, `f_rtol`, `x_tol`, `x_rtol`, respectively (but not identical).

#### Configurations:

The default values for stop condition arguments (i.e. `etol`, `ertol`, `ptol`, `prtol`, `max_iter`) are globally set to
the values defined in [`_defaults.py`](cyroot/_defaults.py), and can be modified dynamically as follows:

```python
import cyroot

cyroot.set_default_stop_condition_args(
    etol=1e-7,
    ptol=0,  # disable precision tolerance
    max_iter=100)

help(cyroot.illinois)  # run to check the updated docstring
```

For more examples, please check the [`examples`](examples) folder.

## Contributing

If you want to contribute, please contact me. \
If you want an algorithm to be implemented, also drop me the paper (I will read if I have time).

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
