# Native secant callback

Install `cy-root`, Cython 3.3.0, setuptools, a C++17 compiler, and matching Python
development headers. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_secant_setup.py build_ext --inplace
python -c "from native_secant import solve; result, history = solve(); print(result)"
```

The example solves \(x^2=2\) from two starting points without supplying a
derivative. `make_scalar_callback` retains the `Quadratic` context owner, so
its native parameter pointer remains valid throughout the solve. The callback
writes `out[0]` and returns zero for success; a nonzero native code raises
`NativeCallbackError` at the Python boundary. Native callbacks must not throw
Python or C++ exceptions through this interface.

`Secant.step()` and `.history` expose copied immutable intermediate states.
With this native residual the numerical run releases the GIL; ordinary Python
callables use the same engine and acquire the GIL per callback. A secant start
need not bracket a root, so convergence depends on the function and starting
pair. The example needs no NumPy arrays or NumPy C headers, while the package
still depends on NumPy and SciPy.
