# Native Scalar Callbacks

Install `cy-root`, Cython 3.3.0, setuptools, a C++ compiler, and the development
headers matching your Python interpreter. From this examples directory:

```sh
python -m pip install Cython==3.3.0 setuptools
python native_scalar_setup.py build_ext --inplace
python -c "from native_scalar import solve, solve_newton; print(solve()[0]); print(solve_newton()[0])"
```

`native_scalar.pyx` imports the installed declarations from
`cyroot.native.scalar`. Its residual has the signature
`int (double x, double* out, void* context) noexcept nogil`. It writes the scalar
result into `out[0]` and returns zero for success. Return a nonzero status for a
callback failure; the Python boundary raises `NativeCallbackError` and preserves
that status as `native_status`. A native callback must handle its own failures
and must not let Python or C++ exceptions cross this interface.

`make_scalar_callback` retains its `keepalive` argument. Here that is a Cython
extension object containing the parameter struct, which keeps the context
pointer valid for every solver that retains the descriptor. A `nogil`
declaration permits execution without the GIL; any shared mutable context must
also support the intended concurrency.

The solver acquires the GIL for Python setup and result/history export. Its
full native run uses the same step engine as Python inspection. This example
needs no NumPy array or NumPy C headers. Ordinary Python residuals are accepted
through the separate Python adapter and acquire the GIL for each callback.

`solve_newton()` supplies an analytic derivative through a second native scalar
callback. Both callbacks use the same output/status contract. `Newton` checks
the residual before evaluating the derivative, so an accepted initial root
needs no derivative call. It also skips derivatives after a terminal trial or
when the iteration budget is exhausted. The returned `f_calls` tuple counts
residual and derivative evaluations separately. `df_root` is NaN when the
returned point did not require a derivative evaluation.

`Newton` accepts Python/native mixtures as well. A native callback failure
records its role as `solver.callback_stage` (`"function"` or `"derivative"`),
in addition to `callback_code`. A failed trial residual retains the previous
finite point and residual; a failed derivative retains the newly accepted
finite point and residual. Python callback exceptions preserve the original
exception and traceback.
