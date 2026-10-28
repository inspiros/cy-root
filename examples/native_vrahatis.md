# Compiled Vrahatis callback

Copy `native_vrahatis.pyx` and `native_vrahatis_setup.py` into a working directory
with cyroot, Cython 3, setuptools and a C++17 compiler available. Build and run:

```sh
python native_vrahatis_setup.py build_ext --inplace
python -c 'import native_vrahatis as e; answer, history=e.solve(); print(answer)'
```

The root is approximately `(0.7, 0.7)`. The residual writes into caller-owned
native output storage and creates no NumPy array. `callback()` pins its Python
factory as the stateless producer owner; parameterized callbacks must keep their
native context alive. The solver releases the GIL for the sweep loop. Its copied
history is exported with the GIL after computation.

The four starting vertices cover every strict residual sign pattern. In multiple
dimensions, this does not certify a root exists. A small polygon reports
stagnation unless residual tolerance is met. See the
[Vrahatis guide](../docs/native-vrahatis.md) for rectangular callbacks, bounded
single-point discovery, result/history and migration behavior.
