"""Pure Python source compiled ahead of time with Cython; see the adjacent manual."""
import cython
from cython.cimports.cyroot.native.scalar import make_scalar_callback
from cython.cimports.cyroot.native.rectangular import make_rectangular_callback


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def residual(x: cython.double) -> cython.double:
    """Typed numerical function: also callable from Python after compilation."""
    return x * x - 2.0


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def derivative(x: cython.double) -> cython.double:
    return 2.0 * x


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _residual_bridge(
    x: cython.double,
    out: cython.pointer(cython.double),
    context: cython.p_void,
) -> cython.int:
    out[0] = residual(x)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _derivative_bridge(
    x: cython.double,
    out: cython.pointer(cython.double),
    context: cython.p_void,
) -> cython.int:
    out[0] = derivative(x)
    return 0


def callback():
    """Return the native residual descriptor, bypassing the Python wrapper."""
    return make_scalar_callback(_residual_bridge, cython.NULL, residual)


def derivative_callback():
    """Return the native analytic derivative descriptor."""
    return make_scalar_callback(_derivative_bridge, cython.NULL, derivative)


def solve(initial=1.5):
    from cyroot.native import Newton

    solver = Newton(callback(), derivative_callback(), initial, history=32)
    result = solver.run()
    return result, solver.history


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _vector_residual(
    x: cython.pointer(cython.const[cython.double]),
    n: cython.size_t,
    m: cython.size_t,
    out: cython.pointer(cython.double),
    context: cython.p_void,
) -> cython.int:
    if n != 2 or m != 3:
        return 1
    out[0] = x[0] * x[0] + x[1] - 3.0
    out[1] = 2.0 * x[0] + x[1] * x[1] - 6.0
    out[2] = x[0] + x[1] - 3.0
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _vector_jacobian(
    x: cython.pointer(cython.const[cython.double]),
    n: cython.size_t,
    m: cython.size_t,
    out: cython.pointer(cython.double),
    context: cython.p_void,
) -> cython.int:
    if n != 2 or m != 3:
        return 1
    # Three rows and two columns; column-major out[row + column*m].
    out[0], out[1], out[2] = 2.0 * x[0], 2.0, 1.0
    out[3], out[4], out[5] = 1.0, 2.0 * x[1], 1.0
    return 0


def vector_callback():
    """Three residuals of two variables, with root (1, 2)."""
    return make_rectangular_callback(
        _vector_residual, 2, 3, cython.NULL, vector_callback)


def vector_jacobian_callback():
    """Native 3-by-2 analytic Jacobian for vector_callback()."""
    return make_rectangular_callback(
        _vector_jacobian, 2, 3, cython.NULL, vector_jacobian_callback,
        jacobian=True)


def solve_vector(initial=(1.4, 1.7), numerical_jacobian=False):
    from cyroot.native import VectorNewton

    jacobian = None if numerical_jacobian else vector_jacobian_callback()
    solver = VectorNewton(vector_callback(), jacobian, initial, history=32,
                          etol=1e-12, ertol=0.0)
    return solver.run(), solver.history
