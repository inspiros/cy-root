# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Sequence, Optional

import cython
import numpy as np
cimport numpy as np
import sympy
import sympy.utilities.autowrap
from cython cimport view
from libc cimport math

from ._check_args cimport _check_stop_condition_initial_guess
from ._check_args import _check_stop_condition_args
from ._defaults import ETOL, PTOL, MAX_ITER
from ._return_types import NewtonMethodReturnType
from .fptr cimport (
    double_scalar_func_type, DoubleScalarFPtr, PyDoubleScalarFPtr,
    double_vector_func_type, DoubleVectorFPtr, PyDoubleVectorFPtr,
)
from .utils.dynamic_default_args import dynamic_default_args, named_default
from .utils.function_tagging import tag

__all__ = [
    'newton',
    'halley',
    'householder',
]

################################################################################
# Newton
################################################################################
# noinspection DuplicatedCode
cdef (double, double, double, long, double, double, bint, bint) newton_kernel(
        double_scalar_func_type f,
        double_scalar_func_type df,
        double x0,
        double f_x0,
        double df_x0,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_condition_initial_guess(x0, f_x0, etol, ptol,
                            &precision, &error, &converged, &optimal):
        return x0, f_x0, df_x0, step, precision, error, converged, optimal

    cdef double x1
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0 or df_x0 == 0:
            converged = False
        step += 1
        x1 = x0 - f_x0 / df_x0
        precision = math.fabs(x1 - x0)
        x0, f_x0, df_x0 = x1, f(x1), df(x1)
        error = math.fabs(f_x0)

    optimal = error <= etol
    return x0, f_x0, df_x0, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.newton')
@dynamic_default_args()
@cython.binding(True)
def newton(f: Callable[[float], float],
           df: Callable[[float], float],
           x0: float,
           f_x0: Optional[float] = None,
           df_x0: Optional[float] = None,
           etol: float = named_default(ETOL=ETOL),
           ptol: float = named_default(PTOL=PTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodReturnType:
    """
    Newton method for root-finding.

    Args:
        f: Function for which the root is sought.
        df: Function return derivative of f.
        x0: Initial point.
        f_x0: Value evald at initial point.
        df_x0: First order derivative at initial point.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    df_wrapper = PyDoubleScalarFPtr(df)
    if f_x0 is None:
        f_x0 = f_wrapper(x0)
    if df_x0 is None:
        df_x0 = df_wrapper(x0)

    res = newton_kernel[DoubleScalarFPtr](
        f_wrapper, df_wrapper, x0, f_x0, df_x0, etol, ptol, max_iter)
    return NewtonMethodReturnType.from_results(res, (f_wrapper.n_f_calls, df_wrapper.n_f_calls))

################################################################################
# Halley
################################################################################
# noinspection DuplicatedCode
cdef (double, double, double, double, long, double, double, bint, bint) halley_kernel(
        double_scalar_func_type f,
        double_scalar_func_type df,
        double_scalar_func_type d2f,
        double x0,
        double f_x0,
        double df_x0,
        double d2f_x0,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_condition_initial_guess(x0, f_x0, etol, ptol,
                            &precision, &error, &converged, &optimal):
        return x0, f_x0, df_x0, d2f_x0, step, precision, error, converged, optimal

    cdef double x1, denom
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        denom = 2 * math.fabs(df_x0) ** 2 - f_x0 * d2f_x0
        if denom == 0:
            converged = False
            break
        x1 = x0 - 2 * f_x0 * df_x0 / denom
        precision = math.fabs(x1 - x0)
        x0, f_x0, df_x0, d2f_x0 = x1, f(x1), df(x1), d2f(x1)
        error = math.fabs(f_x0)

    optimal = error <= etol
    return x0, f_x0, df_x0, d2f_x0, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.newton')
@dynamic_default_args()
@cython.binding(True)
def halley(f: Callable[[float], float],
           df: Callable[[float], float],
           d2f: Callable[[float], float],
           x0: float,
           f_x0: Optional[float] = None,
           df_x0: Optional[float] = None,
           d2f_x0: Optional[float] = None,
           etol: float = named_default(ETOL=ETOL),
           ptol: float = named_default(PTOL=PTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodReturnType:
    """
    Halley's method for root-finding.

    Args:
        f: Function for which the root is sought.
        df: Function return derivative of f.
        d2f: Function return second order derivative of f.
        x0: Initial point.
        f_x0: Value evald at initial point.
        df_x0: First order derivative at initial point.
        d2f_x0: Second order derivative at initial point.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    df_wrapper = PyDoubleScalarFPtr(df)
    d2f_wrapper = PyDoubleScalarFPtr(d2f)
    if f_x0 is None:
        f_x0 = f_wrapper(x0)
    if df_x0 is None:
        df_x0 = df_wrapper(x0)
    if d2f_x0 is None:
        d2f_x0 = d2f_wrapper(x0)

    r, f_r, df_r, d2f_r, step, precision, error, converged, optimal = halley_kernel[DoubleScalarFPtr](
        f_wrapper, df_wrapper, d2f_wrapper, x0, f_x0, df_x0, d2f_x0, etol, ptol, max_iter)
    return NewtonMethodReturnType(r, f_r, (df_r, d2f_r), step,
                                  (f_wrapper.n_f_calls, df_wrapper.n_f_calls, d2f_wrapper.n_f_calls),
                                  precision, error, converged, optimal)

################################################################################
# Householder
################################################################################
# noinspection DuplicatedCode
@cython.returns((double, double[:], int, double, double, bint, bint))
cdef householder_kernel(
        DoubleScalarFPtr[:] fs,  # sadly, can't have memory view of C functions
        double_vector_func_type nom_f,
        double_vector_func_type denom_f,
        double x0_,
        double[:] fs_x0,
        unsigned int d,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_condition_initial_guess(x0_, fs_x0[0], etol, ptol,
                            &precision, &error, &converged, &optimal):
        return x0_, fs_x0, step, precision, error, converged, optimal

    cdef double[:] x0 = view.array(shape=(1,),
                                   itemsize=sizeof(double),
                                   format='d')
    cdef double[:] x1 = view.array(shape=(1,),
                                   itemsize=sizeof(double),
                                   format='d')
    x0[0] = x0_  # wrapped in a memory view to be able to pass into mv_func_type
    cdef unsigned int i
    cdef double[:] nom_x0 = nom_f(fs_x0[:-1]), denom_x0 = denom_f(fs_x0)
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0 or denom_x0[0] == 0:
            converged = False
            break
        step += 1
        x1[0] = x0[0] + d * nom_x0[0] / denom_x0[0]
        precision = math.fabs(x1[0] - x0[0])
        # advance
        x0[0] = x1[0]
        # fs_x0 = fs(x0)
        for i in range(d + 1):
            fs_x0[i] = fs[i](x0[0])
        error = math.fabs(fs_x0[0])
        nom_x0, denom_x0 = nom_f(fs_x0[:-1]), denom_f(fs_x0)

    optimal = error <= etol
    return x0[0], fs_x0, step, precision, error, converged, optimal

#########################
# Sympy Expr Evaluators
#########################
cdef class _Expr:
    cdef double eval(self, double[:] args):
        raise NotImplementedError

    def __call__(self, double[:] args) -> double:
        return self.eval(args)

cdef class _AtomicExpr(_Expr):
    pass

cdef class _Number(_AtomicExpr):
    cdef double value
    def __init__(self, number: sympy.core.numbers.Number):
        self.value = float(number.n())

    cdef inline double eval(self, double[:] args):
        return self.value

cdef class _Arg(_AtomicExpr):
    cdef int index
    def __init__(self, indexed: sympy.Indexed):
        self.index = int(indexed.indices[0])

    cdef inline double eval(self, double[:] args):
        return args[self.index]

cdef class _ParameterizedExpr(_Expr):
    cdef readonly _Expr[:] args
    cdef _Expr _arg_i  # used for iterating args
    cdef unsigned int n_args
    def __init__(self, args):
        self.args = np.array([CyExprEvaluator.parse_symbolic_func(_) for _ in args])
        self.n_args = <unsigned int> len(self.args)

cdef class _Negate(_ParameterizedExpr):
    cdef _Expr arg
    def __init__(self, negative_one: sympy.core.numbers.NegativeOne):
        super().__init__(negative_one.args)
        self.arg = self.args[0]

    cdef inline double eval(self, double[:] args):
        return -self.arg.eval(args)

cdef class _Add(_ParameterizedExpr):
    def __init__(self, mul: sympy.core.Add):
        super().__init__(mul.args)

    cdef inline double eval(self, double[:] args):
        cdef double res = 0.0
        cdef unsigned int i
        for i in range(self.n_args):
            self._arg_i = self.args[i]
            res += self._arg_i.eval(args)
            self._arg_i = None
        return res

cdef class _Mul(_ParameterizedExpr):
    def __init__(self, mul: sympy.core.Mul):
        super().__init__(mul.args)

    cdef inline double eval(self, double[:] args):
        cdef double res = 1.0
        cdef unsigned int i
        for i in range(self.n_args):
            self._arg_i = self.args[i]
            res *= self._arg_i.eval(args)
            self._arg_i = None
        return res

cdef class _Pow(_ParameterizedExpr):
    cdef _Expr arg, exp
    def __init__(self, pow: sympy.core.Pow):
        super().__init__(pow.args)
        self.arg = self.args[0]
        self.exp = self.args[1]

    cdef inline double eval(self, double[:] args):
        return self.arg.eval(args) ** self.exp.eval(args)

#########################
# Reciprocal Derivative
#########################
from .fptr import DoubleVectorFPtr, PyDoubleVectorFPtr

# Some implementations up to 10th order
cdef class R0DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = 1/fs[0]
        return res

cdef class R1DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = -fs[1]/fs[0]**2
        return res

cdef class R2DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = (-fs[0]*fs[2] + 2*fs[1]**2)/fs[0]**3
        return res

cdef class R3DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = (-fs[0]**2*fs[3] + 6*fs[0]*fs[1]*fs[2] - 6*fs[1]**3)/fs[0]**4
        return res

cdef class R4DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((8*fs[1]*fs[3] + 6*fs[2]**2)*fs[0]**2 - fs[0]**3*fs[4] -
                  36*fs[0]*fs[1]**2*fs[2] + 24*fs[1]**4)/fs[0]**5
        return res

cdef class R5DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = (-30*(2*fs[1]*fs[3] + 3*fs[2]**2)*fs[0]**2*fs[1] +
                  10*(fs[1]*fs[4] + 2*fs[2]*fs[3])*fs[0]**3 - fs[0]**4*fs[5] +
                  240*fs[0]*fs[1]**3*fs[2] - 120*fs[1]**5)/fs[0]**6
        return res

cdef class R6DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((480*fs[1]*fs[3] + 1080*fs[2]**2)*fs[0]**2*fs[1]**2 +
                  (12*fs[1]*fs[5] + 30*fs[2]*fs[4] + 20*fs[3]**2)*fs[0]**4 +
                  (-90*fs[1]**2*fs[4] - 360*fs[1]*fs[2]*fs[3] - 90*fs[2]**3)*fs[0]**3 -
                  fs[0]**5*fs[6] - 1800*fs[0]*fs[1]**4*fs[2] + 720*fs[1]**6)/fs[0]**7
        return res

cdef class R7DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = (-4200*(fs[1]*fs[3] + 3*fs[2]**2)*fs[0]**2*fs[1]**3 +
                  14*(fs[1]*fs[6] + 3*fs[2]*fs[5] + 5*fs[3]*fs[4])*fs[0]**5 +
                  840*(fs[1]**2*fs[4] + 6*fs[1]*fs[2]*fs[3] + 3*fs[2]**3)*fs[0]**3*fs[1] -
                  42*(3*fs[1]**2*fs[5] + 15*fs[1]*fs[2]*fs[4] + 10*fs[1]*fs[3]**2 + 15*fs[2]**2*fs[3])*fs[0]**4 -
                  fs[0]**6*fs[7] + 15120*fs[0]*fs[1]**5*fs[2] - 5040*fs[1]**7)/fs[0]**8
        return res

cdef class R8DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((40320*fs[1]*fs[3] + 151200*fs[2]**2)*fs[0]**2*fs[1]**4 + (-8400*fs[1]**2*fs[4] -
                  67200*fs[1]*fs[2]*fs[3] - 50400*fs[2]**3)*fs[0]**3*fs[1]**2 + (16*fs[1]*fs[7] + 56*fs[2]*fs[6] +
                  112*fs[3]*fs[5] + 70*fs[4]**2)*fs[0]**6 + (-168*fs[1]**2*fs[6] - 1008*fs[1]*fs[2]*fs[5] -
                  1680*fs[1]*fs[3]*fs[4] - 1260*fs[2]**2*fs[4] - 1680*fs[2]*fs[3]**2)*fs[0]**5 +
                  (1344*fs[1]**3*fs[5] + 10080*fs[1]**2*fs[2]*fs[4] + 6720*fs[1]**2*fs[3]**2 +
                   20160*fs[1]*fs[2]**2*fs[3] + 2520*fs[2]**4)*fs[0]**4 - fs[0]**7*fs[8] -
                  141120*fs[0]*fs[1]**6*fs[2] + 40320*fs[1]**8)/fs[0]**9
        return res

cdef class R9DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((-423360*fs[1]*fs[3] - 1905120*fs[2]**2)*fs[0]**2*fs[1]**5 + (90720*fs[1]**2*fs[4] +
                  907200*fs[1]*fs[2]*fs[3] + 907200*fs[2]**3)*fs[0]**3*fs[1]**3 + (18*fs[1]*fs[8] + 72*fs[2]*fs[7] +
                  168*fs[3]*fs[6] + 252*fs[4]*fs[5])*fs[0]**7 + 2520*(-6*fs[1]**3*fs[5] - 60*fs[1]**2*fs[2]*fs[4] -
                  40*fs[1]**2*fs[3]**2 - 180*fs[1]*fs[2]**2*fs[3] - 45*fs[2]**4)*fs[0]**4*fs[1] + (2016*fs[1]**3*fs[6] +
                  18144*fs[1]**2*fs[2]*fs[5] + 30240*fs[1]**2*fs[3]*fs[4] + 45360*fs[1]*fs[2]**2*fs[4] +
                  60480*fs[1]*fs[2]*fs[3]**2 + 30240*fs[2]**3*fs[3])*fs[0]**5 + (-216*fs[1]**2*fs[7] -
                  1512*fs[1]*fs[2]*fs[6] - 3024*fs[1]*fs[3]*fs[5] - 1890*fs[1]*fs[4]**2 - 2268*fs[2]**2*fs[5] -
                  7560*fs[2]*fs[3]*fs[4] - 1680*fs[3]**3)*fs[0]**6 - fs[0]**8*fs[9] + 1451520*fs[0]*fs[1]**7*fs[2] -
                  362880*fs[1]**9)/fs[0]**10
        return res

cdef class R10DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((4838400*fs[1]*fs[3] + 25401600*fs[2]**2)*fs[0]**2*fs[1]**6 + (-1058400*fs[1]**2*fs[4] -
                  12700800*fs[1]*fs[2]*fs[3] - 15876000*fs[2]**3)*fs[0]**3*fs[1]**4 + (20*fs[1]*fs[9] +
                  90*fs[2]*fs[8] + 240*fs[3]*fs[7] + 420*fs[4]*fs[6] + 252*fs[5]**2)*fs[0]**8 + (181440*fs[1]**3*fs[5] +
                  2268000*fs[1]**2*fs[2]*fs[4] + 1512000*fs[1]**2*fs[3]**2 + 9072000*fs[1]*fs[2]**2*fs[3] +
                  3402000*fs[2]**4)*fs[0]**4*fs[1]**2 + (-25200*fs[1]**4*fs[6] - 302400*fs[1]**3*fs[2]*fs[5] -
                  504000*fs[1]**3*fs[3]*fs[4] - 1134000*fs[1]**2*fs[2]**2*fs[4] - 1512000*fs[1]**2*fs[2]*fs[3]**2 -
                  1512000*fs[1]*fs[2]**3*fs[3] - 113400*fs[2]**5)*fs[0]**5 + (-270*fs[1]**2*fs[8] -
                  2160*fs[1]*fs[2]*fs[7] - 5040*fs[1]*fs[3]*fs[6] - 7560*fs[1]*fs[4]*fs[5] - 3780*fs[2]**2*fs[6] -
                  15120*fs[2]*fs[3]*fs[5] - 9450*fs[2]*fs[4]**2 - 12600*fs[3]**2*fs[4])*fs[0]**7 +
                  (2880*fs[1]**3*fs[7] + 30240*fs[1]**2*fs[2]*fs[6] + 60480*fs[1]**2*fs[3]*fs[5] +
                  37800*fs[1]**2*fs[4]**2 + 90720*fs[1]*fs[2]**2*fs[5] + 302400*fs[1]*fs[2]*fs[3]*fs[4] +
                  67200*fs[1]*fs[3]**3 + 75600*fs[2]**3*fs[4] + 151200*fs[2]**2*fs[3]**2)*fs[0]**6 - fs[0]**9*fs[10] -
                  16329600*fs[0]*fs[1]**8*fs[2] + 3628800*fs[1]**10)/fs[0]**11
        return res

# For functions of higher order derivatives, use this class to eval expression
# Warning: Very slow, but somehow still slightly faster than Sympy's wrapped function
# (if someone goes this far, they must be insane)
cdef class CyExprEvaluator(DoubleVectorFPtr):
    cdef _Expr cy_expr
    type_map = {
        sympy.core.Number: _Number,
        sympy.Indexed: _Arg,
        sympy.core.numbers.NegativeOne: _Negate,
        sympy.core.Add: _Add,
        sympy.core.Mul: _Mul,
        sympy.core.Pow: _Pow,
    }
    def __init__(self, expr: sympy.core.Expr):
        self.cy_expr = self.parse_symbolic_func(expr)

    @staticmethod
    def parse_symbolic_func(expr):
        if isinstance(expr, sympy.Expr):
            if isinstance(expr, sympy.core.Number):
                return _Number(expr)
            evaluator_cls = CyExprEvaluator.type_map.get(type(expr))
            if evaluator_cls is not None:
                return evaluator_cls(expr)
            else:
                raise NotImplementedError(f'No implementation found for {type(expr)}.')

    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = self.cy_expr.eval(fs)
        return res

# noinspection DuplicatedCode
class ReciprocalDerivativeFuncFactory:
    """
    References:
        https://math.stackexchange.com/questions/5357/whats-the-generalisation-of-the-quotient-rule-for-higher-derivatives
    """
    # dictionary of known functions
    rd_c_funcs: dict[int, DoubleVectorFPtr] = {
        0: R0DFPtr(),
        1: R1DFPtr(),
        2: R2DFPtr(),
        3: R3DFPtr(),
        4: R4DFPtr(),
        5: R5DFPtr(),
        6: R6DFPtr(),
        7: R7DFPtr(),
        8: R8DFPtr(),
        9: R9DFPtr(),
        10: R10DFPtr(),
    }
    rd_py_funcs: dict[int, DoubleVectorFPtr] = dict(
        enumerate(map(lambda o: PyDoubleVectorFPtr(o.__call__), rd_c_funcs.values())))
    def __init__(self):
        raise RuntimeError('Do not initialize this class.')

    @classmethod
    def get(cls, d: int, max_d: int = None, c_code=False):
        if (not c_code and d not in cls.rd_py_funcs.keys()) or (c_code and d not in cls.rd_c_funcs):
            if max_d is None:
                max_d = d
            sym_x = sympy.Dummy('x')
            sym_f = sympy.Function(sympy.Dummy('f'))
            sym_fs = sympy.IndexedBase(sympy.Dummy('fs'), shape=(max_d + 1,))

            expr = 1 / sym_f(sym_x)
            sym_rd_f = expr.diff(sym_x, d).simplify()
            for i in range(d, -1, -1):
                sym_rd_f = sym_rd_f.subs(sympy.Derivative(sym_f(sym_x), (sym_x, i)), sym_fs[i])

            if not c_code:
                rd_f = sympy.lambdify(sym_fs, sympy.Array([sym_rd_f]), modules='numpy')
                cls.rd_py_funcs[d] = PyDoubleVectorFPtr(rd_f)
            else:
                rd_f = <DoubleVectorFPtr> CyExprEvaluator(sym_rd_f)
                cls.rd_c_funcs[d] = rd_f
                # sympy autowrap does not support array
                # sym_fs_mat = sympy.MatrixSymbol('fs', d + 1, 1)
                # for i in range(d, -1, -1):
                #     sym_rd_f = sym_rd_f.subs(sym_fs[i], sym_fs_mat[i])
                # rd_f = sympy.utilities.autowrap.autowrap(sym_rd_f, backend='cython')
                # cls.rd_c_funcs[d] = PyDoubleMemoryViewFPtr(
                #     lambda fs: np.reshape(rd_f(np.reshape(fs, (-1, 1))), (-1,)))
        return cls.rd_py_funcs[d] if not c_code else cls.rd_c_funcs[d]

# noinspection DuplicatedCode
@tag('cyroot.newton')
@dynamic_default_args()
@cython.binding(True)
def householder(f: Callable[[float], float],
                dfs: Sequence[Callable[[float], float]],
                x0: float,
                f_x0: Optional[float] = None,
                dfs_x0: Optional[Sequence[float]] = None,
                etol: float = named_default(ETOL=ETOL),
                ptol: float = named_default(PTOL=PTOL),
                max_iter: int = named_default(MAX_ITER=MAX_ITER),
                c_code: bool = True) -> NewtonMethodReturnType:
    """
    Householder's method for root-finding.

    Args:
        f: Function for which the root is sought.
        dfs: Sequence of derivative functions of f in increasing
         order.
        x0: Initial guess.
        f_x0: Value evaluated at initial guess.
        dfs_x0: Sequence of derivatives in increasing order at
         initial guess.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.
        c_code: Use C implementation of reciprocal derivative
         function or not. Defaults to True.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if len(dfs) < 2:
        raise ValueError(f'Requires at least second order derivative. Got {len(dfs)}.')

    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    fs_wrappers = np.asarray([PyDoubleScalarFPtr(f)] + [PyDoubleScalarFPtr(df) for df in dfs])
    if f_x0 is None:
        f_x0 = fs_wrappers[0](x0)
    if dfs_x0 is None:
        dfs_x0 = [f_wrapper(x0) for f_wrapper in fs_wrappers[1:]]
    fs_x0 = np.asarray([f_x0] + dfs_x0)

    d = len(dfs)
    r, fs_r, step, precision, error, converged, optimal = householder_kernel[DoubleVectorFPtr](
        fs_wrappers,
        <DoubleVectorFPtr>ReciprocalDerivativeFuncFactory.get(d - 1, c_code=c_code),
        <DoubleVectorFPtr>ReciprocalDerivativeFuncFactory.get(d, c_code=c_code),
        x0, fs_x0, d, etol, ptol, max_iter)
    return NewtonMethodReturnType(r, float(fs_r[0]), tuple(fs_r[1:]), step,
                                  tuple(_.n_f_calls for _ in fs_wrappers),
                                  precision, error, converged, optimal)
