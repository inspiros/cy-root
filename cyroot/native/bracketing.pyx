# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Native bracketing state machine with shared scalar callbacks and copied history."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from typing import NamedTuple
from libc.math cimport fabs, fmax, fmin, isfinite, copysign, sqrt, pow, ceil, log2, ldexp, nextafter, NAN, INFINITY
from libc.string cimport memset

from .scalar import NativeCallbackError, _nonnegative_integer
from .scalar cimport ScalarRecord
from cyroot._check_args import _check_stop_cond_args
from cyroot._stopping cimport (
    init_stop, check_stop, RUNNING, RESIDUAL_TOLERANCE,
    MAX_ITERATIONS, BREAKDOWN, NONFINITE, CALLBACK_FAILURE, STAGNATION, residual_met,
)
from cyroot.return_types cimport BracketingMethodsReturnType
from cyroot.return_types import _normalize_reason


cdef inline double midpoint(double a, double b) noexcept nogil:
    return a/2+b/2

cdef inline double safeguard(double x, double a, double b) noexcept nogil:
    return x if isfinite(x) and fmin(a,b)<x<fmax(a,b) else midpoint(a,b)

cdef double secant(double a,double b,double fa,double fb) noexcept nogil:
    cdef double scale=fmax(fabs(fa),fabs(fb)), t
    if scale == 0 or fa == fb:
        return midpoint(a,b)
    fa,fb=fa/scale,fb/scale
    t=-fa/(fb-fa)
    return a*(1-t)+b*t

cdef double inverse3(double a,double b,double c,double fa,double fb,double fc) noexcept nogil:
    cdef double scale=fmax(fabs(fa),fmax(fabs(fb),fabs(fc)))
    fa,fb,fc=fa/scale,fb/scale,fc/scale
    if fa==fb or fa==fc or fb==fc:
        return NAN
    return a*fb/(fa-fb)*fc/(fa-fc)+b*fa/(fb-fa)*fc/(fb-fc)+c*fa/(fc-fa)*fb/(fc-fb)

cdef double hyperbolic3(double a,double b,double c,double fa,double fb,double fc) noexcept nogil:
    # Rational interpolation through three points, scaled in both axes. This
    # is the hyperbolic Brent step; products of raw large residuals overflow.
    cdef double scale=fmax(fabs(fa),fmax(fabs(fb),fabs(fc)))
    cdef double width=fmax(fabs(a-b),fabs(b-c)), u, v, den
    if scale==0 or width==0 or not isfinite(width):
        return NAN
    fa,fb,fc=fa/scale,fb/scale,fc/scale
    u,v=(a-b)/width,(b-c)/width
    den=fa*(fb-fc)*u-fc*(fa-fb)*v
    if den==0:
        return NAN
    return b-(fb*(fa-fc)*v/den)*(a-b)

cdef double inverse4(double a,double b,double c,double d,double fa,double fb,double fc,double fd) noexcept nogil:
    cdef double scale=fmax(fmax(fabs(fa),fabs(fb)),fmax(fabs(fc),fabs(fd)))
    if not isfinite(scale) or scale==0:
        return NAN
    fa,fb,fc,fd=fa/scale,fb/scale,fc/scale,fd/scale
    if fa==fb or fa==fc or fa==fd or fb==fc or fb==fd or fc==fd:
        return NAN
    return -(a*fb/(fa-fb)*fc/(fa-fc)*fd/(fa-fd)+
            b*fa/(fb-fa)*fc/(fb-fc)*fd/(fb-fd)+
            c*fa/(fc-fa)*fb/(fc-fb)*fd/(fc-fd)+
            d*fa/(fd-fa)*fb/(fd-fb)*fc/(fd-fc))

cdef double quadratic(double a,double b,double d,double fa,double fb,double fd,size_t k) noexcept nogil:
    cdef double B,A,r,P,den,x
    cdef size_t i
    if a==b or a==d or b==d:
        return midpoint(a,b)
    B=(fb-fa)/(b-a)
    A=((fd-fb)/(d-b)-B)/(d-a)
    if A==0:
        return secant(a,b,fa,fb)
    r=a if copysign(1.,A)==copysign(1.,fa) else b
    for i in range(k):
        P=(A*(r-b)+B)*(r-a)+fa
        den=B+A*((r-a)+(r-b))
        if den==0:
            return midpoint(a,b)
        x=r-P/den
        if not isfinite(x) or not a<x<b:
            return r if a<r<b else midpoint(a,b)
        r=x
    return r


cdef double wu_quadratic(double p,double q,double r,double fp,double fq,double fr) noexcept nogil:
    # Solve in normalized coordinate/residual units. Raw slopes scale as 1/x
    # and their discriminant can silently underflow even for a finite model.
    cdef double h=fmax(fabs(q-p),fmax(fabs(r-p),fabs(r-q)))
    cdef double fs=fmax(fabs(fp),fmax(fabs(fq),fabs(fr)))
    cdef double qp,rp,rq,u,v,w,t,delta,den,cs
    if p==q or p==r or q==r or fs==0:
        return NAN
    if isfinite(h):
        qp,rp,rq=(q-p)/h,(r-p)/h,(r-q)/h
    else:
        h=fmax(fabs(p),fmax(fabs(q),fabs(r)))
        qp,rp,rq=q/h-p/h,r/h-p/h,r/h-q/h
    if qp==0 or rp==0 or rq==0:
        return NAN
    fp,fq,fr=fp/fs,fq/fs,fr/fs
    u,v,w=(fq-fp)/qp,(fr-fp)/rp,(fr-fq)/rq
    t=w+v-u
    delta=(w-u)/rp
    cs=fmax(fabs(t),fmax(fabs(delta),fabs(fr)))
    if not isfinite(cs) or cs==0:
        return NAN
    t,delta,fr=t/cs,delta/cs,fr/cs
    den=t*t-4*delta*fr
    if den<0 or not isfinite(den):
        return NAN
    den=t+copysign(sqrt(den),t)
    if den==0:
        return NAN
    return r-(2*fr/den)*h


class BracketStep(NamedTuple):
    root: float
    f_root: float
    bracket: tuple
    f_bracket: tuple
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: object
    callback_code: int


_METHODS = ('regula_falsi', 'illinois', 'pegasus', 'anderson_bjorck',
            'dekker', 'brent', 'chandrupatla', 'ridders', 'toms748', 'wu', 'itp')


cdef class BracketSolver:
    """Resumable safeguarded scalar methods sharing the native callback ABI.

    Each step performs one algorithm iteration: Ridders may sample twice and
    TOMS748 may sample k+2 times. History contains copied iteration states;
    neither callback samples nor unbounded workspaces are retained.
    """
    def __init__(self, function, a, b, f_a=None, f_b=None, *, method='brent',
                 etol=ETOL, ertol=ERTOL,
                 ptol=PTOL, prtol=PRTOL,
                 max_iter=MAX_ITER, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024, interp_method=1, sigma=1e-12,
                 k=2, mu=.5, k1=None, k2=2., n0=1):
        cdef double first, second, value0, value1, tol, nmax
        cdef double f_atol, f_rtol, x_atol, x_rtol
        cdef bint cached_a, cached_b
        if self._constructing or self._callback is not None or self._initialized:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            self._callback = function if isinstance(function, ScalarCallback) else ScalarCallback(function)
            self._enter()
            try:
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(
                    etol, ertol, ptol, prtol, max_iter)
                limit = _nonnegative_integer(history, 'history')
                every = _nonnegative_integer(history_every, 'history_every')
                byte_limit = _nonnegative_integer(history_bytes, 'history_bytes')
                if every == 0:
                    raise ValueError('history_every must be positive.')
                if limit > byte_limit // sizeof(ScalarRecord):
                    raise ValueError('Requested history exceeds history_bytes.')
                self._method = _METHODS.index(method)
                if isinstance(interp_method, str):
                    interp_method = {'quadratic': 1, 'hyperbolic': 2}.get(interp_method, 0)
                self._interp = _nonnegative_integer(interp_method, 'interp_method')
                if self._interp not in (1, 2):
                    raise ValueError('interp_method must be 1 or 2.')
                self._sigma = sigma
                self._k = _nonnegative_integer(k, 'k')
                self._mu = mu
                self._k2 = k2
                self._n0 = _nonnegative_integer(n0, 'n0')
                if not isfinite(self._sigma) or self._sigma < 0:
                    raise ValueError('sigma must be finite and nonnegative.')
                if self._k == 0 or self._k > 100:
                    raise ValueError('k must be between 1 and 100.')
                if not isfinite(self._mu) or not 0 < self._mu < 1:
                    raise ValueError('mu must be finite and in (0, 1).')
                if not isfinite(self._k2) or not 1 <= self._k2 < 2.618033988749895:
                    raise ValueError('k2 must be in [1, 1 + phi).')
                first, second = a, b
                cached_a, cached_b = f_a is not None, f_b is not None
                value0 = NAN if f_a is None else f_a
                value1 = NAN if f_b is None else f_b
                if second < first:
                    first, second, value0, value1 = second, first, value1, value0
                    cached_a, cached_b = cached_b, cached_a
                self._default_k1 = k1 is None
                self._initial_width = fabs(second-first)
                self._k1 = .2 if k1 is None else k1
                if not isfinite(self._k1) or self._k1 <= 0:
                    raise ValueError('k1 must be finite and positive.')
                f_atol, f_rtol, x_atol, x_rtol = etol, ertol, ptol, prtol
                self._max_iter = max_iter
                self._history_every = every
                self._records.resize(limit)
                memset(&self._state, 0, sizeof(ScalarRecord))
                self._state.a, self._state.b = first, second
                self._state.f_a, self._state.f_b = value0, value1
                self._state.root, self._state.f_root = first, value0
                self._state.precision = fabs(second-first)
                self._state.error = fabs(value0)
                self._e = self._fe = NAN
                self._t = .5
                init_stop(&self._stop, 0., fmax(fabs(first), fabs(second)), f_atol, f_rtol, x_atol, x_rtol)
                with nogil:
                    self._initialize(cached_a, cached_b)
                    self._record()
                # Projection uses a fixed absolute tolerance; exact stopping
                # still uses the caller's tolerances and current bracket scale.
                if self._method == 10 and self._stop.reason == RUNNING:
                    tol = fmax(x_atol, x_rtol * self._stop.x_scale)
                    if tol <= 0:
                        tol = fmax(self._state.precision * 2.220446049250313e-16, 4.9406564584124654e-324)
                    nmax = fmax(0., ceil(log2(self._state.precision)-log2(tol))) + self._n0
                    # Keep the exponent separate: a large initial projection
                    # radius may overflow, but later radii can be finite.
                    self._itp_tolerance,self._itp_exponent = tol,nmax
                self._initialized = True
                self._raise_callback_error()
            finally:
                self._leave()
        finally:
            self._constructing = False

    cdef void _ready(self) except *:
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')

    cdef void _enter(self) except *:
        if self._callback is None or self._callback._function == NULL:
            raise RuntimeError('Solver requires an initialized callback adapter.')
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if self._callback._active:
            raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        self._callback._active = True
        self._callback._error = None

    cdef void _leave(self) noexcept:
        self._callback._active = False
        self._busy = False

    cdef int _evaluate(self, double x, double* out) noexcept nogil:
        cdef int code
        self._state.f_calls += 1
        out[0] = NAN
        code = self._callback._function(x, out, self._callback._context)
        if code != 0:
            self._state.callback_code = code
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef void _initialize(self, bint cached_a, bint cached_b) noexcept nogil:
        cdef double fa = self._state.f_a, fb = self._state.f_b
        if not isfinite(self._state.a) or not isfinite(self._state.b):
            self._stop.reason = NONFINITE
            return
        if not cached_a and self._evaluate(self._state.a, &fa) != 0:
            return
        if not isfinite(fa):
            self._stop.reason = NONFINITE
            return
        self._state.f_a = self._state.f_root = fa
        self._state.error = fabs(fa)
        self._stop.f_scale = fabs(fa)
        if residual_met(&self._stop, fabs(fa)):
            self._stop.reason = RESIDUAL_TOLERANCE
            return
        if not cached_b and self._evaluate(self._state.b, &fb) != 0:
            return
        if not isfinite(fb):
            self._stop.reason = NONFINITE
            return
        self._state.f_b = fb
        self._stop.f_scale = fmax(fabs(fa), fabs(fb))
        self._best()
        if not residual_met(&self._stop, self._state.error) and copysign(1., fa) == copysign(1., fb):
            self._stop.reason = BREAKDOWN
            return
        if self._method in (4, 5) and fabs(fa) < fabs(fb):
            self._swap()
        self._p, self._fp = self._state.a, self._state.f_a
        self._q = self._p
        self._d, self._fd = self._p, self._fp
        self._weight = self._state.f_a
        self._check()

    cdef void _swap(self) noexcept nogil:
        self._state.a, self._state.b = self._state.b, self._state.a
        self._state.f_a, self._state.f_b = self._state.f_b, self._state.f_a

    cdef void _best(self) noexcept nogil:
        if fabs(self._state.f_a) <= fabs(self._state.f_b):
            self._state.root, self._state.f_root = self._state.a, self._state.f_a
        else:
            self._state.root, self._state.f_root = self._state.b, self._state.f_b
        self._state.error = fabs(self._state.f_root)
        self._state.precision = fabs(self._state.b-self._state.a)
        self._stop.x_scale = fmax(fabs(self._state.a), fabs(self._state.b))

    cdef void _check(self) noexcept nogil:
        if check_stop(&self._stop, self._state.error, self._state.precision,
                      self._state.iters > 0, True):
            return
        if self._max_iter != 0 and self._state.iters >= self._max_iter:
            self._stop.reason = MAX_ITERATIONS
        elif midpoint(self._state.a, self._state.b) in (self._state.a, self._state.b):
            self._stop.reason = STAGNATION

    cdef bint _trial(self, double x) noexcept nogil:
        cdef double value
        x = safeguard(x, self._state.a, self._state.b)
        if x == self._state.a or x == self._state.b:
            self._stop.reason = STAGNATION
            return False
        if self._evaluate(x, &value) != 0:
            return False
        if not isfinite(value):
            self._stop.reason = NONFINITE
            return False
        self._last_x, self._last_f = x, value
        self._e, self._fe = self._d, self._fd
        if copysign(1., self._state.f_a) == copysign(1., value):
            self._d, self._fd = self._state.a, self._state.f_a
            self._state.a, self._state.f_a = x, value
        else:
            self._d, self._fd = self._state.b, self._state.f_b
            self._state.b, self._state.f_b = x, value
        self._best()
        # Keep the evaluated candidate as the result on residual convergence.
        if residual_met(&self._stop, fabs(value)):
            self._state.root, self._state.f_root, self._state.error = x, value, fabs(value)
            self._stop.reason = RESIDUAL_TOLERANCE
        else:
            # A multi-sample iteration must observe bracket tolerance between
            # samples, but only the outer _step enforces the iteration budget.
            check_stop(&self._stop, self._state.error, self._state.precision, True, True)
        return self._stop.reason == RUNNING

    cdef void _record(self) noexcept nogil:
        self._state.reason = <int>self._stop.reason
        if self._records.size() == 0:
            return
        if self._record_count > 0 and self._records[self._record_count - 1].iters == self._state.iters:
            self._records[self._record_count - 1] = self._state
            return
        if self._state.iters % self._history_every != 0 and self._stop.reason == RUNNING:
            return
        if self._record_count >= self._records.size():
            self._history_truncated = True
            return
        self._records[self._record_count] = self._state
        self._record_count += 1

    cdef void _step(self) noexcept nogil:
        cdef double a=self._state.a, b=self._state.b
        cdef double fa=self._state.f_a, fb=self._state.f_b
        cdef double c=midpoint(a,b), x, y, fm, scale, u, v, w, den, t, delta, rho
        cdef double old_width=fabs(b-a), old_p=self._p, old_fp=self._fp, old_q=self._q
        cdef size_t j
        cdef bint interpolated=False
        if self._stop.reason != RUNNING:
            return
        self._state.iters += 1
        if self._method <= 3:
            x = secant(a,b,self._weight,fb)
            if self._trial(x):
                if copysign(1.,fb) != copysign(1.,self._last_f):
                    self._state.a, self._state.f_a = b, fb
                    self._weight = fb
                else:
                    self._state.a, self._state.f_a = a, fa
                    if self._method == 1:
                        self._weight *= .5
                    elif self._method == 2:
                        scale=fmax(fabs(fb),fabs(self._last_f))
                        self._weight *= (fb/scale)/(fb/scale+self._last_f/scale)
                    elif self._method == 3:
                        t=1-self._last_f/fb
                        self._weight *= t if t > 0 else .5
                self._state.b, self._state.f_b = self._last_x,self._last_f
                self._best()
        elif self._method in (4,5):
            x = secant(old_p,b,old_fp,fb)
            if self._method == 4:
                if not fmin(b,c) < x < fmax(b,c):
                    x=c
            else:
                if self._interp == 1 and fa != fb and fa != old_fp and fb != old_fp:
                    x=inverse3(a,b,old_p,fa,fb,old_fp)
                elif self._interp == 2 and a != b and b != old_p and a != old_p:
                    x=hyperbolic3(a,b,old_p,fa,fb,old_fp)
                u=.75*a+.25*b
                if (not isfinite(x) or not fmin(b,u)<x<fmax(b,u) or
                    (self._last_method == 0 and fabs(x-b)>=fabs(b-old_p)/2) or
                    (self._last_method == 1 and fabs(x-b)>=fabs(old_p-old_q)/2) or
                    (self._last_method == 0 and self._sigma>fabs(b-old_p)) or
                    (self._last_method == 1 and self._sigma>fabs(old_p-old_q))):
                    x=c
                    self._last_method=0
                else:
                    self._last_method=1
            self._q,self._p,self._fp=old_p,b,fb
            if self._trial(x) and fabs(self._state.f_a)<fabs(self._state.f_b):
                self._swap()
        elif self._method == 6:
            x=a*(1-self._t)+b*self._t
            if self._trial(x):
                # Chandrupatla keeps newest sample at a; d is the discarded end.
                if self._state.b == self._last_x:
                    self._swap()
                a,b,fa,fb=self._state.a,self._state.b,self._state.f_a,self._state.f_b
                self._t=.5
                u=(a-b)/(self._d-b)
                v=(fa-fb)/(self._fd-fb) if self._fd != fb else NAN
                # In the published admissibility test xi is in (0,1).
                if 0<u<1 and 1-sqrt(1-u)<v<sqrt(u):
                    x=inverse3(a,b,self._d,fa,fb,self._fd)
                    self._t=(x-a)/(b-a)
                t=fmin(.5,(self._stop.ptol+self._stop.prtol*self._stop.x_scale+self._sigma/2)/fabs(b-a))
                self._t=fmax(t,fmin(self._t,1-t))
        elif self._method == 7:
            if self._trial(c):
                fm=self._last_f
                scale=fmax(fabs(fm),fmax(fabs(fa),fabs(fb)))
                u,v,w=fa/scale,fb/scale,fm/scale
                den=sqrt(w*w-u*v)
                if den == 0 or not isfinite(den):
                    self._stop.reason=BREAKDOWN
                else:
                    x=c+(c-a)*copysign(1.,fa)*w/den
                    if x != c:
                        self._trial(x)
        elif self._method == 8:
            if self._state.iters == 1:
                self._trial(secant(a,b,fa,fb))
            else:
                for j in range(2,self._k+2):
                    a,b,fa,fb=self._state.a,self._state.b,self._state.f_a,self._state.f_b
                    x=inverse4(a,b,self._d,self._e,fa,fb,self._fd,self._fe)
                    if not isfinite(x) or not a<x<b:
                        x=quadratic(a,b,self._d,fa,fb,self._fd,j)
                    if not self._trial(x):
                        break
                if self._stop.reason == RUNNING:
                    a,b,fa,fb=self._state.a,self._state.b,self._state.f_a,self._state.f_b
                    u,v=(a,fa) if fabs(fa)<fabs(fb) else (b,fb)
                    x=u+2*(secant(a,b,fa,fb)-u)
                    if fabs(x-u)>fabs(b-a)/2:
                        x=midpoint(a,b)
                    if self._trial(x) and self._state.precision>self._mu*old_width:
                        self._trial(midpoint(self._state.a,self._state.b))
        elif self._method == 9:
            x=c
            if self._state.iters>1 and self._p != self._q and self._q != self._r and self._r != self._p:
                x=wu_quadratic(self._p,self._q,self._r,self._fp,self._fq,self._fr)
            self._p,self._q,self._fp,self._fq=a,b,fa,fb
            if self._trial(x):
                self._r,self._fr=self._last_x,self._last_f
        else:
            # ptol measures full width; paper epsilon=tol/2. With iters=1
            # on the first step, this is epsilon*2**(nmax-j), j=0.
            # Clamping only the exponent avoids unsafe narrowing of huge n0;
            # +/-4096 saturates every positive finite double base safely.
            self._rho=ldexp(self._itp_tolerance,<int>fmax(-4096.,fmin(
                self._itp_exponent-<double>self._state.iters,4096.)))
            rho=fmax(0.,self._rho-old_width/2)
            delta=(.2*pow(old_width,self._k2-1)*(old_width/self._initial_width)
                   if self._default_k1 else self._k1*pow(old_width,self._k2))
            x=secant(a,b,fa,fb)
            t=copysign(1.,c-x)
            x=x+t*delta if delta<=fabs(c-x) else c
            if fabs(x-c)>rho:
                x=c-t*rho
            # Inward rounding of the equivalent projection bounds prevents a
            # subtraction from putting x just outside its permitted interval.
            # An exactly collapsed interval keeps its single midpoint.
            u,v=b-self._rho,a+self._rho
            if u<v:
                u,v=nextafter(u,b),nextafter(v,a)
                if u<=v:
                    x=fmax(u,fmin(x,v))
            self._trial(x)
        self._check()
        self._record()

    cdef void _raise_callback_error(self) except *:
        if self._callback._error is not None:
            error = self._callback._error
            self._callback._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code, self._copy_result(), 'function')

    cdef object _snapshot(self, ScalarRecord state):
        reason = None if state.reason == RUNNING else _normalize_reason(state.reason)
        bracket=(state.a,state.b) if state.a<=state.b else (state.b,state.a)
        values=(state.f_a,state.f_b) if state.a<=state.b else (state.f_b,state.f_a)
        return BracketStep(state.root,state.f_root,bracket,values,state.iters,
                           state.f_calls,state.precision,state.error,reason,state.callback_code)

    cdef object _copy_result(self):
        cdef ScalarRecord state = self._state
        bracket=(state.a,state.b) if state.a<=state.b else (state.b,state.a)
        values=(state.f_a,state.f_b) if state.a<=state.b else (state.f_b,state.f_a)
        return BracketingMethodsReturnType(state.root,state.f_root,state.iters,state.f_calls,
                    bracket,values,state.precision,state.error,False,False,
                    termination_reason=None if state.reason == RUNNING else state.reason)

    def step(self):
        """Perform one algorithm iteration and return an immutable copied state."""
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    self._step()
                self._raise_callback_error()
            return self._snapshot(self._state)
        finally:
            self._leave()

    def run(self):
        """Use the remaining total iteration budget through the same step engine."""
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    while self._stop.reason == RUNNING:
                        self._step()
                self._raise_callback_error()
            return self._copy_result()
        finally:
            self._leave()

    @property
    def result(self):
        self._ready()
        self._enter()
        try:
            return self._copy_result()
        finally:
            self._leave()

    @property
    def history(self):
        self._ready()
        self._enter()
        try:
            return tuple([self._snapshot(self._records[i]) for i in range(self._record_count)])
        finally:
            self._leave()

    @property
    def done(self):
        self._ready()
        return self._stop.reason != RUNNING

    @property
    def history_truncated(self):
        self._ready()
        return bool(self._history_truncated)

    @property
    def callback_code(self):
        self._ready()
        return self._state.callback_code

    @property
    def callback_stage(self):
        self._ready()
        return 'function' if self._stop.reason == CALLBACK_FAILURE else None

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        return self.step()


def regula_falsi(function, a, b, **options):
    """Run regula falsi with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="regula_falsi", **options).run()

def illinois(function, a, b, **options):
    """Run illinois with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="illinois", **options).run()

def pegasus(function, a, b, **options):
    """Run pegasus with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="pegasus", **options).run()

def anderson_bjorck(function, a, b, **options):
    """Run anderson bjorck with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="anderson_bjorck", **options).run()

def dekker(function, a, b, **options):
    """Run dekker with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="dekker", **options).run()

def brent(function, a, b, **options):
    """Run brent with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="brent", **options).run()

def chandrupatla(function, a, b, **options):
    """Run chandrupatla with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="chandrupatla", **options).run()

def ridders(function, a, b, **options):
    """Run ridders with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="ridders", **options).run()

def toms748(function, a, b, **options):
    """Run toms748 with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="toms748", **options).run()

def wu(function, a, b, **options):
    """Run wu with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="wu", **options).run()

def itp(function, a, b, **options):
    """Run itp with Python or native scalar callbacks."""
    return BracketSolver(function, a, b, method="itp", **options).run()
