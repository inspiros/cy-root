"""Native direct/inverse Broyden with a pinned parameter owner."""
from libc.stddef cimport size_t
from cyroot.native.vector cimport make_vector_callback
from cyroot.native import BroydenGood,BroydenBad

cdef struct Parameters:
    double first_target
    double second_target

cdef int residual(const double* x,size_t n,double* out,void* context) noexcept nogil:
    if n!=2:return -1
    cdef Parameters* p=<Parameters*>context
    out[0]=x[0]*x[0]+x[1]-p.first_target
    out[1]=x[0]+2*x[1]*x[1]-p.second_target
    return 0

cdef class CoupledSystem:
    cdef Parameters parameters
    def __init__(self):
        self.parameters.first_target=2.
        self.parameters.second_target=3.
    def callback(self):return make_vector_callback(residual,2,<void*>&self.parameters,self)


def solve(method='good',initial=(1.4,.7)):
    """Return result and copied history. Bad takes an explicit initial inverse."""
    owner=CoupledSystem()
    if method=='good':
        approximation=((2.8,1.),(1.,2.8))
        solver=BroydenGood(owner.callback(),approximation,initial,history=32)
    elif method=='bad':
        # Analytic inverse of the chosen approximation, not a solver option.
        denominator=2.8*2.8-1.
        approximation=((2.8/denominator,-1./denominator),(-1./denominator,2.8/denominator))
        solver=BroydenBad(owner.callback(),approximation,initial,history=32)
    else:raise ValueError('method must be good or bad')
    solver.step()
    return solver.run(),solver.history



def solve_initialized(method='good',scheme='central',cached=False):
    """Initialize with a native plan, optionally reusing a supplied base residual."""
    from cyroot.native import JacobianPlan
    owner=CoupledSystem()
    plan=JacobianPlan(2,scheme=scheme,step=.001)
    options={'f_x0':(.66,-.62)} if cached else {}
    if method=='good':
        solver=BroydenGood(owner.callback(),plan,(1.4,.7),history=32,**options)
    elif method=='bad':
        solver=BroydenBad.from_jacobian(owner.callback(),plan,(1.4,.7),history=32,**options)
    else:raise ValueError('method must be good or bad')
    initial=solver.result
    return solver.run(),solver.history,initial
