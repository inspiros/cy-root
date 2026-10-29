# distutils: language=c++
"""A compiled objective Hessian using native callbacks with the GIL released."""
from libc.stddef cimport size_t
from cyroot.native.objective cimport LeastSquaresObjective
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.tensor_callbacks cimport make_hessian_callback

cdef int residual(const double* x,size_t n,double* out,void* context) noexcept nogil:
    if n!=2:return -1
    out[0]=x[0]*x[0]+x[1]-2.
    out[1]=x[0]*x[1]-.5
    return 0

cdef int jacobian(const double* x,size_t n,double* out,void* context) noexcept nogil:
    if n!=2:return -1
    out[0],out[1],out[2],out[3]=2*x[0],x[1],1.,x[0]
    return 0

cdef int hessian(const double* x,size_t n,double* out,void* context) noexcept nogil:
    cdef size_t i
    if n!=2:return -1
    for i in range(8):out[i]=0.
    out[0],out[3],out[5]=2.,1.,1.
    return 0

def evaluate(point=(1.,.4)):
    cdef LeastSquaresObjective objective=LeastSquaresObjective(
        make_vector_callback(residual,2,NULL,None,False),2,
        jacobian=make_vector_callback(jacobian,2,NULL,None,True),
        hessian=make_hessian_callback(hessian,2,NULL,None))
    cdef double x[2]
    cdef double out[4]
    cdef int status
    if len(point)!=2:raise ValueError('Expected a point of length two.')
    x[0],x[1]=point[0],point[1]
    objective._enter()
    try:
        with nogil:status=objective.evaluate_raw(x,2,out,4)
        objective._raise_callback_error(None)
        if status!=0:raise ArithmeticError('Native objective evaluation failed.')
        return ((out[0],out[2]),(out[1],out[3]))
    finally:objective._release()
