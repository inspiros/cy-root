"""Installed modified-Halley consumer with pivot and scaled-matvec probes."""
from libc.stddef cimport size_t
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.tensor_callbacks cimport make_hessian_callback
from cyroot.native.vector_chebyshev cimport VectorChebyshev,VectorModifiedHalley,VectorChebyshevRecord
from cyroot.native.tensor_ops cimport matrix_vector

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Parameters:
    size_t calls[3]
    size_t gil_held
    Py_ssize_t fail_after[3]
    int code

cdef bint sample(Parameters* p,size_t role) noexcept nogil:
    p.calls[role]+=1
    if PyGILState_Check():p.gil_held+=1
    return p.fail_after[role]<0 or p.calls[role]<=<size_t>p.fail_after[role]

cdef int residual(const double* x,size_t n,double* out,void* context) noexcept nogil:
    cdef Parameters* p=<Parameters*>context
    if n!=2:return -101
    if not sample(p,0):return p.code
    out[0]=x[0]*x[0]+x[0]*x[1]+x[1]-5.
    out[1]=2*x[0]+x[1]*x[1]-6.
    return 0

cdef int jacobian(const double* x,size_t n,double* out,void* context) noexcept nogil:
    cdef Parameters* p=<Parameters*>context
    if n!=2:return -102
    if not sample(p,1):return p.code
    out[0],out[1],out[2],out[3]=2*x[0]+x[1],2.,x[0]+1.,2*x[1]
    return 0

cdef int hessian(const double* x,size_t n,double* out,void* context) noexcept nogil:
    cdef Parameters* p=<Parameters*>context
    if n!=2:return -103
    if not sample(p,2):return p.code
    out[0],out[1],out[2],out[3]=2.,0.,1.,0.
    out[4],out[5],out[6],out[7]=1.,0.,0.,2.
    return 0

cdef class ContextOwner:
    cdef Parameters parameters
    cdef object __weakref__
    def __init__(self,fail_after=(-1,-1,-1),code=-47):
        cdef size_t k
        self.parameters.gil_held=0
        self.parameters.code=code
        for k in range(3):
            self.parameters.calls[k]=0
            self.parameters.fail_after[k]=fail_after[k]
    def function(self):return make_vector_callback(residual,2,<void*>&self.parameters,self)
    def jacobian(self):return make_vector_callback(jacobian,2,<void*>&self.parameters,self,jacobian=True)
    def hessian(self):return make_hessian_callback(hessian,2,<void*>&self.parameters,self)
    @property
    def calls(self):return tuple(self.parameters.calls[k] for k in range(3))
    @property
    def gil_held(self):return self.parameters.gil_held

def public_type():return VectorModifiedHalley

def allocation(VectorChebyshev solver):
    if not solver._initialized or solver._busy:raise RuntimeError('An idle initialized solver is required.')
    return (
        (<size_t>solver._storage.data(),<size_t>solver._diff_work,<size_t>solver._secondary,
         <size_t>solver._directional,<size_t>solver._pivots.data(),
         <size_t>solver._secondary_pivots.data(),<size_t>solver._history_data.data(),<size_t>solver._records.data()),
        (solver._storage.size(),solver._pivots.size(),solver._pivots.capacity(),
         solver._secondary_pivots.size(),solver._secondary_pivots.capacity(),
         solver._history_data.size(),solver._records.size(),solver._records.capacity()),
    )

def pivots(VectorChebyshev solver):
    if not solver._initialized or solver._busy:raise RuntimeError('An idle initialized solver is required.')
    return (tuple(solver._pivots[k] for k in range(solver._pivots.size())),
            tuple(solver._secondary_pivots[k] for k in range(solver._secondary_pivots.size())))

def storage_bytes(size_t n,int path,size_t scratch=0,size_t plan_steps=0):
    # path0Chebyshev, path1fixed matrix, path2general correction.
    return ((8*n+(2+path)*n*n+n*n*n+scratch+plan_steps)*sizeof(double)+
            (2 if path==2 else 1)*n*sizeof(int),
            (2*n+n*n+n*n*n)*sizeof(double)+sizeof(VectorChebyshevRecord))

def matvec(matrix,direction,n_override=None):
    cdef double values[4]
    cdef double step[2]
    cdef double output[4]
    cdef size_t n=2 if n_override is None else n_override
    cdef size_t i,j
    cdef int status
    for i in range(2):
        step[i]=direction[i]
        for j in range(2):values[i+2*j]=matrix[i][j]
    output[0],output[1],output[2],output[3]=1234567.,5555555.,5555555.,7654321.
    with nogil:status=matrix_vector(values,step,n,&output[1])
    return dict(status=status,value=(output[1],output[2]),guards=(output[0],output[3]),
                matrix=((values[0],values[2]),(values[1],values[3])),direction=(step[0],step[1]))

cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootVectorModifiedTestGate {
        std::atomic<bool> entered_{false};
        std::atomic<bool> released_{false};
    public:
        bool entered() const noexcept { return entered_.load(); }
        void release() noexcept { released_.store(true); }
        int evaluate(const double* x,std::size_t n,double* out) noexcept {
            if(n!=2)return 81;
            entered_.store(true);
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while(!released_.load()) {
                if(std::chrono::steady_clock::now()>=deadline)return 83;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            out[0]=x[0]*x[0]+x[0]*x[1]+x[1]-5.;
            out[1]=2*x[0]+x[1]*x[1]-6.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootVectorModifiedTestGate:
        CyrootVectorModifiedTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(const double* x,size_t n,double* out) noexcept nogil

cdef int waiting(const double* x,size_t n,double* out,void* context) noexcept nogil:
    return (<CyrootVectorModifiedTestGate*>context).evaluate(x,n,out)

cdef class WaitingOwner:
    cdef CyrootVectorModifiedTestGate* gate
    def __cinit__(self):self.gate=new CyrootVectorModifiedTestGate()
    def __dealloc__(self):del self.gate
    def callback(self):return make_vector_callback(waiting,2,<void*>self.gate,self)
    @property
    def entered(self):return self.gate.entered()
    def release(self):self.gate.release()
