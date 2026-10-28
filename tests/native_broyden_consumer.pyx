"""Installed native Broyden callbacks, header probes and allocation inspection."""
from libc.stddef cimport size_t
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.broyden cimport BroydenGood, BroydenRecord
from cyroot.native.broyden_ops cimport broyden_update
from cyroot.native.buffers cimport OwnedDoubleBuffer

cdef extern from 'Python.h':
    int PyGILState_Check() noexcept nogil

cdef struct Context:
    size_t calls
    size_t gil_held
    int fail_after
    int code

cdef int residual(const double* x,size_t n,double* out,void* context) noexcept nogil:
    cdef Context* p=<Context*>context
    p.calls+=1
    if PyGILState_Check():p.gil_held+=1
    if p.fail_after>=0 and p.calls><size_t>p.fail_after:return p.code
    if n!=2:return -99
    out[0]=x[0]*x[0]+x[1]-2.
    out[1]=x[0]+2*x[1]*x[1]-3.
    return 0

cdef class Owner:
    cdef Context context
    cdef object __weakref__
    def __init__(self,fail_after=-1,code=-71):
        self.context.calls=0
        self.context.gil_held=0
        self.context.fail_after=fail_after
        self.context.code=code
    def callback(self):return make_vector_callback(residual,2,<void*>&self.context,self)
    @property
    def calls(self):return self.context.calls
    @property
    def gil_held(self):return self.context.gil_held


def allocation(BroydenGood solver):
    if not solver._initialized or solver._busy:raise RuntimeError('idle initialized solver required')
    return ((<size_t>solver._storage.data(),<size_t>solver._history_data.data(),
             <size_t>solver._pivots.data(),<size_t>solver._records.data()),
            (solver._storage.size(),solver._history_data.size(),solver._pivots.size(),
             solver._pivots.capacity(),solver._records.size(),solver._records.capacity()))


def budget(size_t n,bint inverse,size_t history=0):
    return ((6*n+(2 if inverse else 3)*n*n)*sizeof(double)+(0 if inverse else n*sizeof(int)),
            history*((2*n+n*n)*sizeof(double)+sizeof(BroydenRecord)))


def update(matrix,direction,target,n_override=None):
    cdef size_t n=len(direction),i,j
    cdef size_t requested=n if n_override is None else n_override
    cdef OwnedDoubleBuffer a=OwnedDoubleBuffer(n*n)
    cdef OwnedDoubleBuffer d=OwnedDoubleBuffer(n)
    cdef OwnedDoubleBuffer t=OwnedDoubleBuffer(n)
    cdef OwnedDoubleBuffer out=OwnedDoubleBuffer(n*n+2)
    cdef int status
    for i in range(n):
        d.data()[i]=direction[i];t.data()[i]=target[i]
        for j in range(n):a.data()[i+n*j]=matrix[i][j]
    out.fill(987654.)
    with nogil:status=broyden_update(a.data(),d.data(),t.data(),requested,out.data()+1)
    return (status,tuple(tuple(out.data()[1+i+n*j] for j in range(n)) for i in range(n)),
            (out.data()[0],out.data()[n*n+1]))

cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootBroydenGate {
        std::atomic<bool> entered_{false}, released_{false};
    public:
        bool entered() const noexcept {return entered_.load();}
        void release() noexcept {released_.store(true);}
        int compute(const double* x,std::size_t n,double* out) noexcept {
            if(n!=1)return -97;
            entered_.store(true);
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while(!released_.load()) {
                if(std::chrono::steady_clock::now()>=deadline)return -98;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            out[0]=x[0]-1.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootBroydenGate:
        CyrootBroydenGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int compute(const double* x,size_t n,double* out) noexcept nogil

cdef int waiting(const double* x,size_t n,double* out,void* context) noexcept nogil:
    return (<CyrootBroydenGate*>context).compute(x,n,out)

cdef class WaitingOwner:
    cdef CyrootBroydenGate* gate
    def __cinit__(self):self.gate=new CyrootBroydenGate()
    def __dealloc__(self):del self.gate
    def callback(self):return make_vector_callback(waiting,1,<void*>self.gate,self)
    @property
    def entered(self):return self.gate.entered()
    def release(self):self.gate.release()


def initialization_budget(size_t n,bint inverse,bint numerical,size_t steps=0,size_t history=0):
    # Initial inverse from a Jacobian uses good's LU/pivot footprint. Existing
    # direct-inverse constructor continues using the smaller old budget().
    return ((6*n+3*n*n+(5*n if numerical else 0)+steps)*sizeof(double)+n*sizeof(int),
            history*((2*n+n*n)*sizeof(double)+sizeof(BroydenRecord)))
