# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Transform user-provided residual derivative tensors into objective derivatives."""
from numbers import Complex, Real
from libc.limits cimport INT_MAX
from libc.math cimport isfinite
from .buffers cimport OwnedDoubleBuffer
from .higher_derivatives import _checked_integer


cdef void _pack_axes(object source,double* out,size_t n,size_t m,
                     size_t remaining,size_t flat,size_t stride,size_t output) except *:
    cdef size_t j
    if remaining == 0:
        if ((isinstance(source,Complex) and not isinstance(source,Real)) or
                getattr(getattr(source,'dtype',None),'kind',None) == 'c'):
            raise TypeError('Residual derivatives must contain real scalars.')
        out[output+m*flat] = source
        return
    if isinstance(source,(str,bytes,bytearray)) or len(source) != n:
        raise ValueError('Derivative tensor shape must be (m,)+(n,)*order.')
    for j in range(n):
        _pack_axes(source[j],out,n,m,remaining-1,flat+j*stride,stride*n,output)


cdef object _nested(const double* data,size_t n,size_t remaining,size_t flat,size_t stride):
    cdef size_t j
    if remaining == 0:return data[flat]
    return tuple([_nested(data,n,remaining-1,flat+j*stride,stride*n) for j in range(n)])


cdef class ObjectiveDerivativePlan:
    """Exact order-r derivative of .5*sum(F_i**2), from D^0F through D^rF.

    transform(tensors) consumes derivative VALUES. evaluate(F,derivatives,x)
    calls ordinary Python providers once each at a copied point, then uses the
    same native transform. Typed native callers supply packed tensors to compute.
    No finite-difference fallback, cached samples or symmetry assumptions.
    """
    def __init__(self,input_dimension,output_dimension,order,*,
                 max_terms=1_000_000,memory_bytes=64*1024*1024):
        cdef size_t k
        if self._initialized or self._constructing:
            raise RuntimeError('Objective derivative plans cannot be reinitialized.')
        self._constructing = True
        try:
            n=_checked_integer(input_dimension,'input_dimension')
            m=_checked_integer(output_dimension,'output_dimension')
            r=_checked_integer(order,'order')
            term_limit=_checked_integer(max_terms,'max_terms')
            budget=_checked_integer(memory_bytes,'memory_bytes')
            if not n or not m or n>INT_MAX or m>INT_MAX:
                raise ValueError('Positive dimensions must fit the native C-int callback convention.')
            if not r or r>=8*sizeof(size_t):
                raise ValueError('Positive derivative order must fit the native subset bit capacity.')
            output=n**r
            terms=output*m*(1<<(r-1))
            if terms>term_limit:
                raise ValueError('Objective derivative terms exceed max_terms.')
            sizes=[m*n**k for k in range(r+1)]
            inputs=sum(sizes)
            evaluation=(inputs+2*output)*sizeof(double)+(r+1)*(sizeof(size_t)+sizeof(double*))
            if evaluation>budget or inputs+2*output>OwnedDoubleBuffer.max_size():
                raise ValueError('Objective derivative storage exceeds memory_bytes or native capacity.')
            self._sizes.resize(r+1)
            for k in range(r+1):self._sizes[k]=sizes[k]
            self._n,self._m,self._order=n,m,r
            self._input_size,self._output_size=inputs,output
            self._terms,self._memory_bytes=terms,evaluation
            self._initialized=True
        finally:self._constructing=False

    cdef void _ready(self) except *:
        if self._constructing or not self._initialized:
            raise RuntimeError('Objective derivative plan is not initialized.')
        if self._busy:raise RuntimeError('Objective derivative plan is already active.')

    cdef int compute(self,const double* const* derivatives,const size_t* capacities,
                     size_t derivative_count,double* out,size_t output_capacity,
                     double* scratch,size_t scratch_capacity) noexcept nogil:
        if not self._initialized or self._constructing or derivative_count!=self._order+1:return 2
        return objective_derivative(derivatives,capacities,self._n,self._m,self._order,
                                    out,output_capacity,scratch,scratch_capacity,self._terms)

    cdef object _transform(self,object tensors,object point=None):
        cdef OwnedDoubleBuffer* storage=NULL
        cdef vector[const double*] pointers
        cdef double* data
        cdef double* out
        cdef double* scratch
        cdef size_t k,o,offset=0
        cdef int status
        if isinstance(tensors,(str,bytes,bytearray)) or len(tensors)!=self._order+1:
            raise ValueError('Supply residual values and every derivative through the requested order.')
        pointers.resize(self._order+1)
        try:
            storage=new OwnedDoubleBuffer(self._input_size+2*self._output_size)
            data=storage.data()
            for k in range(self._order+1):
                # Pack each provider result before invoking another provider:
                # user functions may return views into the same scratch array.
                tensor=tensors[k] if point is None else tensors[k](point)
                if isinstance(tensor,(str,bytes,bytearray)) or len(tensor)!=self._m:
                    raise ValueError('Derivative tensor shape must be (m,)+(n,)*order.')
                pointers[k]=data+offset
                for o in range(self._m):
                    _pack_axes(tensor[o],data+offset,self._n,self._m,k,0,1,o)
                offset+=self._sizes[k]
            out=data+self._input_size
            scratch=out+self._output_size
            with nogil:
                status=self.compute(pointers.data(),self._sizes.data(),self._order+1,
                                    out,self._output_size,scratch,self._output_size)
            if status==1:raise ValueError('Objective derivative inputs and final entries must be finite.')
            if status!=0:raise ValueError('Invalid objective derivative shape or workspace capacity.')
            return _nested(out,self._n,self._order,0,1)
        finally:
            if storage!=NULL:del storage

    def transform(self,tensors):
        """Convert [F(x),DF(x),...,D^rF(x)] into the dense order-r objective tensor."""
        self._ready();self._busy=True
        try:return self._transform(tensors)
        finally:self._busy=False

    def evaluate(self,function,derivatives,x):
        """Use ordinary Python derivative providers; provider j returns D^(j+1)F.

        All providers are evaluated exactly once. This convenience boundary uses
        Python/GIL; the packed native compute entry point does not call Python.
        """
        self._ready();self._busy=True
        try:
            if not callable(function):raise TypeError('function must be an ordinary Python callable.')
            if isinstance(derivatives,(str,bytes,bytearray)) or len(derivatives)!=self._order:
                raise ValueError('Supply one callable for each residual derivative order.')
            providers=tuple(derivatives[k] for k in range(self._order))
            if not all(callable(provider) for provider in providers):
                raise TypeError('Every residual derivative provider must be an ordinary Python callable.')
            if isinstance(x,(str,bytes,bytearray)) or len(x)!=self._n:
                raise ValueError('Point length must match input_dimension.')
            values=tuple(x[i] for i in range(self._n))
            if any((isinstance(v,Complex) and not isinstance(v,Real)) or
                   getattr(getattr(v,'dtype',None),'kind',None)=='c' for v in values):
                raise TypeError('Point coordinates must be real scalars.')
            point=tuple(float(v) for v in values)
            if not all(isfinite(v) for v in point):raise ValueError('Point must be finite.')
            return self._transform((function,)+providers,point)
        finally:self._busy=False

    @property
    def input_dimension(self):return self._n
    @property
    def output_dimension(self):return self._m
    @property
    def order(self):return self._order
    @property
    def shape(self):return (int(self._n),)*int(self._order)
    @property
    def term_count(self):return self._terms
    @property
    def memory_bytes(self):return self._memory_bytes
