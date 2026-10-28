from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.objective_derivatives cimport ObjectiveDerivativePlan

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

def raw(ObjectiveDerivativePlan plan, tensors, output_capacity=None,
        scratch_capacity=None, derivative_count=None):
    cdef vector[vector[double]] values
    cdef vector[const double*] pointers
    cdef vector[size_t] capacities
    cdef vector[double] out,work
    cdef size_t i,j,count=len(tensors),needed=plan._output_size
    cdef size_t supplied=needed if output_capacity is None else output_capacity
    cdef size_t scratch=needed if scratch_capacity is None else scratch_capacity
    cdef size_t provider_count=count if derivative_count is None else derivative_count
    cdef int status,gil
    values.resize(count);pointers.resize(count);capacities.resize(count)
    for i in range(count):
        values[i].resize(len(tensors[i]))
        for j in range(len(tensors[i])):values[i][j]=tensors[i][j]
        capacities[i]=values[i].size();pointers[i]=values[i].data()
    out.resize(max(needed,supplied)+2,1234.)
    work.resize(max(needed,scratch)+2,5678.)
    with nogil:
        gil=PyGILState_Check()
        status=plan.compute(pointers.data(),capacities.data(),provider_count,
                            out.data()+1,supplied,work.data()+1,scratch)
    return status,tuple(out[i+1] for i in range(needed)),gil,(out[0],out[out.size()-1],work[0],work[work.size()-1])
