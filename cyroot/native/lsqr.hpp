#ifndef CYROOT_NATIVE_LSQR_HPP
#define CYROOT_NATIVE_LSQR_HPP
#include "gmres.hpp"

namespace cyroot { namespace native {

// Verify the actual residual and its transpose product, rather than accepting
// the bidiagonal recurrence's estimates. The normalized transpose residual
// avoids constructing ||A^T rhs||, which can overflow unnecessarily.
inline int lsqr_verify(std::size_t n,std::size_t m,const double* rhs,
    double rhs_norm,double initial_alpha,double rtol,const double* solution,
    double* applied,double* residual,double* normal,KrylovOperator apply,
    KrylovOperator transpose,void* context,double* relative,
    double* stationarity) noexcept {
    const double poison=std::numeric_limits<double>::quiet_NaN();
    std::fill(applied,applied+m,poison);
    if (apply(solution,applied,context)!=0) return 4;
    for (std::size_t i=0;i<m;++i) {
        ObjectiveAccumulator sum;
        sum.add_product(rhs[i],1.0);sum.add_product(-applied[i],1.0);
        if (sum.value(residual+i)!=0) return 3;
    }
    const double actual=krylov_norm(residual,m);
    if (!std::isfinite(actual)) return 3;
    *relative=actual/rhs_norm;
    if (!std::isfinite(*relative)) return 3;
    for (std::size_t i=0;i<m;++i) residual[i]/=rhs_norm;
    std::fill(normal,normal+n,poison);
    if (transpose(residual,normal,context)!=0) return 4;
    const double normal_norm=krylov_norm(normal,n);
    if (!std::isfinite(normal_norm)) return 3;
    *stationarity=initial_alpha==0.0
        ? (normal_norm==0.0?0.0:std::numeric_limits<double>::infinity())
        : normal_norm/initial_alpha;
    if (*relative<=rtol || *stationarity<=rtol) return 0;
    return 1;
}

// Golub-Kahan LSQR for a rectangular linear operator. Trusted n,m>0,
// max_iterations>0,0<rtol<1; disjoint rhs[m], solution[n], applied[m],
// work[3*m+4*n]. apply maps n->m, transpose maps m->n; the two callbacks
// MUST be a consistent linear operator and its adjoint. No allocation or
// normal matrix. The zero initial guess keeps the step in range(A^T), giving
// the minimum-norm least-squares solution in exact arithmetic. Loss of
// orthogonality in finite precision can limit accuracy on ill-conditioned
// systems; this storage-bounded method does not retain a Krylov basis.
//
// Returns 0 verified compatible residual OR least-squares stationarity,
// 1 iteration limit,2 recurrence breakdown,3 nonfinite arithmetic/output,
// 4 callback failure. Status0 does NOT imply the original problem has a root.
// relative=||rhs-A*solution||/||rhs||, stationarity=||A^T residual||/||A^T rhs||.
// A zero denominator has stationarity0 only for a zero normal residual.
// The recurrence's Paige-Saunders estimates only decide WHEN to verify: the
// actual residual and normal residual are recomputed (two extra operator
// applications) once an estimate meets rtol, and at every return, including
// the iteration limit and breakdown. Status0 therefore always rests on fresh
// products, and the returned diagnostics and applied=A*solution are verified.
// Each callback output is poisoned before use; unwritten/nonfinite elements
// cause failure. iterations counts completed LSQR updates, not callback calls.
// On failures outputs are scratch.
inline int native_lsqr(std::size_t n,std::size_t m,
    std::size_t max_iterations,double rtol,const double* rhs,double* solution,
    double* applied,double* work,KrylovOperator apply,KrylovOperator transpose,
    void* context,std::size_t* iterations,double* relative,
    double* stationarity) noexcept {
    double* u=work;
    double* next_u=u+m;
    double* residual=next_u+m;
    double* v=residual+m;
    double* next_v=v+n;
    double* w=next_v+n;
    double* normal=w+n;
    const double poison=std::numeric_limits<double>::quiet_NaN();
    *iterations=0;
    *relative=*stationarity=std::numeric_limits<double>::infinity();
    std::fill(solution,solution+n,0.0);
    std::fill(applied,applied+m,poison);
    const double rhs_norm=krylov_norm(rhs,m);
    if (!std::isfinite(rhs_norm)) return 3;
    if (rhs_norm==0.0) {
        std::fill(applied,applied+m,0.0);
        *relative=*stationarity=0.0;return 0;
    }
    for (std::size_t i=0;i<m;++i) u[i]=rhs[i]/rhs_norm;
    std::fill(v,v+n,poison);
    if (transpose(u,v,context)!=0) return 4;
    double alpha=krylov_norm(v,n);
    if (!std::isfinite(alpha)) return 3;
    const double initial_alpha=alpha;
    if (alpha==0.0) return lsqr_verify(n,m,rhs,rhs_norm,initial_alpha,
        rtol,solution,applied,residual,normal,apply,transpose,context,
        relative,stationarity);
    for (std::size_t j=0;j<n;++j) w[j]=v[j]/=alpha;
    double rhobar=alpha,phibar=rhs_norm;
    while (*iterations<max_iterations) {
        std::fill(next_u,next_u+m,poison);
        if (apply(v,next_u,context)!=0) return 4;
        for (std::size_t i=0;i<m;++i) {
            next_u[i]=std::fma(-alpha,u[i],next_u[i]);
            if (!std::isfinite(next_u[i])) return 3;
        }
        const double beta=krylov_norm(next_u,m);
        if (!std::isfinite(beta)) return 3;
        if (beta!=0.0) {
            for (std::size_t i=0;i<m;++i) next_u[i]/=beta;
            std::fill(next_v,next_v+n,poison);
            if (transpose(next_u,next_v,context)!=0) return 4;
            for (std::size_t j=0;j<n;++j) {
                next_v[j]=std::fma(-beta,v[j],next_v[j]);
                if (!std::isfinite(next_v[j])) return 3;
            }
        } else std::fill(next_v,next_v+n,0.0);
        const double next_alpha=krylov_norm(next_v,n);
        if (!std::isfinite(next_alpha)) return 3;
        if (next_alpha!=0.0)
            for (std::size_t j=0;j<n;++j) next_v[j]/=next_alpha;
        const double rho=std::hypot(rhobar,beta);
        if (!std::isfinite(rho)) return 3;
        if (rho==0.0) {
            const int status=lsqr_verify(n,m,rhs,rhs_norm,initial_alpha,
                rtol,solution,applied,residual,normal,apply,transpose,context,
                relative,stationarity);
            return status==1?2:status;
        }
        const double cosine=rhobar/rho,sine=beta/rho;
        const double theta=sine*next_alpha;
        rhobar=-cosine*next_alpha;
        const double phi=cosine*phibar;
        phibar=sine*phibar;
        const double update=phi/rho,rotation=theta/rho;
        if (!std::isfinite(update) || !std::isfinite(rotation)) return 3;
        for (std::size_t j=0;j<n;++j) {
            ObjectiveAccumulator sum;
            sum.add_product(solution[j],1.0);sum.add_product(update,w[j]);
            if (sum.value(solution+j)!=0) return 3;
            w[j]=std::fma(-rotation,w[j],next_v[j]);
            if (!std::isfinite(w[j])) return 3;
        }
        ++*iterations;
        // ||r||=phibar and ||A^T r||=phibar*alpha_next*|c| in exact arithmetic,
        // normalized like the verified diagnostics. They may be optimistic
        // after loss of orthogonality, which a failed verification absorbs.
        const double estimated_relative=phibar/rhs_norm;
        const double estimated_stationarity=
            estimated_relative*(next_alpha/initial_alpha)*std::fabs(cosine);
        const bool last=*iterations>=max_iterations || beta==0.0 || next_alpha==0.0;
        if (last || !(estimated_relative>rtol) || !(estimated_stationarity>rtol)) {
            const int status=lsqr_verify(n,m,rhs,rhs_norm,initial_alpha,
                rtol,solution,applied,residual,normal,apply,transpose,context,
                relative,stationarity);
            if (status!=1) return status;
            if (beta==0.0 || next_alpha==0.0) return 2;
        }
        std::swap(u,next_u);std::swap(v,next_v);alpha=next_alpha;
    }
    return 1;
}
}} // namespace cyroot::native
#endif
