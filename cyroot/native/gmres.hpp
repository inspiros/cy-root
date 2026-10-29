#ifndef CYROOT_NATIVE_GMRES_HPP
#define CYROOT_NATIVE_GMRES_HPP
#include "objective.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cyroot { namespace native {
using KrylovOperator = int (*)(const double*, double*, void*);

inline double krylov_norm(const double* x, std::size_t n) noexcept {
    double scale=0.0, sum=1.0;
    for (std::size_t i=0;i<n;++i) {
        const double a=std::fabs(x[i]);
        if (!std::isfinite(a)) return std::numeric_limits<double>::infinity();
        if (a==0.0) continue;
        if (a>scale) {const double q=scale/a;sum=1.0+sum*q*q;scale=a;}
        else {const double q=a/scale;sum+=q*q;}
    }
    return scale==0.0?0.0:scale*std::sqrt(sum);
}

inline int krylov_dot(const double* a,const double* b,std::size_t n,double* out) noexcept {
    ObjectiveAccumulator sum;
    for (std::size_t i=0;i<n;++i) sum.add_product(a[i],b[i]);
    return sum.value(out);
}

// Restarted GMRES with twice-modified Gram-Schmidt and true-residual checks.
// Trusted n,k>0, k<=n, max_iterations>0, 0<rtol<1; disjoint rhs[n],
// solution[n], applied[n], work[(k+1)*n+(k+1)*k+4*k+1+2*n]. No allocation.
// Returns 0 convergence,1 iteration limit,2 breakdown,3 nonfinite arithmetic,
//4 operator failure. apply must write all n outputs on zero status. Scratch
// outputs are poisoned before each call. Successful applied is A*solution. Convergence
// requires ||rhs-A*solution||_2 <= rtol*||rhs||_2, never just a Givens estimate.
inline int restarted_gmres(std::size_t n,std::size_t k,std::size_t max_iterations,
    double rtol,const double* rhs,double* solution,double* applied,double* work,
    KrylovOperator apply,void* context,std::size_t* iterations,double* relative) noexcept {
    double* basis=work;
    double* h=basis+(k+1)*n;
    double* cs=h+(k+1)*k;
    double* sn=cs+k;
    double* g=sn+k;
    double* y=g+k+1;
    double* w=y+k;
    double* residual=w+n;
    *iterations=0;
    *relative=std::numeric_limits<double>::infinity();
    std::fill(solution,solution+n,0.0);
    std::fill(applied,applied+n,std::numeric_limits<double>::quiet_NaN());
    std::copy(rhs,rhs+n,residual);
    const double rhs_norm=krylov_norm(rhs,n);
    if (!std::isfinite(rhs_norm)) return 3;
    if (rhs_norm==0.0) {std::fill(applied,applied+n,0.0);*relative=0.0;return 0;}
    const double target=rtol*rhs_norm;
    while (*iterations<max_iterations) {
        const double beta=krylov_norm(residual,n);
        if (!std::isfinite(beta)) return 3;
        *relative=beta/rhs_norm;
        if (beta<=target) return 0;
        for (std::size_t i=0;i<n;++i) basis[i]=residual[i]/beta;
        std::fill(h,h+(k+1)*k,0.0);
        std::fill(g,g+k+1,0.0);
        g[0]=beta;
        std::size_t count=0;
        bool breakdown=false;
        for (std::size_t j=0;j<k && *iterations<max_iterations;++j) {
            std::fill(w,w+n,std::numeric_limits<double>::quiet_NaN());
            if (apply(basis+j*n,w,context)!=0) return 4;
            ++*iterations;
            for (int pass=0;pass<2;++pass) for (std::size_t i=0;i<=j;++i) {
                double projection;
                if (krylov_dot(basis+i*n,w,n,&projection)!=0) return 3;
                h[i+(k+1)*j]+=projection;
                if (!std::isfinite(h[i+(k+1)*j])) return 3;
                for (std::size_t row=0;row<n;++row) {
                    w[row]=std::fma(-projection,basis[row+i*n],w[row]);
                    if (!std::isfinite(w[row])) return 3;
                }
            }
            const double next_norm=krylov_norm(w,n);
            if (!std::isfinite(next_norm)) return 3;
            h[j+1+(k+1)*j]=next_norm;
            breakdown=next_norm==0.0;
            if (!breakdown) for (std::size_t row=0;row<n;++row)
                basis[row+(j+1)*n]=w[row]/next_norm;
            for (std::size_t i=0;i<j;++i) {
                const double a=h[i+(k+1)*j],b=h[i+1+(k+1)*j];
                h[i+(k+1)*j]=std::fma(cs[i],a,sn[i]*b);
                h[i+1+(k+1)*j]=std::fma(-sn[i],a,cs[i]*b);
            }
            const double a=h[j+(k+1)*j],b=h[j+1+(k+1)*j];
            const double rho=std::hypot(a,b);
            if (!std::isfinite(rho)) return 3;
            if (rho==0.0) {breakdown=true;break;}
            cs[j]=a/rho;sn[j]=b/rho;
            h[j+(k+1)*j]=rho;h[j+1+(k+1)*j]=0.0;
            const double old=g[j];g[j]=cs[j]*old;g[j+1]=-sn[j]*old;
            count=j+1;
            if (std::fabs(g[j+1])<=target || breakdown) break;
        }
        if (count==0) return 2;
        for (std::size_t row=count;row-->0;) {
            ObjectiveAccumulator sum;sum.add_product(g[row],1.0);
            for (std::size_t j=row+1;j<count;++j) sum.add_product(-h[row+(k+1)*j],y[j]);
            double value;
            if (sum.value(&value)!=0) return 3;
            const double diagonal=h[row+(k+1)*row];
            if (diagonal==0.0) return 2;
            y[row]=value/diagonal;
            if (!std::isfinite(y[row])) return 3;
        }
        for (std::size_t row=0;row<n;++row) {
            ObjectiveAccumulator sum;sum.add_product(solution[row],1.0);
            for (std::size_t j=0;j<count;++j) sum.add_product(basis[row+j*n],y[j]);
            if (sum.value(solution+row)!=0) return 3;
        }
        std::fill(applied,applied+n,std::numeric_limits<double>::quiet_NaN());
        if (apply(solution,applied,context)!=0) return 4;
        for (std::size_t row=0;row<n;++row) residual[row]=rhs[row]-applied[row];
        const double actual=krylov_norm(residual,n);
        if (!std::isfinite(actual)) return 3;
        *relative=actual/rhs_norm;
        if (actual<=target) return 0;
        if (breakdown) return 2;
    }
    return 1;
}

inline int krylov_merit_slope(const double* f,const double* normalized_product,
                              std::size_t n,double scale,double* out) noexcept {
    ObjectiveAccumulator sum;
    for (std::size_t i=0;i<n;++i) sum.add_product(f[i]/scale,normalized_product[i]);
    return sum.value(out);
}
}} // namespace cyroot::native
#endif
