#ifndef CYROOT_NATIVE_DOGLEG_MATH_HPP
#define CYROOT_NATIVE_DOGLEG_MATH_HPP
#include "bfgs_math.hpp"
#include <cmath>
namespace cyroot { namespace native {
// Trusted sized/disjoint spans; A[m*n] column-major, normalized residual rhs[m].
inline int dogleg_prepare(const double* f,const double* j,std::size_t n,std::size_t m,
    double* scales,double* a,double* rhs,double* g,double* scale,double* gradient) noexcept {
    double s=0.;for (std::size_t i=0;i<m;++i) {
        if (!std::isfinite(f[i])) return 1;
        s=std::max(s,std::fabs(f[i]));
    }
    if (s==0.) s=1.;
    for (std::size_t i=0;i<m;++i) rhs[i]=-f[i]/s;
    double norm=0.;
    for (std::size_t k=0;k<n;++k) {
        double d=0.;for (std::size_t i=0;i<m;++i) {
            if (!std::isfinite(j[i+m*k])) return 1;
            d=std::max(d,std::fabs(j[i+m*k]));
        }
        if (d==0.) d=1.;
        scales[k]=d;
        ObjectiveAccumulator sum;
        for (std::size_t i=0;i<m;++i) {a[i+m*k]=j[i+m*k]/d;sum.add_product(a[i+m*k],-rhs[i]);}
        if (sum.value(g+k)!=0) return 1;
        norm=std::max(norm,std::fabs(g[k]));
    }
    *scale=s;*gradient=norm;return 0;
}
inline double dogleg_norm(const double* x,std::size_t n) noexcept {
    double norm=0.;for (std::size_t i=0;i<n;++i) norm=std::hypot(norm,x[i]);return norm;
}
// Powell path: zero -> Cauchy point -> minimum-norm Gauss-Newton point.
// kind0 GN,1 gradient boundary,2 second-segment boundary. Scratch c[n],ag[m].
// Status1 nonfinite/range,2 degenerate model. No allocation or normal equations.
inline int dogleg_step(const double* a,const double* g,const double* gn,
    std::size_t n,std::size_t m,double radius,double* c,double* ag,double* out,int* kind) noexcept {
    if (!(radius>0.) || !std::isfinite(radius)) return 1;
    const double nn=dogleg_norm(gn,n),ng=dogleg_norm(g,n);
    if (!std::isfinite(nn) || !std::isfinite(ng)) return 1;
    if (nn<=radius) {for (std::size_t i=0;i<n;++i) out[i]=gn[i];*kind=0;return 0;}
    if (!(ng>0.)) return 2;
    for (std::size_t i=0;i<n;++i) c[i]=g[i]/ng;
    for (std::size_t i=0;i<m;++i) {
        ObjectiveAccumulator sum;for (std::size_t k=0;k<n;++k) sum.add_product(a[i+m*k],c[k]);
        if (sum.value(ag+i)!=0) return 1;
    }
    const double na=dogleg_norm(ag,m);
    if (!(na>0.) || !std::isfinite(na)) return 2;
    const double length=(ng/na)/na;
    if (length>=radius) {
        for (std::size_t i=0;i<n;++i) out[i]=-radius*c[i];
        *kind=1;return 0;
    }
    if (!std::isfinite(length)) return 1;
    for (std::size_t i=0;i<n;++i) {c[i]*=-length;out[i]=gn[i]-c[i];}
    const double nd=dogleg_norm(out,n);
    if (!(nd>0.) || !std::isfinite(nd)) return 2;
    ObjectiveAccumulator dot,norm;
    for (std::size_t i=0;i<n;++i) {
        out[i]/=nd;dot.add_product(c[i]/radius,out[i]);norm.add_product(c[i]/radius,c[i]/radius);
    }
    double b,cc;
    if (dot.value(&b)!=0 || norm.value(&cc)!=0) return 1;
    const double remaining=std::max(0.,1.-cc);
    const double root=std::sqrt(b*b+remaining);
    // Rationalized positive root when b>0 prevents subtraction cancellation.
    const double distance=radius*(b>0.?remaining/(root+b):root-b);
    if (!(distance>=0.) || !std::isfinite(distance)) return 1;
    for (std::size_t i=0;i<n;++i) {
        out[i]=c[i]+distance*out[i];if (!std::isfinite(out[i])) return 1;
    }
    *kind=2;return 0;
}
}}
#endif
