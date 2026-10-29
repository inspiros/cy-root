#ifndef CYROOT_NATIVE_BFGS_MATH_HPP
#define CYROOT_NATIVE_BFGS_MATH_HPP
#include "objective.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace cyroot { namespace native {
// Trusted sized/disjoint spans. Matrices column-major. No heap or Python.
// Status0 success,1 unusable numerical result. Scratch may be partial on failure.
inline int bfgs_value(const double* f,std::size_t m,double scale,double* out) noexcept {
    if (!(scale>0.) || !std::isfinite(scale)) return 1;
    int es=0; const double ms=std::frexp(scale,&es);
    ObjectiveAccumulator sum;
    for (std::size_t i=0;i<m;++i) {
        if (!std::isfinite(f[i])) return 1;
        int e=0; const double v=std::frexp(f[i],&e)/ms;
        sum.add_scaled(.5*v*v,2*(e-es));
    }
    return sum.value(out);
}
inline int bfgs_gradient(const double* f,const double* j,std::size_t n,
                         std::size_t m,double scale,double* out) noexcept {
    if (!(scale>0.) || !std::isfinite(scale)) return 1;
    int es=0; const double ms=std::frexp(scale,&es);
    for (std::size_t k=0;k<n;++k) {
        ObjectiveAccumulator sum;
        for (std::size_t i=0;i<m;++i) {
            if (!std::isfinite(f[i]) || !std::isfinite(j[i+m*k])) return 1;
            int ef=0,ej=0;
            const double mf=std::frexp(f[i],&ef),mj=std::frexp(j[i+m*k],&ej);
            sum.add_scaled((mf/ms)*(mj/ms),ef+ej-2*es);
        }
        if (sum.value(out+k)!=0) return 1;
    }
    return 0;
}
// Residual-relative stationarity ||J^T F||inf / (||F||inf * max|J_ij|), which
// is invariant to residual scaling and stays O(1) near a simple or multiple
// root, unlike the objective gradient. A zero Jacobian is exactly stationary.
inline int bfgs_stationarity(const double* f,const double* j,std::size_t n,
                             std::size_t m,double* out) noexcept {
    double fs=0.,js=0.;
    for (std::size_t i=0;i<m;++i) {
        if (!std::isfinite(f[i])) return 1;
        fs=std::max(fs,std::fabs(f[i]));
    }
    for (std::size_t k=0;k<n*m;++k) {
        if (!std::isfinite(j[k])) return 1;
        js=std::max(js,std::fabs(j[k]));
    }
    if (fs==0. || js==0.) {*out=0.;return 0;}
    double norm=0.;
    for (std::size_t k=0;k<n;++k) {
        ObjectiveAccumulator sum;
        for (std::size_t i=0;i<m;++i) sum.add_product(j[i+m*k]/js,f[i]/fs);
        double value;
        if (sum.value(&value)!=0) return 1;
        norm=std::max(norm,std::fabs(value));
    }
    *out=norm;return 0;
}
inline int bfgs_change(const double* f,const double* trial,std::size_t m,
                       double scale,double* out) noexcept {
    int es=0; const double ms=std::frexp(scale,&es);
    ObjectiveAccumulator sum;
    for (std::size_t i=0;i<m;++i) {
        int ef=0,et=0;
        const double mf=std::frexp(f[i],&ef),mt=std::frexp(trial[i],&et);
        const int e=mf==0.?et:mt==0.?ef:std::max(ef,et);
        const double a=std::ldexp(mf,ef-e),b=std::ldexp(mt,et-e);
        sum.add_scaled(.5*((b-a)/ms)*((b+a)/ms),2*(e-es));
    }
    return sum.value(out);
}
inline int bfgs_dot(const double* a,const double* b,std::size_t n,double* out) noexcept {
    ObjectiveAccumulator sum;
    for (std::size_t i=0;i<n;++i) sum.add_product(a[i],b[i]);
    return sum.value(out);
}
inline void bfgs_identity(double* h,std::size_t n) noexcept {
    for (std::size_t k=0;k<n;++k)
        for (std::size_t i=0;i<n;++i) h[i+n*k]=i==k?1.:0.;
}
inline int bfgs_direction(const double* h,const double* g,std::size_t n,
                          double* direction,double* slope) noexcept {
    for (std::size_t i=0;i<n;++i) {
        ObjectiveAccumulator sum;
        for (std::size_t k=0;k<n;++k) sum.add_product(-h[i+n*k],g[k]);
        if (sum.value(direction+i)!=0) return 1;
    }
    return bfgs_dot(g,direction,n,slope);
}
inline int bfgs_trial(const double* x,const double* p,std::size_t n,double alpha,
                      double* trial,double* dx) noexcept {
    for (std::size_t i=0;i<n;++i) {
        const double next=x[i]+alpha*p[i];
        if (!std::isfinite(next) || !std::isfinite(next-x[i])) return 1;
        trial[i]=next; dx[i]=next-x[i];
    }
    return 0;
}
// Normalized inverse-BFGS update avoids forming overflowing s^T y or y^T H y.
// h is unchanged on failure; caller may retain it and count a skipped update.
// u,v,w[n] and candidate[n*n] are scratch, disjoint from h,g,next_g,step.
inline int bfgs_update(double* h,const double* g,const double* next_g,
                       const double* step,std::size_t n,double* candidate,
                       double* u,double* v,double* w) noexcept {
    double sn=0.,yn=0.;
    for (std::size_t i=0;i<n;++i) {
        const double y=next_g[i]-g[i];
        if (!std::isfinite(y) || !std::isfinite(step[i])) return 1;
        v[i]=y; sn=std::max(sn,std::fabs(step[i]));yn=std::max(yn,std::fabs(y));
    }
    if (!(sn>0.) || !(yn>0.)) return 1;
    for (std::size_t i=0;i<n;++i) {u[i]=step[i]/sn;v[i]/=yn;}
    double curvature=0.;
    if (bfgs_dot(u,v,n,&curvature)!=0 || !(curvature>1e-12)) return 1;
    for (std::size_t i=0;i<n;++i) {
        ObjectiveAccumulator sum;
        for (std::size_t k=0;k<n;++k) sum.add_product(h[i+n*k],v[k]);
        if (sum.value(w+i)!=0) return 1;
    }
    double vhw=0.;
    if (bfgs_dot(v,w,n,&vhw)!=0 || !(vhw>0.)) return 1;
    int es=0,ey=0;
    const double ms=std::frexp(sn,&es),my=std::frexp(yn,&ey);
    const double first=std::ldexp(ms/(my*curvature),es-ey);
    const double second=(vhw/curvature)/curvature;
    if (!std::isfinite(first) || !std::isfinite(second)) return 1;
    for (std::size_t k=0;k<n;++k) for (std::size_t i=0;i<=k;++i) {
        ObjectiveAccumulator sum;
        sum.add_product(h[i+n*k],1.);
        sum.add_triple(first,u[i],u[k]);sum.add_triple(second,u[i],u[k]);
        sum.add_triple(-w[i]/curvature,u[k],1.);
        sum.add_triple(-u[i]/curvature,w[k],1.);
        double value=0.;
        if (sum.value(&value)!=0 || (i==k && !(value>0.))) return 1;
        candidate[i+n*k]=value;candidate[k+n*i]=value;
    }
    for (std::size_t i=0;i<n*n;++i) h[i]=candidate[i];
    return 0;
}
}}
#endif
