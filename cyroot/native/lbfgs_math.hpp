#ifndef CYROOT_NATIVE_LBFGS_MATH_HPP
#define CYROOT_NATIVE_LBFGS_MATH_HPP
#include "bfgs_math.hpp"
namespace cyroot { namespace native {
// Trusted disjoint spans, normalized secant pairs in a preallocated ring.
// First loop q -= ((u^T q)/c)v; second loop adds
// (ratio*alpha-(v^T r)/c)u, where u=s/|s|inf,v=y/|y|inf,
// c=u^T v, ratio=|s|inf/|y|inf. No raw overflowing s^T y.
inline int lbfgs_direction(const double* g,std::size_t n,std::size_t capacity,
    std::size_t count,std::size_t next,const double* us,const double* vs,
    const double* ratios,const double* curvatures,double gamma,
    double* alphas,double* q,double* direction,double* slope) noexcept {
    for (std::size_t i=0;i<n;++i) q[i]=g[i];
    for (std::size_t k=0;k<count;++k) {
        const std::size_t slot=(next+capacity-1-k)%capacity;
        double dot;
        if (bfgs_dot(us+slot*n,q,n,&dot)!=0) return 1;
        const double a=dot/curvatures[slot];
        if (!std::isfinite(a)) return 1;
        alphas[slot]=a;
        for (std::size_t i=0;i<n;++i) {
            ObjectiveAccumulator sum;sum.add_product(q[i],1.);
            sum.add_product(-a,vs[slot*n+i]);
            if (sum.value(q+i)!=0) return 1;
        }
    }
    for (std::size_t i=0;i<n;++i) {
        direction[i]=gamma*q[i];
        if (!std::isfinite(direction[i])) return 1;
    }
    for (std::size_t k=count;k>0;--k) {
        const std::size_t slot=(next+capacity-k)%capacity;
        double dot,coefficient;
        if (bfgs_dot(vs+slot*n,direction,n,&dot)!=0) return 1;
        ObjectiveAccumulator sum;sum.add_product(ratios[slot],alphas[slot]);
        sum.add_product(-dot/curvatures[slot],1.);
        if (sum.value(&coefficient)!=0) return 1;
        for (std::size_t i=0;i<n;++i) {
            ObjectiveAccumulator item;item.add_product(direction[i],1.);
            item.add_product(coefficient,us[slot*n+i]);
            if (item.value(direction+i)!=0) return 1;
        }
    }
    for (std::size_t i=0;i<n;++i) direction[i]=-direction[i];
    return bfgs_dot(g,direction,n,slope);
}
// Fully validate a pair before overwriting an old ring slot. A failed pair
// leaves every retained pair, count, next and gamma unchanged.
inline int lbfgs_store(const double* g,const double* next_g,const double* step,
    std::size_t n,std::size_t capacity,std::size_t* count,std::size_t* next,
    double* us,double* vs,double* ratios,double* curvatures,double* gamma,
    double* u,double* v) noexcept {
    double sn=0.,yn=0.;
    for (std::size_t i=0;i<n;++i) {
        v[i]=next_g[i]-g[i];
        if (!std::isfinite(v[i]) || !std::isfinite(step[i])) return 1;
        sn=std::max(sn,std::fabs(step[i]));yn=std::max(yn,std::fabs(v[i]));
    }
    if (!(sn>0.) || !(yn>0.)) return 1;
    for (std::size_t i=0;i<n;++i) {u[i]=step[i]/sn;v[i]/=yn;}
    double c,vv;
    if (bfgs_dot(u,v,n,&c)!=0 || !(c>1e-12) || bfgs_dot(v,v,n,&vv)!=0) return 1;
    int es=0,ey=0;const double ms=std::frexp(sn,&es),my=std::frexp(yn,&ey);
    const double ratio=std::ldexp(ms/my,es-ey);
    const double scaling=ratio*(c/vv);
    if (!(ratio>0.) || !std::isfinite(ratio) || !(scaling>0.) || !std::isfinite(scaling)) return 1;
    const std::size_t slot=*next;
    for (std::size_t i=0;i<n;++i) {us[slot*n+i]=u[i];vs[slot*n+i]=v[i];}
    ratios[slot]=ratio;curvatures[slot]=c;*gamma=scaling;
    *next=(slot+1)%capacity;if (*count<capacity) ++*count;
    return 0;
}
}}
#endif
