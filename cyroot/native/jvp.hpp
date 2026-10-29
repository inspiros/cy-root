#ifndef CYROOT_NATIVE_JVP_HPP
#define CYROOT_NATIVE_JVP_HPP
#include "callbacks.h"
#include "objective.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cyroot { namespace native {

// Trusted sized/disjoint spans: x[n], v[n], optional base[m], out[m],
// work[2*n+3*m]. No allocation or Python. Scheme 0/1/2=forward/backward/central.
// Status 0/1/2/3=ok/nonfinite/breakdown/callback_failure, matching DiffStatus.
inline int jvp_fail(double* out, std::size_t m, int status) noexcept {
    for (std::size_t i=0;i<m;++i) out[i]=std::numeric_limits<double>::quiet_NaN();
    return status;
}

inline int jvp_sample(const double* x, std::size_t n, std::size_t m, double* out,
                      cyroot_rectangular_function function, void* context,
                      std::size_t* calls, int* callback_code) noexcept {
    jvp_fail(out,m,0);
    if (*calls==std::numeric_limits<std::size_t>::max()) return 2;
    ++*calls;
    const int code=function(x,n,m,out,context);
    if (code!=0) { *callback_code=code; return 3; }
    for (std::size_t i=0;i<m;++i) if (!std::isfinite(out[i])) return 1;
    return 0;
}

inline void jvp_secant(double f, double base, double t, double weight,
                       double direction_scale, ObjectiveAccumulator& sum) noexcept {
    int ef=0, eb=0, et=0, ev=0;
    const double mf=std::frexp(f,&ef), mb=std::frexp(base,&eb);
    const int e=mf==0.0?eb:mb==0.0?ef:std::max(ef,eb);
    // Subtract in a common binary scale before division, so neither a large
    // residual difference nor an intermediate directional derivative overflows.
    const double difference=std::ldexp(mf,ef-e)-std::ldexp(mb,eb-e);
    const double mt=std::frexp(t,&et), mv=std::frexp(direction_scale,&ev);
    sum.add_scaled((difference*mv/mt)*weight,e+ev-et);
}

inline int jvp_compute(std::size_t n, std::size_t m, int scheme, double step,
                       const double* x, const double* v, const double* cached,
                       double* out, double* work,
                       cyroot_rectangular_function function, void* context,
                       std::size_t* calls, int* callback_code) noexcept {
    *callback_code=0;
    jvp_fail(out,m,0);
    double vscale=0.0, xscale=1.0;
    std::size_t pivot=0;
    for (std::size_t i=0;i<n;++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(v[i])) return 1;
        xscale=std::max(xscale,std::fabs(x[i]));
        if (std::fabs(v[i])>vscale) {vscale=std::fabs(v[i]);pivot=i;}
    }
    if (vscale==0.0) {
        for (std::size_t i=0;i<m;++i) out[i]=0.0;
        return 0; // no samples, including when a cached base is unavailable
    }
    const double h=step==0.0 ? xscale*(scheme==2 ? std::cbrt(std::numeric_limits<double>::epsilon())
                                                      : std::sqrt(std::numeric_limits<double>::epsilon())) : step;
    if (!std::isfinite(h) || !(h>0.0)) return 1;
    double* plus=work;
    double* minus=plus+n;
    double* fplus=minus+n;
    double* fminus=fplus+m;
    double* base_storage=fminus+m;
    const double sign=v[pivot]>0.0?1.0:-1.0;
    // Recover the scalar offsets that survive rounding at a unit-direction
    // pivot. Central weights allow different rounded positive/negative offsets.
    double a=0.0,b=0.0;
    if (scheme!=1) {
        const double p=x[pivot]+h*sign;
        a=(p-x[pivot])*sign;
        if (!std::isfinite(p) || !std::isfinite(a)) return 1;
        if (!(a>0.0)) return 2;
    }
    if (scheme!=0) {
        const double p=x[pivot]-h*sign;
        b=(p-x[pivot])*sign;
        if (!std::isfinite(p) || !std::isfinite(b)) return 1;
        if (!(b<0.0)) return 2;
    }
    // Validate the full directional stencil before evaluating any residual.
    // An active coordinate that vanishes on normalization or coordinate
    // rounding is a breakdown, not a silent change of differentiation direction.
    for (std::size_t i=0;i<n;++i) {
        const double u=v[i]/vscale;
        if (v[i]!=0.0 && u==0.0) return 2;
        if (scheme!=1) {
            plus[i]=std::fma(a,u,x[i]);
            if (!std::isfinite(plus[i])) return 1;
            if (v[i]!=0.0 && plus[i]==x[i]) return 2;
        }
        if (scheme!=0) {
            minus[i]=std::fma(b,u,x[i]);
            if (!std::isfinite(minus[i])) return 1;
            if (v[i]!=0.0 && minus[i]==x[i]) return 2;
        }
    }
    int status=0;
    const double* base=cached;
    if (base==nullptr) {
        status=jvp_sample(x,n,m,base_storage,function,context,calls,callback_code);
        if (status!=0) return status;
        base=base_storage;
    } else {
        for (std::size_t i=0;i<m;++i) if (!std::isfinite(base[i])) return 1;
    }
    if (scheme!=1) {
        status=jvp_sample(plus,n,m,fplus,function,context,calls,callback_code);
        if (status!=0) return status;
    }
    if (scheme!=0) {
        status=jvp_sample(minus,n,m,fminus,function,context,calls,callback_code);
        if (status!=0) return status;
    }
    double wp=1.0,wm=1.0;
    if (scheme==2) {
        const double scale=std::max(a,-b), ap=a/scale, bp=b/scale;
        wp=bp/(bp-ap);wm=ap/(ap-bp);
    }
    for (std::size_t i=0;i<m;++i) {
        ObjectiveAccumulator sum;
        if (scheme!=1) jvp_secant(fplus[i],base[i],a,wp,vscale,sum);
        if (scheme!=0) jvp_secant(fminus[i],base[i],b,wm,vscale,sum);
        if (sum.value(out+i)!=0) return jvp_fail(out,m,1);
    }
    return 0;
}
}} // namespace cyroot::native
#endif
