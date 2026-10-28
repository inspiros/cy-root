#ifndef CYROOT_DIFFERENTIATION_HPP
#define CYROOT_DIFFERENTIATION_HPP
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cyroot {
// All spans are disjoint, dimensions/products checked by the owner. Callbacks
// write all output components, return zero on success and must not throw.
using DiffRealFunction = int (*)(const double*, std::size_t, std::size_t, double*, void*);
using DiffComplexFunction = int (*)(const double*, const double*, std::size_t,
                                   std::size_t, double*, double*, void*);
enum EstimateStatus { ESTIMATE_OK, ESTIMATE_NONFINITE, ESTIMATE_BREAKDOWN,
                      ESTIMATE_CALLBACK_FAILURE, ESTIMATE_MAX_STEPS, ESTIMATE_ROUNDOFF };
struct EstimateReport { std::size_t calls; int status; int callback_code; };
inline void diff_nan(double* out, std::size_t count) noexcept {
    std::fill_n(out, count, std::numeric_limits<double>::quiet_NaN());
}
inline bool diff_finite(const double* values, std::size_t count) noexcept {
    for (std::size_t i=0; i<count; ++i) if (!std::isfinite(values[i])) return false;
    return true;
}
inline int diff_failure(EstimateReport& report, int status, double* out,
                        double* error, std::size_t count) noexcept {
    report.status=status; diff_nan(out,count);
    if (error) diff_nan(error,count);
    return status;
}
// work: 2*n+2*m doubles; no allocation, Python access or mutable global state.
inline int complex_step_jacobian(const double* x, std::size_t n, std::size_t m,
        const double* steps, double* out, double* work, DiffComplexFunction function,
        void* context, EstimateReport& report) noexcept {
    report={0,ESTIMATE_OK,0}; diff_nan(out,n*m);
    if (!diff_finite(x,n)) return diff_failure(report,ESTIMATE_NONFINITE,out,nullptr,n*m);
    double* xr=work; double* xi=xr+n; double* fr=xi+n; double* fi=fr+m;
    std::copy_n(x,n,xr); std::fill_n(xi,n,0.);
    for (std::size_t j=0; j<n; ++j) {
        const double h=steps ? steps[j] : 1e-20*std::max(1.,std::abs(x[j]));
        if (!std::isfinite(h) || h==0.) return diff_failure(report,ESTIMATE_BREAKDOWN,out,nullptr,n*m);
        xi[j]=h; diff_nan(fr,m); diff_nan(fi,m); ++report.calls;
        const int code=function(xr,xi,n,m,fr,fi,context); xi[j]=0.;
        if (code) { report.callback_code=code; return diff_failure(report,ESTIMATE_CALLBACK_FAILURE,out,nullptr,n*m); }
        if (!diff_finite(fr,m) || !diff_finite(fi,m)) return diff_failure(report,ESTIMATE_NONFINITE,out,nullptr,n*m);
        for (std::size_t i=0; i<m; ++i) out[i+j*m]=fi[i]/h;
        if (!diff_finite(out+j*m,m)) return diff_failure(report,ESTIMATE_NONFINITE,out,nullptr,n*m);
    }
    return ESTIMATE_OK;
}

// One Richardson row shared by scalar, Jacobian and Hessian estimators.
// table[levels*count],diagonal[count],previous[count] start at zero;
// error starts at infinity. Caller checks k<levels and poisons fatal output.
inline int richardson_update(const double* raw, std::size_t count, std::size_t k,
        double* table, double* diagonal, double* previous, double* best,
        double* error, int* improved) noexcept {
    std::copy_n(diagonal,count,previous); *improved=0;
    for (std::size_t i=0; i<count; ++i) {
        double value=raw[i];
        if (!std::isfinite(value)) return ESTIMATE_NONFINITE;
        double old=table[i]; table[i]=value;
        if (!k) best[i]=value;
        double factor=4.;
        for (std::size_t l=1; l<=k; ++l,factor*=4.) {
            const double parent=value, saved=table[l*count+i];
            value+=(value-old)/(factor-1.);
            const double e=std::max(std::abs(value-parent),std::abs(value-old));
            if (!std::isfinite(value) || !std::isfinite(e)) return ESTIMATE_NONFINITE;
            if (e<error[i]) { error[i]=e; best[i]=value; *improved=1; }
            old=saved; table[l*count+i]=value;
        }
        diagonal[i]=value;
    }
    return ESTIMATE_OK;
}

// Adaptive central differences with a Richardson triangle. Best estimates and
// conservative adjacent-table differences are retained component by component.
// work: n+(levels+5)*m doubles. Full output/error: m*n each, column-major.
inline int adaptive_columns(const double* x, std::size_t n, std::size_t m,
        const double* steps, double atol, double rtol, std::size_t levels,
        double* out, double* errors, double* work, DiffRealFunction function,
        void* context, EstimateReport& report, int derivative_order) noexcept {
    report={0,ESTIMATE_OK,0}; diff_nan(out,n*m); diff_nan(errors,n*m);
    if (!diff_finite(x,n)) return diff_failure(report,ESTIMATE_NONFINITE,out,errors,n*m);
    double* point=work; double* base=point+n; double* plus=base+m;
    double* minus=plus+m; double* table=minus+m; double* diagonal=table+levels*m;
    double* previous=diagonal+m;
    std::copy_n(x,n,point); diff_nan(base,m); ++report.calls;
    int code=function(point,n,m,base,context);
    if (code) { report.callback_code=code; return diff_failure(report,ESTIMATE_CALLBACK_FAILURE,out,errors,n*m); }
    if (!diff_finite(base,m)) return diff_failure(report,ESTIMATE_NONFINITE,out,errors,n*m);
    for (std::size_t j=0; j<n; ++j) {
        double h=steps ? steps[j] : .1*std::max(1.,std::abs(x[j]));
        double* best=out+j*m; double* error=errors+j*m;
        std::fill_n(error,m,std::numeric_limits<double>::infinity());
        std::fill_n(table,(levels+2)*m,0.);
        double previous_xp=std::numeric_limits<double>::quiet_NaN(), previous_xm=previous_xp;
        bool converged=false; int column_status=ESTIMATE_MAX_STEPS;
        std::size_t stalled=0;
        for (std::size_t k=0; k<levels; ++k,h*=.5) {
            const double xp=x[j]+h, xm=x[j]-h, a=xp-x[j], b=xm-x[j];
            if (!std::isfinite(xp) || !std::isfinite(xm) || !std::isfinite(a) || !std::isfinite(b))
                return diff_failure(report,ESTIMATE_NONFINITE,out,errors,n*m);
            if (a==0. || b==0. || xp==xm) {
                if (!k) return diff_failure(report,ESTIMATE_BREAKDOWN,out,errors,n*m);
                column_status=ESTIMATE_ROUNDOFF; break;
            }
            // Repeated rounded stencils are not independent accuracy evidence.
            if (k && xp==previous_xp && xm==previous_xm) { column_status=ESTIMATE_ROUNDOFF; break; }
            previous_xp=xp; previous_xm=xm;
            point[j]=xp; diff_nan(plus,m); ++report.calls;
            code=function(point,n,m,plus,context); point[j]=x[j];
            if (code) { report.callback_code=code; return diff_failure(report,ESTIMATE_CALLBACK_FAILURE,out,errors,n*m); }
            point[j]=xm; diff_nan(minus,m); ++report.calls;
            code=function(point,n,m,minus,context); point[j]=x[j];
            if (code) { report.callback_code=code; return diff_failure(report,ESTIMATE_CALLBACK_FAILURE,out,errors,n*m); }
            if (!diff_finite(plus,m) || !diff_finite(minus,m)) return diff_failure(report,ESTIMATE_NONFINITE,out,errors,n*m);
            const double scale=std::max(std::abs(a),std::abs(b));
            const double aa=a/scale, bb=b/scale;
            const double wp=bb/(bb-aa), wm=aa/(aa-bb);
            for (std::size_t i=0; i<m; ++i) {
                const double sp=(plus[i]-base[i])/a, sm=(minus[i]-base[i])/b;
                plus[i]=derivative_order==2 ? ((sp-sm)/scale)*(2./(aa-bb)) : wp*sp+wm*sm;
            }
            int improved=0;
            if (richardson_update(plus,m,k,table,diagonal,previous,best,error,&improved))
                return diff_failure(report,ESTIMATE_NONFINITE,out,errors,n*m);
            if (k) {
                stalled=improved ? 0 : stalled+1;
                converged=true; bool deteriorated=true;
                for (std::size_t i=0; i<m; ++i) {
                    if (!(error[i]<=atol+rtol*std::abs(best[i]))) converged=false;
                    if (!(std::abs(diagonal[i]-previous[i])>2.*error[i])) deteriorated=false;
                }
                if (converged) break;
                // A transient worsening before the asymptotic regime is not
                // enough evidence of roundoff; allow higher extrapolation rows.
                if (k>=4 && stalled>=2 && deteriorated) { column_status=ESTIMATE_ROUNDOFF; break; }
            }
        }
        if (!converged && report.status==ESTIMATE_OK) report.status=column_status;
    }
    return report.status;
}

inline int adaptive_jacobian(const double* x, std::size_t n, std::size_t m,
        const double* steps, double atol, double rtol, std::size_t levels,
        double* out, double* errors, double* work, DiffRealFunction function,
        void* context, EstimateReport& report) noexcept {
    return adaptive_columns(x,n,m,steps,atol,rtol,levels,out,errors,work,
                            function,context,report,1);
}

// Scalar second derivative. Same bounded Richardson table/work contract as
// n=m=1 above; actual rounded offsets define the quadratic interpolant.
inline int adaptive_second_derivative(double x, const double* step, double atol,
        double rtol, std::size_t levels, double* out, double* error, double* work,
        DiffRealFunction function, void* context, EstimateReport& report) noexcept {
    return adaptive_columns(&x,1,1,step,atol,rtol,levels,out,error,work,
                             function,context,report,2);
}
} // namespace cyroot
#endif
