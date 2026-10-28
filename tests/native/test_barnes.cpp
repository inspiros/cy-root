#include "barnes.hpp"
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace {
bool close(double a, double b) { return std::fabs(a-b) < 2e-13; }
void poisoned(const double* p, std::size_t n) {
    for (std::size_t k=0; k<n; ++k) assert(std::isnan(p[k]));
}
}

int main() {
    using cyroot::native::barnes_direction;
    using cyroot::native::barnes_update;
    const double inf=std::numeric_limits<double>::infinity();
    const double nan=std::numeric_limits<double>::quiet_NaN();
    double basis[14], row[6], z[6], out[18];
    for (double* p : {basis,row,z,out}) p[0]=123.;
    basis[13]=row[5]=z[5]=out[17]=321.;

    // Rational projection: (2,0,1) - ((2,0,1).(1,1,0))/2*(1,1,0).
    const double history[]={1.,1.,0.}, step[]={2.,0.,1.};
    assert(barnes_direction(history,1,step,3,basis+1,row+1,z+1)==0);
    const double unit=1./std::sqrt(3.);
    assert(close(z[1],unit) && close(z[2],-unit) && close(z[3],unit));
    const double matrix[]={1.,4.,7.,2.,5.,8.,3.,6.,10.};
    const double target[]={10.,20.,30.};
    assert(barnes_update(matrix,step,target,z+1,3,out+1)==0);
    const double residual[]={5.,6.,6.};
    for (int j=0; j<3; ++j)
        for (int i=0; i<3; ++i)
            assert(close(out[1+i+3*j],matrix[i+3*j]+residual[i]*(j==1?-1.:1.)/3.));
    for (int i=0; i<3; ++i) {
        assert(close(2.*out[1+i]+out[1+i+6],target[i]));
        assert(close(out[1+i]+out[1+i+3],matrix[i]+matrix[i+3]));
    }

    // Zero and rank-dependent history rows leave the same single constraint.
    const double dependent[]={0.,0.,0.,0., 1.,2.,0.,0., 2.,4.,0.,0.};
    const double d4[]={2.,0.,1.,0.};
    assert(barnes_direction(dependent,3,d4,4,basis+1,row+1,z+1)==0);
    const double length=std::sqrt(4.2);
    assert(close(z[1],1.6/length) && close(z[2],-.8/length));
    assert(close(z[3],1./length) && z[4]==0.);
    // Independent rows actually span two constraints, with a known cross
    // product (1,-1,-1); projection orientation agrees with the current step.
    const double independent[]={1.,1.,0., 1.,0.,1.};
    const double axis3[]={1.,0.,0.};
    assert(barnes_direction(independent,2,axis3,3,basis+1,row+1,z+1)==0);
    assert(close(z[1],unit) && close(z[2],-unit) && close(z[3],-unit));
    const double collinear[]={3.,3.,0.};
    assert(barnes_direction(history,1,collinear,3,basis+1,row+1,z+1)==3);
    poisoned(z+1,3);
    const double nearly[]={1.,1.,1e-15};
    assert(barnes_direction(history,1,nearly,3,basis+1,row+1,z+1)==3);
    poisoned(z+1,3);
    const double resolved[]={1.,1.,1e-10};
    assert(barnes_direction(history,1,resolved,3,basis+1,row+1,z+1)==0);
    assert(close(z[3],1.));

    for (int exponent : {-1000,1000}) {
        const double scale=std::ldexp(1.,exponent);
        const double h[]={scale,scale,0.}, s[]={scale,0.,scale};
        assert(barnes_direction(h,1,s,3,basis+1,row+1,z+1)==0);
        const double norm=std::sqrt(1.5);
        assert(close(z[1],.5/norm) && close(z[2],-.5/norm) && close(z[3],1./norm));
        // Exponent-scaled denominator must not underflow or overflow.
        const double b[]={1.,0.,0.,1.}, d[]={scale,0.}, t[]={2.*scale,0.};
        const double direction[]={scale,0.};
        assert(barnes_update(b,d,t,direction,2,out+1)==0);
        assert(out[1]==2. && out[2]==0. && out[3]==0. && out[4]==1.);
    }
    // Both residual components must survive the huge matrix cancellation.
    const double huge=std::ldexp(1.,1000);
    const double coupled[]={huge,huge,0.,0.}, d[]={huge,0.}, t[]={huge,huge}, q[]={1.,0.};
    assert(barnes_update(coupled,d,t,q,2,out+1)==0);
    assert(out[1]==1. && out[2]==1. && out[3]==0. && out[4]==0.);
    const double scalar_b=huge, three=3., one=1., zero=0.;
    assert(barnes_update(&scalar_b,&three,&one,&one,1,out+1)==0 && out[1]==1./3.);
    assert(barnes_update(&scalar_b,&three,&one,&zero,1,out+1)==3);
    poisoned(out+1,1);
    assert(barnes_direction(nullptr,0,&three,1,nullptr,row+1,z+1)==0 && z[1]==1.);
    const double negative=-3.;
    assert(barnes_direction(nullptr,0,&negative,1,nullptr,row+1,z+1)==0 && z[1]==-1.);

    // Full finite-input preflight precedes zero/dependent early exits.
    const double zero_step[]={0.,0.,0.}, bad_history[]={0.,0.,nan};
    assert(barnes_direction(bad_history,1,zero_step,3,basis+1,row+1,z+1)==1);
    poisoned(z+1,3);
    const double bad_step[]={0.,0.,inf};
    assert(barnes_direction(history,1,bad_step,3,basis+1,row+1,z+1)==1);
    poisoned(z+1,3);
    assert(barnes_direction(history,1,zero_step,3,basis+1,row+1,z+1)==3);
    const double identity[]={1.,0.,0.,1.}, s2[]={1.,0.}, orthogonal[]={0.,1.}, t2[]={1.,2.};
    assert(barnes_update(identity,s2,t2,orthogonal,2,out+1)==3);
    poisoned(out+1,4);
    const double bad_matrix[]={1.,0.,0.,inf};
    assert(barnes_update(bad_matrix,s2,t2,orthogonal,2,out+1)==1);
    poisoned(out+1,4);
    const double bad_target[]={1.,nan}, bad_z[]={1.,inf};
    assert(barnes_update(identity,s2,bad_target,q,2,out+1)==1);
    poisoned(out+1,4);
    assert(barnes_update(identity,s2,t2,bad_z,2,out+1)==1);
    poisoned(out+1,4);
    // Final overflow after an earlier finite row still poisons the whole output.
    const double tiny=std::numeric_limits<double>::denorm_min();
    const double tiny_step[]={tiny,0.}, overflow_target[]={0.,1.};
    assert(barnes_update(identity,tiny_step,overflow_target,q,2,out+1)==1);
    poisoned(out+1,4);

    z[1]=out[1]=999.;
    const auto invalid=std::numeric_limits<std::size_t>::max();
    assert(barnes_direction(nullptr,0,nullptr,0,nullptr,nullptr,z+1)==2);
    assert(barnes_direction(nullptr,0,nullptr,invalid,nullptr,nullptr,z+1)==2);
    assert(barnes_direction(nullptr,3,nullptr,3,nullptr,nullptr,z+1)==2);
    assert(barnes_update(nullptr,nullptr,nullptr,nullptr,0,out+1)==2);
    assert(barnes_update(nullptr,nullptr,nullptr,nullptr,invalid,out+1)==2);
    assert(z[1]==999. && out[1]==999.);
    for (double* p : {basis,row,z,out}) assert(p[0]==123.);
    assert(basis[13]==321. && row[5]==321. && z[5]==321. && out[17]==321.);
}
