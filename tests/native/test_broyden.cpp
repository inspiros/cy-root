#include "broyden.hpp"
#include <initializer_list>
#include <cassert>
#include <cmath>
#include <limits>

int main() {
    using cyroot::native::broyden_update;
    const double matrix[] = {1.,3.,2.,4.};
    const double direction[] = {1.,2.}, target[] = {7.,13.};
    double output[] = {123.,0.,0.,0.,0.,321.};
    assert(broyden_update(matrix,direction,target,2,output+1)==0);
    // matrix*d=(5,11), residual=(2,2), denominator5.
    const double expected[] = {1.4,3.4,2.8,4.8};
    for (int i=0;i<4;++i)assert(std::fabs(output[1+i]-expected[i])<1e-14);
    assert(output[0]==123. && output[5]==321.);
    // Squared denominator alone underflows/overflows; final update is2.
    for (int exponent : {-1000,1000}) {
        const double d=std::ldexp(1.,exponent), t=std::ldexp(1.,exponent+1), b=1.;
        assert(broyden_update(&b,&d,&t,1,output+1)==0);
        assert(output[1]==2.);
    }
    // matrix*d overflows; combined rank-one update still equals1 in scalar case.
    const double b=std::ldexp(1.,1000), d=std::ldexp(1.,1000), t=d;
    assert(broyden_update(&b,&d,&t,1,output+1)==0);
    assert(output[1]==1.);
    const double coupled[]={std::ldexp(1.,1000),std::ldexp(1.,1000),0.,0.};
    const double axis[]={std::ldexp(1.,1000),0.}, goal[]={std::ldexp(1.,1000),std::ldexp(1.,1000)};
    assert(broyden_update(coupled,axis,goal,2,output+1)==0);
    assert(output[1]==1. && output[2]==1. && output[3]==0. && output[4]==0.);
    const double three=3.,one=1.;
    assert(broyden_update(&b,&three,&one,1,output+1)==0);
    assert(output[1]==1./3.);
    const double zero=0.;
    assert(broyden_update(&b,&zero,&t,1,output+1)==3);
    assert(std::isnan(output[1]));
    output[1]=999.;
    assert(broyden_update(&b,&d,&t,0,output+1)==2 && output[1]==999.);
    assert(broyden_update(&b,&d,&t,std::numeric_limits<std::size_t>::max(),output+1)==2);
    const double bad[]={1.,3.,2.,std::numeric_limits<double>::infinity()};
    assert(broyden_update(bad,direction,target,2,output+1)==1);
    for (int i=1;i<=4;++i)assert(std::isnan(output[i]));
    const double tiny=std::numeric_limits<double>::denorm_min(), huge=std::numeric_limits<double>::max();
    assert(broyden_update(&zero,&tiny,&huge,1,output+1)==1 && std::isnan(output[1]));
    assert(output[0]==123. && output[5]==321.);
}
