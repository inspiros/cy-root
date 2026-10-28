#include "broyden.hpp"
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <limits>

int main() {
    using cyroot::native::klement_update;
    const double matrix[] = {1.,3.,2.,4.};
    const double direction[] = {1.,2.}, target[] = {7.,13.};
    double out[] = {123.,0.,0.,0.,0.,321.};
    assert(klement_update(matrix,direction,target,2,out+1)==0);
    // Independent rational rows: u=(1,4),(3,8); D=17,73; residual=2.
    const double expected[] = {19./17.,237./73.,50./17.,356./73.};
    for (int i=0;i<4;++i) assert(std::fabs(out[i+1]-expected[i])<1e-14);
    for (int i=0;i<2;++i)
        assert(std::fabs(out[1+i]*direction[0]+out[3+i]*direction[1]-target[i])<1e-14);
    // u and its square exceed binary64 range in BOTH directions; retaining
    // the low residual component must expose the unit target after cancelling B.
    for (int exponent : {-1000,1000}) {
        const double b=std::ldexp(1.,exponent);
        const double diagonal[]={b,0.,0.,b}, step[]={b,b}, goal[]={b,b};
        assert(klement_update(diagonal,step,goal,2,out+1)==0);
        assert(out[1]==1. && out[2]==0. && out[3]==0. && out[4]==1.);
    }
    const double b=std::ldexp(1.,1000), three=3., one=1., zero=0.;
    assert(klement_update(&b,&three,&one,1,out+1)==0 && out[1]==1./3.);
    assert(klement_update(&zero,&one,&one,1,out+1)==3 && std::isnan(out[1]));
    assert(klement_update(&one,&zero,&one,1,out+1)==3 && std::isnan(out[1]));
    const double late_zero[]={1.,0.,2.,0.};
    assert(klement_update(late_zero,direction,target,2,out+1)==3);
    for (int i=1;i<=4;++i) assert(std::isnan(out[i]));
    const double bad[]={1.,3.,2.,std::numeric_limits<double>::infinity()};
    assert(klement_update(bad,direction,target,2,out+1)==1);
    for (int i=1;i<=4;++i) assert(std::isnan(out[i]));
    const double tiny=std::numeric_limits<double>::denorm_min();
    const double identity[]={1.,0.,0.,1.}, step[]={1.,tiny};
    const double large_goal[]={1.,std::numeric_limits<double>::max()};
    assert(klement_update(identity,step,large_goal,2,out+1)==1);
    for (int i=1;i<=4;++i) assert(std::isnan(out[i]));
    out[1]=999.;
    assert(klement_update(&one,&one,&one,0,out+1)==2 && out[1]==999.);
    assert(klement_update(&one,&one,&one,std::numeric_limits<std::size_t>::max(),out+1)==2);
    assert(out[0]==123. && out[5]==321.);
}
