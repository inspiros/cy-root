#include "robinson.hpp"
#include <cassert>
#include <initializer_list>

namespace {
bool close(double a, double b) { return std::fabs(a-b) < 3e-14; }
void poisoned(const double* p, std::size_t n) {
    for (std::size_t k=0; k<n; ++k) assert(std::isnan(p[k]));
}
}

int main() {
    using cyroot::native::robinson_nodes;
    double nodes[11], transpose[11], scratch[5];
    nodes[0]=transpose[0]=scratch[0]=123.;
    nodes[10]=transpose[10]=scratch[4]=321.;
    // Rational direction (2,3,6)/7: recover the full orthogonal basis.
    const double current[]={0.,0.,0.}, previous[]={-2.,-3.,-6.};
    assert(robinson_nodes(previous,current,3,false,nodes+1,transpose+1,scratch+1)==0);
    for (int i=0; i<3; ++i) assert(close(nodes[1+i],previous[i]));
    for (int j=0; j<3; ++j)
        for (int k=0; k<3; ++k) {
            double dot=0.;
            for (int i=0; i<3; ++i) {
                dot+=nodes[1+i+3*j]*nodes[1+i+3*k]/49.;
                assert(transpose[1+j+3*i]==nodes[1+i+3*j]);
            }
            assert(close(dot,j==k?1.:0.));
        }
    assert(robinson_nodes(previous,current,3,true,nodes+1,transpose+1,scratch+1)==0);
    for (int j=0; j<3; ++j)
        for (int i=0; i<3; ++i) assert(nodes[1+i+3*j]==(i==j?7.:0.));
    // Rounded displacement is nonsymmetric even though Householder Q is
    // symmetric. The solver needs S^T, not Q, nor the unrounded radius*Q.
    const double big=std::ldexp(1.,55);
    const double rounded_current[]={big,0.}, rounded_previous[]={big-4.,-1.};
    assert(robinson_nodes(rounded_previous,rounded_current,2,false,nodes+1,transpose+1,scratch+1)==0);
    assert(transpose[2]!=transpose[3]);
    for (int j=0; j<2; ++j)
        for (int i=0; i<2; ++i)
            assert(transpose[1+j+2*i]==nodes[1+i+2*j]-rounded_current[i]);
    // Finite norms when naive squaring would overflow or underflow.
    for (int exponent : {-1000,1000}) {
        const double scale=std::ldexp(1.,exponent);
        const double before[]={-3.*scale,-4.*scale}, now[]={0.,0.};
        assert(robinson_nodes(before,now,2,false,nodes+1,transpose+1,scratch+1)==0);
        assert(close(nodes[1]/scale,-3.) && close(nodes[2]/scale,-4.));
        assert(close(nodes[3]/scale,-4.) && close(nodes[4]/scale,3.));
    }
    // Negative and zero leading coordinates select stable signed reflectors.
    for (double leading : {-3.,0.}) {
        const double before[]={-leading,-4.}, now[]={0.,0.};
        assert(robinson_nodes(before,now,2,false,nodes+1,transpose+1,scratch+1)==0);
        const double sign=leading<0.?-1.:1.;
        assert(close(nodes[1],-sign*leading) && close(nodes[2],-sign*4.));
    }
    const double zero[]={0.,0.}, one[]={1.,0.};
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    const double bad_nan[]={0.,nan}, bad_inf[]={0.,inf};
    assert(robinson_nodes(zero,zero,2,false,nodes+1,transpose+1,scratch+1)==3);
    poisoned(nodes+1,4); poisoned(transpose+1,4);
    for (const double* bad : {bad_nan,bad_inf}) {
        assert(robinson_nodes(zero,bad,2,false,nodes+1,transpose+1,scratch+1)==1);
        poisoned(nodes+1,4); poisoned(transpose+1,4);
    }
    const double maximum=std::numeric_limits<double>::max();
    const double low[]={-maximum,0.}, high[]={maximum,0.};
    assert(robinson_nodes(low,high,2,false,nodes+1,transpose+1,scratch+1)==1);
    const double large[]={maximum,maximum};
    assert(robinson_nodes(zero,large,2,false,nodes+1,transpose+1,scratch+1)==1);
    // A later node overflow poisons the earlier successfully written column.
    const double near_limit[]={0.,maximum}, near_before[]={-maximum,maximum};
    assert(robinson_nodes(near_before,near_limit,2,true,nodes+1,transpose+1,scratch+1)==1);
    poisoned(nodes+1,4); poisoned(transpose+1,4);
    const double collapsed_current[]={big,0.}, collapsed_previous[]={big,-1.};
    assert(robinson_nodes(collapsed_previous,collapsed_current,2,true,nodes+1,transpose+1,scratch+1)==3);
    poisoned(nodes+1,4); poisoned(transpose+1,4);
    assert(robinson_nodes(zero,one,1,false,nodes+1,transpose+1,scratch+1)==0);
    assert(nodes[1]==0. && transpose[1]==-1.);
    nodes[1]=transpose[1]=scratch[1]=999.;
    for (std::size_t n : {std::size_t(0),std::numeric_limits<std::size_t>::max()})
        assert(robinson_nodes(nullptr,nullptr,n,false,nodes+1,transpose+1,scratch+1)==2);
    assert(nodes[1]==999. && transpose[1]==999. && scratch[1]==999.);
    assert(nodes[0]==123. && transpose[0]==123. && scratch[0]==123.);
    assert(nodes[10]==321. && transpose[10]==321. && scratch[4]==321.);
}
