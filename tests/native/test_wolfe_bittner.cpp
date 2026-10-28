#include "wolfe_bittner.hpp"
#include <cassert>
#include <initializer_list>

namespace {
void poisoned(const double* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) assert(std::isnan(p[i]));
}
}

int main() {
    using cyroot::native::wolfe_bittner_candidate;
    // Two residual coordinates, three pool points, two domain coordinates.
    // The exact affine weights are (1/2,1/4,1/4); use a non-symmetric
    // matrix so row-major interpretation cannot accidentally pass.
    const double matrix[] = {1.,1.,0., 1.,-2.,1., 1.,0.,-1.};
    const double weights[] = {.5,.25,.25};
    const double points[] = {2.,-4., 6.,8., -2.,12.};
    double output[] = {123.,999.,999.,321.};
    double errors[] = {456.,999.,654.};
    assert(wolfe_bittner_candidate(matrix,weights,points,3,3,2,
                                  output+1,errors+1)==0);
    assert(output[1]==2. && output[2]==3. && errors[1]==0.);
    // Wide affine system with a third domain coordinate and zero weight.
    const double wide[] = {1.,-1., 1.,1., 1.,4.};
    const double wide_weights[] = {.5,.5,0.};
    const double wide_points[] = {0.,2.,4., 4.,6.,8., 100.,200.,300.};
    double candidate[3];
    double error;
    assert(wolfe_bittner_candidate(wide,wide_weights,wide_points,2,3,3,
                                  candidate,&error)==0);
    assert(candidate[0]==2. && candidate[1]==4. && candidate[2]==6.);
    // Tall system, one point: valid exactly when all residual rows vanish.
    const double tall[] = {1.,0.,0.,0.}, unit[] = {1.};
    assert(wolfe_bittner_candidate(tall,unit,points,4,1,2,
                                  output+1,errors+1)==0);
    assert(output[1]==2. && output[2]==-4.);
    const double wrong[] = {.5,.5,0.};
    assert(wolfe_bittner_candidate(matrix,wrong,points,3,3,2,
                                  output+1,errors+1)==3);
    assert(errors[1]==.5); poisoned(output+1,2);
    // The stated roundoff bound, rather than exact equality, gates success.
    const double eps = std::numeric_limits<double>::epsilon();
    const double near[] = {1.,16.*eps};
    assert(wolfe_bittner_candidate(near,unit,points,2,1,2,
                                  output+1,errors+1)==0);
    assert(errors[1]==16.*eps);
    const double far[] = {1.,256.*eps};
    assert(wolfe_bittner_candidate(far,unit,points,2,1,2,
                                  output+1,errors+1)==3);
    assert(errors[1]==256.*eps); poisoned(output+1,2);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double bad : {nan,inf}) {
        const double bad_matrix[] = {1.,bad};
        const double bad_weight[] = {bad};
        const double bad_point[] = {2.,bad};
        assert(wolfe_bittner_candidate(bad_matrix,unit,points,2,1,2,
                                      output+1,errors+1)==1);
        assert(std::isnan(errors[1])); poisoned(output+1,2);
        assert(wolfe_bittner_candidate(tall,bad_weight,points,4,1,2,
                                      output+1,errors+1)==1);
        assert(std::isnan(errors[1])); poisoned(output+1,2);
        assert(wolfe_bittner_candidate(tall,unit,bad_point,4,1,2,
                                      output+1,errors+1)==1);
        assert(std::isnan(errors[1])); poisoned(output+1,2);
    }
    // A zero weight does not hide a nonfinite point coordinate.
    const double bad_unused[] = {0.,2.,4., 4.,6.,8., nan,200.,300.};
    assert(wolfe_bittner_candidate(wide,wide_weights,bad_unused,2,3,3,
                                  candidate,&error)==1);
    assert(std::isnan(error)); poisoned(candidate,3);
    const double maximum = std::numeric_limits<double>::max();
    // Finite inputs can overflow the matrix product or row norm.
    const double huge_matrix[] = {1.,maximum};
    const double two[] = {2.};
    assert(wolfe_bittner_candidate(huge_matrix,two,points,2,1,2,
                                  output+1,errors+1)==1);
    assert(std::isnan(errors[1])); poisoned(output+1,2);
    const double huge_row[] = {1.,maximum, 1.,maximum};
    const double half[] = {.5,.5};
    assert(wolfe_bittner_candidate(huge_row,half,points,2,2,2,
                                  output+1,errors+1)==1);
    assert(std::isnan(errors[1])); poisoned(output+1,2);
    // Valid affine weights may produce a nonfinite domain coordinate.
    // Fail after the first successful coordinate: the entire output is NaN.
    const double signed_matrix[] = {1.,0., 1.,0.};
    const double signed_weights[] = {2.,-1.};
    const double huge_points[] = {1.,maximum, 0.,0.};
    assert(wolfe_bittner_candidate(signed_matrix,signed_weights,huge_points,
                                  2,2,2,output+1,errors+1)==1);
    assert(errors[1]==0.); poisoned(output+1,2);
    // Zero or impossible dimensions must not dereference inputs or outputs.
    output[1]=errors[1]=999.;
    const auto size_max = std::numeric_limits<std::size_t>::max();
    const std::size_t invalid[][3] = {
        {0,1,1},{1,0,1},{1,1,0},{size_max,2,1},
        {1,size_max,1},{1,2,size_max}
    };
    for (const auto& shape : invalid)
        assert(wolfe_bittner_candidate(nullptr,nullptr,nullptr,
                                      shape[0],shape[1],shape[2],
                                      output+1,errors+1)==2);
    assert(output[1]==999. && errors[1]==999.);
    assert(output[0]==123. && output[3]==321.);
    assert(errors[0]==456. && errors[2]==654.);
}
