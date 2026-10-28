#include "tensor_ops.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>

int main() {
    using cyroot::native::contract_hessian;
    // Analytically expand two asymmetric quadratic forms: input symmetry is
    // not presumed, so H01 and H10 must each contribute their own coefficient.
    const double h[] = {2., 11., 5., 17., 3., 13., 7., 19.};
    const double direction[] = {2., -3.};
    double guarded[] = {123., 0., 0., 456.};
    assert(contract_hessian(h, direction, 2, guarded + 1) == 0);
    assert(guarded[1] == 23. && guarded[2] == 35.);
    assert(guarded[0] == 123. && guarded[3] == 456.);

    for (const int sign : {-1, 1}) {
        const double scalar_h[] = {std::ldexp(1., -sign * 1000)};
        const double scalar_direction[] = {std::ldexp(1., sign * 700)};
        double value = 0.;
        assert(contract_hessian(scalar_h, scalar_direction, 1, &value) == 0);
        assert(value == std::ldexp(1., sign * 400));
    }
    const double zero[] = {0.};
    const double huge[] = {std::numeric_limits<double>::max()};
    double scalar = 42.;
    assert(contract_hessian(zero, huge, 1, &scalar) == 0 && scalar == 0.);
    const double minimum[] = {std::numeric_limits<double>::denorm_min()};
    const double one[] = {1.};
    assert(contract_hessian(minimum, one, 1, &scalar) == 0);
    assert(scalar == minimum[0]);

    // Exact cancellation of two individually unrepresentable products leaves
    // a finite quadratic form. No product is rounded to infinity beforehand.
    const double cancellation[] = {
        std::ldexp(1., 900), 0., -std::ldexp(1., 900), 0.,
        0., 0., std::ldexp(1., -400), 0.};
    const double big_direction[] = {std::ldexp(1., 200), std::ldexp(1., 200)};
    assert(contract_hessian(cancellation, big_direction, 2, guarded + 1) == 0);
    assert(guarded[1] == 1. && guarded[2] == 0.);

    double bad[8] = {0., 0., 0., 0., 0., 0., 0.,
                     std::numeric_limits<double>::infinity()};
    assert(contract_hessian(bad, direction, 2, guarded + 1) == 1);
    assert(std::isnan(guarded[1]) && std::isnan(guarded[2]));
    assert(contract_hessian(one, huge, 1, &scalar) == 1 && std::isnan(scalar));
    const double invalid_direction[] = {std::numeric_limits<double>::quiet_NaN()};
    assert(contract_hessian(one, invalid_direction, 1, &scalar) == 1 && std::isnan(scalar));

    scalar = 42.;
    assert(contract_hessian(nullptr, nullptr, 0, &scalar) == 2 && scalar == 42.);
    assert(contract_hessian(nullptr, nullptr, std::numeric_limits<std::size_t>::max(),
                            &scalar) == 2 && scalar == 42.);
    assert(guarded[0] == 123. && guarded[3] == 456.);
    using cyroot::native::contract_hessian_direction;
    using cyroot::native::negative_half_sum;
    double matrix_guarded[] = {123., 0., 0., 0., 0., 456.};
    assert(contract_hessian_direction(h, direction, 2, matrix_guarded + 1) == 0);
    assert(matrix_guarded[1] == -11. && matrix_guarded[2] == -29.);
    assert(matrix_guarded[3] == -15. && matrix_guarded[4] == -31.);
    assert(matrix_guarded[0] == 123. && matrix_guarded[5] == 456.);

    const double directional_cancellation[] = {
        std::ldexp(1., 900), 0., -std::ldexp(1., 900), 0.,
        std::ldexp(1., -400), 0., 0., 0.};
    assert(contract_hessian_direction(directional_cancellation, big_direction, 2,
                                       matrix_guarded + 1) == 0);
    assert(matrix_guarded[1] == 0. && matrix_guarded[2] == 0.);
    assert(matrix_guarded[3] == std::ldexp(1., -200) && matrix_guarded[4] == 0.);
    const double rescued_direction[] = {std::ldexp(1., -700)};
    const double huge_hessian[] = {std::ldexp(1., 1000)};
    assert(contract_hessian_direction(huge_hessian, rescued_direction, 1, &scalar) == 0);
    assert(scalar == std::ldexp(1., 300));
    assert(contract_hessian_direction(minimum, one, 1, &scalar) == 0);
    assert(scalar == minimum[0]);
    assert(contract_hessian_direction(bad, direction, 2, matrix_guarded + 1) == 1);
    for (int i = 1; i <= 4; ++i) assert(std::isnan(matrix_guarded[i]));
    scalar = 42.;
    assert(contract_hessian_direction(nullptr, nullptr, 0, &scalar) == 2 && scalar == 42.);
    assert(contract_hessian_direction(nullptr, nullptr,
                                       std::numeric_limits<std::size_t>::max(),
                                       &scalar) == 2 && scalar == 42.);
    assert(matrix_guarded[0] == 123. && matrix_guarded[5] == 456.);

    const double maximum = std::numeric_limits<double>::max();
    const double tiny = std::numeric_limits<double>::denorm_min();
    assert(negative_half_sum(maximum, maximum) == -maximum);
    assert(negative_half_sum(-maximum, -maximum) == maximum);
    assert(negative_half_sum(tiny, tiny) == -tiny);
    assert(negative_half_sum(tiny, 2. * tiny) == -2. * tiny);
    assert(negative_half_sum(maximum, -maximum) == 0.);
    assert(std::isnan(negative_half_sum(maximum, std::numeric_limits<double>::infinity())));



    using cyroot::native::matrix_vector;
    const double asymmetric_matrix[] = {2., 11., 5., 17.};
    assert(matrix_vector(asymmetric_matrix, direction, 2, guarded + 1) == 0);
    assert(guarded[1] == -11. && guarded[2] == -29.);
    const double cancel_matrix[] = {std::ldexp(1., 900), std::ldexp(1., -400),
                                     -std::ldexp(1., 900), 0.};
    assert(matrix_vector(cancel_matrix, big_direction, 2, guarded + 1) == 0);
    assert(guarded[1] == 0. && guarded[2] == std::ldexp(1., -200));
    assert(matrix_vector(huge_hessian, rescued_direction, 1, &scalar) == 0);
    assert(scalar == std::ldexp(1., 300));
    assert(matrix_vector(minimum, one, 1, &scalar) == 0 && scalar == minimum[0]);
    const double bad_matrix[] = {1., 0., 0., std::numeric_limits<double>::infinity()};
    const double zero_direction[] = {0., 0.};
    assert(matrix_vector(bad_matrix, zero_direction, 2, guarded + 1) == 1);
    assert(std::isnan(guarded[1]) && std::isnan(guarded[2]));
    assert(matrix_vector(one, huge, 1, &scalar) == 0 && scalar == huge[0]);
    assert(matrix_vector(huge, huge, 1, &scalar) == 1 && std::isnan(scalar));
    scalar = 42.;
    assert(matrix_vector(nullptr, nullptr, 0, &scalar) == 2 && scalar == 42.);
    assert(matrix_vector(nullptr, nullptr, std::numeric_limits<std::size_t>::max(),
                          &scalar) == 2 && scalar == 42.);
    assert(guarded[0] == 123. && guarded[3] == 456.);

    using cyroot::native::contract_rectangular_hessian;
    using cyroot::native::contract_rectangular_hessian_direction;
    using cyroot::native::rectangular_matrix_vector;
    // Tall: three unrelated asymmetric quadratic forms in two inputs.
    const double tall_h[] = {2., 11., 23., 5., 17., 31.,
                             3., 13., 29., 7., 19., 37.};
    double tall_result[] = {123., 0., 0., 0., 456.};
    double tall_matrix[] = {123., 0., 0., 0., 0., 0., 0., 456.};
    assert(contract_rectangular_hessian(tall_h, direction, 2, 3,
                                        tall_result + 1) == 0);
    assert(tall_result[1] == 23. && tall_result[2] == 35. && tall_result[3] == 65.);
    assert(contract_rectangular_hessian_direction(tall_h, direction, 2, 3,
                                                  tall_matrix + 1) == 0);
    const double tall_expected[] = {-11., -29., -47., -15., -31., -53.};
    for (std::size_t i = 0; i < 6; ++i) assert(tall_matrix[i + 1] == tall_expected[i]);
    assert(rectangular_matrix_vector(tall_matrix + 1, direction, 2, 3,
                                      tall_result + 1) == 0);
    assert(tall_result[1] == 23. && tall_result[2] == 35. && tall_result[3] == 65.);

    // Wide: two asymmetric forms in three inputs. H[j,k] and H[k,j] differ.
    const double wide_h[] = {1., 11., 4., 14., 7., 17.,
                             2., 12., 5., 15., 8., 18.,
                             3., 13., 6., 16., 10., 21.};
    const double wide_direction[] = {2., -3., 4.};
    assert(contract_rectangular_hessian(wide_h, wide_direction, 3, 2,
                                        guarded + 1) == 0);
    assert(guarded[1] == 85. && guarded[2] == 191.);
    assert(contract_rectangular_hessian_direction(wide_h, wide_direction, 3, 2,
                                                  tall_matrix + 1) == 0);
    const double wide_expected[] = {18., 48., 21., 51., 28., 62.};
    for (std::size_t i = 0; i < 6; ++i) assert(tall_matrix[i + 1] == wide_expected[i]);
    assert(rectangular_matrix_vector(tall_matrix + 1, wide_direction, 3, 2,
                                      guarded + 1) == 0);
    assert(guarded[1] == 85. && guarded[2] == 191.);

    // All square entry points retain bit-for-bit arithmetic/order.
    double rectangular_square[4], square_result[4];
    assert(contract_hessian(h, direction, 2, square_result) == 0);
    assert(contract_rectangular_hessian(h, direction, 2, 2, rectangular_square) == 0);
    for (std::size_t i = 0; i < 2; ++i) assert(square_result[i] == rectangular_square[i]);
    assert(contract_hessian_direction(h, direction, 2, square_result) == 0);
    assert(contract_rectangular_hessian_direction(h, direction, 2, 2,
                                                  rectangular_square) == 0);
    for (std::size_t i = 0; i < 4; ++i) assert(square_result[i] == rectangular_square[i]);
    assert(matrix_vector(asymmetric_matrix, direction, 2, square_result) == 0);
    assert(rectangular_matrix_vector(asymmetric_matrix, direction, 2, 2,
                                      rectangular_square) == 0);
    for (std::size_t i = 0; i < 2; ++i) assert(square_result[i] == rectangular_square[i]);

    // Rectangular range rescue, subnormal retention and cancellation.
    for (const int sign : {-1, 1}) {
        const double range_h[] = {std::ldexp(1., -sign * 1000),
                                  -std::ldexp(1., -sign * 1000)};
        const double range_s[] = {std::ldexp(1., sign * 700)};
        assert(contract_rectangular_hessian(range_h, range_s, 1, 2, guarded + 1) == 0);
        assert(guarded[1] == std::ldexp(1., sign * 400));
        assert(guarded[2] == -std::ldexp(1., sign * 400));
        assert(contract_rectangular_hessian_direction(range_h, range_s, 1, 2,
                                                      guarded + 1) == 0);
        assert(guarded[1] == std::ldexp(1., -sign * 300));
        assert(guarded[2] == -std::ldexp(1., -sign * 300));
        assert(rectangular_matrix_vector(range_h, range_s, 1, 2, guarded + 1) == 0);
        assert(guarded[1] == std::ldexp(1., -sign * 300));
        assert(guarded[2] == -std::ldexp(1., -sign * 300));
    }
    const double tiny_pair[] = {tiny, -tiny};
    assert(contract_rectangular_hessian(tiny_pair, one, 1, 2, guarded + 1) == 0);
    assert(guarded[1] == tiny && guarded[2] == -tiny);
    assert(contract_rectangular_hessian_direction(tiny_pair, one, 1, 2,
                                                  guarded + 1) == 0);
    assert(guarded[1] == tiny && guarded[2] == -tiny);
    assert(rectangular_matrix_vector(tiny_pair, one, 1, 2, guarded + 1) == 0);
    assert(guarded[1] == tiny && guarded[2] == -tiny);
    const double wide_cancellation[] = {std::ldexp(1., 900), 0.,
                                        -std::ldexp(1., 900), 0.,
                                        std::ldexp(1., -400), 0.};
    const double three_big[] = {std::ldexp(1., 200), std::ldexp(1., 200),
                                std::ldexp(1., 200)};
    assert(rectangular_matrix_vector(wide_cancellation, three_big, 3, 2,
                                      guarded + 1) == 0);
    assert(guarded[1] == std::ldexp(1., -200) && guarded[2] == 0.);

    // Full poisoning even after earlier outputs were written, and even for
    // nonfinite coefficients multiplied by a zero direction.
    double bad_tall[12];
    for (std::size_t i = 0; i < 12; ++i) bad_tall[i] = tall_h[i];
    bad_tall[11] = std::numeric_limits<double>::infinity();
    assert(contract_rectangular_hessian(bad_tall, zero_direction, 2, 3,
                                        tall_result + 1) == 1);
    for (std::size_t i = 1; i <= 3; ++i) assert(std::isnan(tall_result[i]));
    assert(contract_rectangular_hessian_direction(bad_tall, zero_direction, 2, 3,
                                                  tall_matrix + 1) == 1);
    for (std::size_t i = 1; i <= 6; ++i) assert(std::isnan(tall_matrix[i]));
    const double bad_tall_matrix[] = {1., 2., 3., 4., 5.,
                                      std::numeric_limits<double>::quiet_NaN()};
    assert(rectangular_matrix_vector(bad_tall_matrix, zero_direction, 2, 3,
                                      tall_result + 1) == 1);
    for (std::size_t i = 1; i <= 3; ++i) assert(std::isnan(tall_result[i]));
    const double overflow_pair[] = {1., maximum};
    assert(contract_rectangular_hessian(overflow_pair, huge, 1, 2, guarded + 1) == 1);
    assert(std::isnan(guarded[1]) && std::isnan(guarded[2]));
    assert(contract_rectangular_hessian_direction(overflow_pair, huge, 1, 2,
                                                  guarded + 1) == 1);
    assert(std::isnan(guarded[1]) && std::isnan(guarded[2]));
    assert(rectangular_matrix_vector(overflow_pair, huge, 1, 2, guarded + 1) == 1);
    assert(std::isnan(guarded[1]) && std::isnan(guarded[2]));
    assert(contract_rectangular_hessian(tiny_pair, invalid_direction, 1, 2,
                                        guarded + 1) == 1);
    assert(std::isnan(guarded[1]) && std::isnan(guarded[2]));
    assert(contract_rectangular_hessian_direction(tiny_pair, invalid_direction, 1, 2,
                                                  guarded + 1) == 1);
    assert(std::isnan(guarded[1]) && std::isnan(guarded[2]));
    assert(rectangular_matrix_vector(tiny_pair, invalid_direction, 1, 2,
                                      guarded + 1) == 1);
    assert(std::isnan(guarded[1]) && std::isnan(guarded[2]));

    // Dimension validation must precede reads/writes and reject byte spans,
    // not just overflow of size_t element counts. H can overflow when A fits.
    const std::size_t byte_limit = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    const std::size_t invalid[][2] = {
        {0, 2}, {2, 0}, {0, 0},
        {std::numeric_limits<std::size_t>::max(), 1},
        {1, std::numeric_limits<std::size_t>::max()},
        {byte_limit, 2}, {2, byte_limit}, {byte_limit + 1, 1},
        {1, byte_limit + 1}};
    for (const auto& dimensions : invalid) {
        scalar = 42.;
        const auto n = dimensions[0], m = dimensions[1];
        assert(contract_rectangular_hessian(nullptr, nullptr, n, m, &scalar) == 2);
        assert(scalar == 42.);
        assert(contract_rectangular_hessian_direction(nullptr, nullptr, n, m,
                                                      &scalar) == 2);
        assert(scalar == 42.);
        assert(rectangular_matrix_vector(nullptr, nullptr, n, m, &scalar) == 2);
        assert(scalar == 42.);
    }
    assert(contract_rectangular_hessian(nullptr, nullptr, byte_limit, 1,
                                        &scalar) == 2 && scalar == 42.);
    assert(contract_rectangular_hessian_direction(nullptr, nullptr, byte_limit, 1,
                                                  &scalar) == 2 && scalar == 42.);
    assert(tall_result[0] == 123. && tall_result[4] == 456.);
    assert(tall_matrix[0] == 123. && tall_matrix[7] == 456.);
    assert(guarded[0] == 123. && guarded[3] == 456.);

}
