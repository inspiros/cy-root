import numpy as np

from cyroot import *


def get_test_resources():
    F = lambda x: np.array([x[0] ** 3 - 3 * x[0] * x[1] + 5 * x[1] - 7,
                            x[0] ** 2 + x[0] * x[1] ** 2 - 4 * x[1] ** 2 + 3.5])
    J = lambda x: np.array([
        [3 * x[0] ** 2 - 3 * x[1], -3 * x[0] + 5],
        [2 * x[0] + x[1] ** 2, 2 * x[0] * x[1] - 8 * x[1]]
    ])
    H = lambda x: np.array([
        [[6 * x[0], -3],
         [-3, 0]],
        [[2, 2 * x[1]],
         [2 * x[1], 2 * x[0] - 8]],
    ])
    return F, J, H


def test_output(etol=1e-8, ptol=1e-12):
    # examples output
    print('Output Test')

    F, J, H = get_test_resources()
    x0 = np.array([10., 10.])
    J_x0 = J(x0)

    print(f'\n{"Bracketing":-^50}')
    print('[Vrahatis]', vrahatis(F, np.array([[0., 0.],
                                              [3, -1.],
                                              [2.5, -5.],
                                              [1., -2.]]),
                                 etol=etol, ptol=ptol))

    print(f'\n{"Quasi-Newton":-^50}')
    print('[Wolfe-Bittner]', wolfe_bittner(F, np.array([[2., 2.],
                                                        [4., 7.],
                                                        [-1., 0.]]),
                                           etol=etol, ptol=ptol))
    print('[Robinson]', robinson(F, np.array([2., 2.]), np.array([4., 7.]),
                                 etol=etol, ptol=ptol))
    print('[Barnes]', barnes(F, x0, J_x0=J_x0 + np.random.rand(*J_x0.shape) * 1e-4,
                             etol=etol, ptol=ptol))
    print('[Traub-Steffensen]', traub_steffensen(F, np.array([1.5, -2]), etol=etol, ptol=ptol))
    print('[Broyden Good]', broyden(F, x0, J_x0=J_x0, algo='good', etol=etol, ptol=ptol))
    print('[Broyden Bad]', broyden(F, x0, J_x0=J_x0, algo='bad', etol=etol, ptol=ptol))
    print('[Klement]', klement(F, x0, J_x0=J_x0, etol=etol, ptol=ptol))

    print(f'\n{"Newton":-^50}')
    print('[Newton]', generalized_newton(F, J, x0, etol=etol, ptol=ptol))
    print('[Halley]', generalized_halley(F, J, H, x0, etol=etol, ptol=ptol))
    print('[Super-Halley]', generalized_super_halley(F, J, H, x0, etol=etol, ptol=ptol))
    print('[Chebyshev]', generalized_chebyshev(F, J, H, x0, etol=etol, ptol=ptol))
    print('[Tangent Hyperbolas]', generalized_tangent_hyperbolas(F, J, H, x0, formula=1, etol=etol, ptol=ptol))


if __name__ == '__main__':
    test_output()
