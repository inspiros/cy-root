import numpy as np

from cyroot import *

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


def test_output(etol=1e-8, ptol=1e-12):
    # examples output
    print('Output Test')

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
    print('[Robinson]', robinson(F, x0=[2., 2.], x1=[4., 7.], etol=etol, ptol=ptol))
    print('[Barnes]', barnes(F, [10., 10.], etol=etol, ptol=ptol))
    print('[Traub-Steffensen]', traub_steffensen(F, [4., -2.], etol=etol, ptol=ptol))
    print('[Broyden Good]', broyden(F, [10., 10.], algo='good', etol=etol, ptol=ptol))
    print('[Broyden Bad]', broyden(F, [10., 10.], algo='bad', etol=etol, ptol=ptol))
    print('[Klement]', klement(F, [10., 10.], etol=etol, ptol=ptol))

    print(f'\n{"Newton":-^50}')
    print('[Newton]', generalized_newton(F, J, [10., 10.], etol=etol, ptol=ptol))
    print('[Halley]', generalized_halley(F, J, H, [10., 10.], etol=etol, ptol=ptol))
    print('[Super-Halley]', generalized_super_halley(F, J, H, [10., 10.], etol=etol, ptol=ptol))
    print('[Chebyshev]', generalized_chebyshev(F, J, H, [10., 10.], etol=etol, ptol=ptol))
    print('[Tangent Hyperbolas]', generalized_tangent_hyperbolas(F, J, H, x0=[10., 10.], etol=etol, ptol=ptol))


if __name__ == '__main__':
    test_output()
