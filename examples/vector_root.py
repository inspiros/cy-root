import numpy as np

from cyroot import *
from utils import timeit


def get_test_resources():
    # f = lambda x: np.array([x[0] + 2 * x[1] - 2, x[0] ** 2 + 4 * x[1] ** 2 - 4])
    # J = lambda x: np.array([
    #     [1, 2],
    #     [2 * x[0], 8 * x[1]]
    # ])
    f = lambda x: np.array([x[0] ** 3 - 3 * x[0] * x[1] + 5 * x[1] - 7,
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
    # H = lambda x: np.array([
    #     [6 * x[0], 0],
    #     [2, 2 * x[0] - 8],
    # ])
    return f, J, H


def test_output(etol=1e-8, ptol=1e-10):
    # examples output
    print('Output Test')

    f, J, H = get_test_resources()
    x0 = np.array([10., 10.])
    J_x0 = J(x0)
    print('x0', x0)
    print('J_x0', J_x0)
    print('H_x0', H(x0))

    print(f'\n{"Quasi-Newton":-^50}')
    print('[Secant]', generalized_secant(f, np.array([[2., 2.],
                                                      [4., 7.],
                                                      [-1., 0.]]),
                                         etol=etol, ptol=ptol))
    print('[Broyden Good]', broyden(f, x0, J_x0=J_x0, algo='good', etol=etol, ptol=ptol))
    print('[Broyden Bad]', broyden(f, x0, J_x0=J_x0, algo='bad', etol=etol, ptol=ptol))
    print('[Klement]', klement(f, x0, J_x0=J_x0, etol=etol, ptol=ptol))

    print(f'\n{"Newton":-^50}')
    print('[Newton]', generalized_newton(f, J, x0, etol=etol, ptol=ptol))
    print('[Halley]', generalized_halley(f, J, H, x0, etol=etol, ptol=ptol))
    print('[Tangent Hyperbolas]', generalized_tangent_hyperbolas(f, J, H, x0, formula=1, etol=etol, ptol=ptol))


def test_speed(etol=1e-8, ptol=1e-10, times=100):
    # examples speed
    print('Speed Test')

    f, J, H = get_test_resources()
    x0 = np.array([10., 10.])
    J_x0 = J(x0)

    print(f'\n{"Quasi-Newton":-^50}')
    timeit(broyden, args=(f, x0), kwargs=dict(J_x0=J_x0, algo='good', etol=etol, ptol=ptol),
           name='Broyden Good', number=times)
    timeit(broyden, args=(f, x0), kwargs=dict(J_x0=J_x0, algo='bad', etol=etol, ptol=ptol),
           name='Broyden Bad', number=times)
    timeit(klement, args=(f, x0), kwargs=dict(J_x0=J_x0, etol=etol, ptol=ptol),
           name='Klement', number=times)

    print(f'\n{"Newton":-^50}')
    timeit(generalized_newton, args=(f, J, x0), kwargs=dict(etol=etol, ptol=ptol), name='Newton', number=times)


if __name__ == '__main__':
    test_output()
    # test_speed()
