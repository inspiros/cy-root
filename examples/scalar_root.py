import math

from cyroot import *
from utils import timeit

f = lambda x: x ** 3 + 2 * x ** 2 - 3 * x + 1
df = lambda x: 3 * x ** 2 + 4 * x - 3
d2f = lambda x: 3 * x + 4
d3f = lambda x: 3
d4f = lambda x: 0
interval_f = lambda x_l, x_h: (x_l ** 3 + 2 * (min(abs(x_l), abs(x_h))
                                               if math.copysign(1, x_l) * math.copysign(1, x_h) > 0
                                               else 0) ** 2 - 3 * x_h + 1,
                               x_h ** 3 + 2 * max(abs(x_l), abs(x_h)) ** 2 - 3 * x_l + 1)


def test_output(etol=1e-8, ptol=1e-10):
    # examples output
    print('Output Test')

    print(f'\n{"Bracketing":-^50}')
    print('[Bisect]', bisect(f, -10, 10, etol=etol, ptol=ptol))
    print('[Modified Bisect]', bisect(f, -10, 10, algo='modified', etol=etol, ptol=ptol))
    print('[HyBisect]', hybisect(f, interval_f, -10, 10, etol=etol, ptol=ptol))
    print('[Regula Falsi]', regula_falsi(f, -10, 10, etol=etol, ptol=ptol))
    print('[Illinois]', illinois(f, -10, 10, etol=etol, ptol=ptol))
    print('[Pegasus]', pegasus(f, -10, 10, etol=etol, ptol=ptol))
    print('[Anderson-Bjorck]', anderson_bjorck(f, -10, 10, etol=etol, ptol=ptol))
    print('[Dekker]', dekker(f, -10, 10, etol=etol, ptol=ptol))
    print('[Brent]', brent(f, -10, 10, etol=etol, ptol=ptol))
    print('[Chandrupatla]', chandrupatla(f, -10, 10, etol=etol, ptol=ptol))
    print('[Ridders]', ridders(f, -10, 10, etol=etol, ptol=ptol))
    print('[Toms748]', toms748(f, -10, 10, k=1, etol=etol, ptol=ptol))
    print('[Wu]', wu(f, -10, 10, etol=etol, ptol=ptol))
    print('[ITP]', itp(f, -10, 10, etol=etol, ptol=ptol))

    print(f'\n{"Quasi-Newton":-^50}')
    print('[Secant]', secant(f, -10, 10, etol=etol, ptol=ptol))
    print('[Sidi]', sidi(f, [-10, -5, 0, 5, 10], etol=etol, ptol=ptol))
    print('[Steffensen]', steffensen(f, -5, etol=etol, ptol=ptol))
    print('[Inverse Quadratic Interp]', inverse_quadratic_interp(f, -10, -5, 0, etol=etol, ptol=ptol))
    print('[Hyperbolic Interp]', hyperbolic_interp(f, -10, -5, 0, etol=etol, ptol=ptol))
    print('[Muller]', muller(f, -10, -5, 0, etol=etol, ptol=ptol))

    print(f'\n{"Newton":-^50}')
    print('[Newton]', newton(f, df, -5, etol=etol, ptol=ptol))
    print('[Halley]', halley(f, df, d2f, -5, etol=etol, ptol=ptol))
    print('[Super-Halley]', super_halley(f, df, d2f, -5, etol=etol, ptol=ptol))
    print('[Chebyshev]', chebyshev(f, df, d2f, -5, etol=etol, ptol=ptol))
    print('[Householder]', householder(f, [df, d2f, d3f], -5, etol=etol, ptol=ptol))


def test_speed(etol=1e-8, ptol=1e-10, times=100):
    # examples speed
    print('Speed Test')

    print(f'\n{"Bracketing":-^50}')
    timeit(bisect, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Bisect', number=times)
    timeit(regula_falsi, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Regula Falsi', number=times)
    timeit(illinois, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Illinois', number=times)
    timeit(pegasus, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Pegasus', number=times)
    timeit(anderson_bjorck, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Anderson-Bjork', number=times)
    timeit(dekker, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Dekker', number=times)
    timeit(brent, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Brent', number=times)
    timeit(chandrupatla, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Chandrupatla', number=times)
    timeit(ridders, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Ridders', number=times)
    timeit(toms748, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Toms748', number=times)
    timeit(wu, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='Wu', number=times)
    timeit(itp, args=(f, -10, 10), kwargs=dict(etol=etol, ptol=ptol), name='ITP', number=times)

    print(f'\n{"Quasi-Newton":-^50}')
    timeit(secant, args=(f, -10, -5), kwargs=dict(etol=etol, ptol=ptol), name='Secant', number=times)
    timeit(sidi, args=(f, [-10, -5, 0, 5, 10]), kwargs=dict(etol=etol, ptol=ptol), name='Sidi', number=times)
    timeit(steffensen, args=(f, -5), kwargs=dict(etol=etol, ptol=ptol), name='Steffensen', number=times)
    timeit(inverse_quadratic_interp, args=(f, -10, -5, 0), kwargs=dict(etol=etol, ptol=ptol),
           name='Inverse Quadratic Interp', number=times)
    timeit(hyperbolic_interp, args=(f, -10, -5, 0), kwargs=dict(etol=etol, ptol=ptol),
           name='Hyperbolic Interp', number=times)
    timeit(muller, args=(f, -10, -5, 0), kwargs=dict(etol=etol, ptol=ptol), name='Muller', number=times)

    print(f'\n{"Newton":-^50}')
    timeit(newton, args=(f, df, -5), kwargs=dict(etol=etol, ptol=ptol), name='Newton', number=times)
    timeit(halley, args=(f, df, d2f, -5), kwargs=dict(etol=etol, ptol=ptol), name='Halley', number=times)
    timeit(super_halley, args=(f, df, d2f, -5), kwargs=dict(etol=etol, ptol=ptol), name='Super-Halley', number=times)
    timeit(chebyshev, args=(f, df, d2f, -5), kwargs=dict(etol=etol, ptol=ptol), name='Chebyshev', number=times)
    timeit(householder, args=(f, [df, d2f, d3f], -5), kwargs=dict(etol=etol, ptol=ptol, c_code=True),
           name='Householder', number=times, warmup=True)


if __name__ == '__main__':
    test_output()
    # test_speed()
