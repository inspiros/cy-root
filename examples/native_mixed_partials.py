"""Inspect third mixed partials without NumPy inside the derivative plan."""
from cyroot.native import TensorDifferencePlan


def residual(point):
    x, y = point
    return (x*x*y, x*y*y, x*x*x+y*y*y)


def main():
    plan = TensorDifferencePlan(2, 3, 3, step=.125)
    estimate = plan.evaluate(residual, (.5, 1.))
    assert estimate.status == 'ok'
    assert abs(estimate.derivative[0][0][0][1]-2.) < 1e-10
    assert abs(estimate.derivative[1][0][1][1]-2.) < 1e-10
    print('shape:', plan.shape)
    print('d³F₀/dx²dy:', estimate.derivative[0][0][0][1])
    print('d³F₁/dxdy²:', estimate.derivative[1][0][1][1])
    print('residual evaluations:', estimate.f_calls)


if __name__ == '__main__':
    main()
