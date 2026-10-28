"""Derivative, Jacobian and Hessian providers beyond fixed finite differences.

Run: python examples/numerical_derivatives.py

Complex-step requires a holomorphic complex implementation of a real residual.
The arithmetic-only functions here support both real and complex inputs. abs,
conjugation and real casts can silently invalidate complex-step estimates.
Adaptive Richardson uses real calls and reports an estimated error, not a bound.
Both methods are explicit providers; existing default differences are unchanged.
Solver counters count supplied derivative/Jacobian calls. estimate.f_calls counts
the underlying function samples for that one derivative/Jacobian evaluation.
"""
from cyroot.native import VectorChebyshev
from cyroot import (
    ComplexStepDerivative, AdaptiveDerivative, AdaptiveSecondDerivative, ComplexStepJacobian, AdaptiveJacobian, AdaptiveHessian,
    newton, halley, generalized_newton,
)


def function(x):
    return x*x*x-x-2.


def residual(point):
    x,y=point
    return x*x-1.,y*y-4.,x+y-3.


def main():
    for method in (ComplexStepDerivative,AdaptiveDerivative):
        derivative=method(function)
        estimate=derivative.evaluate(1.5)
        print(method.__name__,estimate)
        result=newton(function,derivative,1.5)
        assert result.success
        print('Scalar root:',result.root)
    first=ComplexStepDerivative(function)
    second=AdaptiveSecondDerivative(function)
    print('Adaptive second derivative:',second.evaluate(1.5))
    result=halley(function,first,second,1.5)
    assert result.success
    print('Halley root:',result.root)
    for method in (ComplexStepJacobian,AdaptiveJacobian):
        jacobian=method(residual,2,3)
        print(method.__name__,jacobian.evaluate((1.5,2.5)))
        result=generalized_newton(residual,jacobian,(1.5,2.5))
        assert result.success
        print('Vector root:',result.root)
    hessian=AdaptiveHessian(residual,2,3)
    print('Adaptive residual Hessian:',hessian.evaluate((1.5,2.5)))
    result=VectorChebyshev(residual,AdaptiveJacobian(residual,2,3),hessian,(1.5,2.5)).run()
    assert result.success
    print('Chebyshev root:',result.root)


if __name__=='__main__':
    main()
