"""Compute J(x)v without allocating a Jacobian; no plotting dependencies needed."""
from cyroot import JacobianVectorCallback, JacobianVectorPlan


def residual(x):
    return (x[0]*x[0]+x[1], x[0]*x[1], 2*x[0]-x[1])


def product(x, v):
    return (2*x[0]*v[0]+v[1], x[1]*v[0]+x[0]*v[1], 2*v[0]-v[1])


def main():
    x, v = (1.2, .7), (.5, -2.)
    supplied = JacobianVectorCallback(product, 2, 3)
    print("Supplied:", supplied.evaluate(x, v))
    numerical = JacobianVectorPlan(2, 3)
    print("Numerical:", numerical.evaluate(residual, x, v))
    print("Cached Base:", numerical.evaluate(residual, x, v, f_x=residual(x)))
    print("Scratch Bytes:", numerical.workspace_bytes)


if __name__ == "__main__":
    main()
