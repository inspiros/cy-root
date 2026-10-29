"""Solve a coupled nonlinear system without constructing a Jacobian."""
import math
from cyroot import NewtonKrylov


def main():
    n = 32
    target = tuple(math.sin((i+1)*math.pi/(n+1)) for i in range(n))
    def model(x):
        return tuple(2*x[i]-(x[i-1] if i else 0.)-(x[i+1] if i+1<n else 0.)
                     + .1*math.sin(x[i]) for i in range(n))
    forcing = model(target)
    def residual(x):
        return tuple(a-b for a, b in zip(model(x), forcing))
    def product(x, v):
        return tuple((2+.1*math.cos(x[i]))*v[i]-(v[i-1] if i else 0.)
                     -(v[i+1] if i+1<n else 0.) for i in range(n))
    # Omit jvp=product to use directional finite differences instead.
    solver = NewtonKrylov(residual, (0.,)*n, jvp=product, history=20)
    for state in solver:
        print(f"Step {state.iters}: residual={state.error:.3g}, "
              f"linear residual={state.linear_residual:.3g}, alpha={state.step_scale:.3g}")
    result = solver.result
    print("Success:", result.success)
    print("Residual / Product Calls:", result.f_calls)
    print("Maximum Root Error:", max(abs(a-b) for a, b in zip(result.root, target)))


if __name__ == "__main__":
    main()
