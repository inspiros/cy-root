"""Compare full Newton with Armijo-backtracked scalar and vector steps.

Run from a source checkout after building CyRoot: python examples/damped_newton.py
No plotting dependencies are needed. Histories contain accepted states; residual
call counts include rejected trials and numerical differentiation samples.
"""
from cyroot import DampedNewton, VectorDampedNewton, as_root_snapshot


def main():
    scalar = DampedNewton(lambda x: x*x-2., lambda x: 2*x, .1, history=30)
    # Two residuals, one unknown: no special shape flag is needed.
    vector = VectorDampedNewton(
        lambda x: (x[0]*x[0]-2., 2*(x[0]*x[0]-2.)),
        lambda x: ((2*x[0],), (4*x[0],)), (.1,), history=30)
    for name, solver in (("Scalar", scalar), ("Rectangular Vector", vector)):
        print(name)
        for state in solver:
            snapshot = as_root_snapshot(state)
            print(f"  step {snapshot.iters}: root={snapshot.root}, "
                  f"error={snapshot.error:.3g}, alpha={solver.step_scale:.3g}, "
                  f"backtracks={solver.backtracks}")
        print(" ", solver.result)


if __name__ == "__main__":
    main()
