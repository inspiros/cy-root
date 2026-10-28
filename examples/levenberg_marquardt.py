"""Native damped least-squares steps for a coupled nonlinear root problem."""
from cyroot.optimization import LeastSquaresObjective, LevenbergMarquardt


def residual(x):
    return (10*(x[1]-x[0]*x[0]), 1-x[0])


def jacobian(x):
    return ((-20*x[0],10.),(-1.,0.))


def main():
    objective=LeastSquaresObjective(residual,2,jacobian=jacobian)
    solver=LevenbergMarquardt(objective,(-1.2,1.),history=32)
    result=solver.run()
    assert result.success and result.error<=1e-10
    print(result)
    print(f'{len(solver.history)} snapshots, {result.rejected_trials} rejected trials')


if __name__=='__main__':main()
