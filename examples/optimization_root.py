"""Use a familiar optimizer and distinguish its status from actual root success."""
from math import sqrt
from scipy.optimize import minimize
from cyroot.optimization import LeastSquaresObjective, ObjectiveDerivativePlan, root_via_optimizer


def main():
    objective=LeastSquaresObjective(
        lambda x:(x[0]**2+x[1]-2.,x[0]*x[1]-.5),2,
        jacobian=lambda x:((2*x[0],1.),(x[1],x[0])),
        hessian=lambda x:(((2.,0.),(0.,0.)),((0.,1.),(1.,0.))))
    root=root_via_optimizer(objective,minimize,(1.,.4),
        optimizer_options={'method':'BFGS','jac':objective.gradient,'options':{'gtol':1e-12}})
    assert root.success and root.error<=1e-10
    print('Coupled root:',root.root,'residual:',root.f_root)
    print('Exact objective Hessian:',objective.hessian(root.root))

    trap=LeastSquaresObjective(lambda x:(x[0]**3-2*x[0]+2.,x[1]),2,
        jacobian=lambda x:((3*x[0]**2-2.,0.),(0.,1.)))
    stopped=root_via_optimizer(trap,minimize,(sqrt(2/3),0.),
        optimizer_options={'method':'BFGS','jac':trap.gradient})
    assert stopped.optimizer_success and not stopped.success
    print('Optimizer stopped at non-root:',stopped.root,'error:',stopped.error)

    third=ObjectiveDerivativePlan(1,1,3).evaluate(lambda x:(x[0]**3,),
        [lambda x:((3*x[0]**2,),),lambda x:(((6*x[0],),),),lambda x:((((6.,),),),)],(2.,))
    assert third==(((480.,),),)
    print('Third objective derivative:',third)


if __name__=='__main__':main()
