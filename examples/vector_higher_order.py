"""Square matrix methods with tuple callbacks, native plans and copied history."""
from cyroot import generalized_halley, generalized_chebyshev, generalized_super_halley
from cyroot.native import JacobianPlan, HessianPlan, VectorHalley


def residual(point):
    x,y=point
    return (x*x+x*y+y-5., 2*x+y*y-6.)


def jacobian(point):
    x,y=point
    return ((2*x+y,x+1.), (2.,2*y))


def hessian(point):
    return (((2.,1.),(1.,0.)), ((0.,0.),(0.,2.)))


def main():
    for method in (generalized_halley,generalized_chebyshev,generalized_super_halley):
        result=method(residual,jacobian,hessian,(1.25,1.75))
        assert result.success
        print(method.__name__,result.root)
    result=generalized_halley(residual,JacobianPlan(2,step=1e-4),
                              HessianPlan(2,step=1e-3),(1.25,1.75),alpha=.25)
    assert result.success
    print('modified Halley with native plans',result.root)
    solver=VectorHalley(residual,jacobian,hessian,(1.25,1.75),history=16)
    solver.step()
    result=solver.run()
    assert result.success
    for snapshot in solver.history:
        print(snapshot.iters,snapshot.root,snapshot.linear_stage)


if __name__=='__main__':
    main()
