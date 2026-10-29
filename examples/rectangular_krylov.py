"""Tall and wide root systems use the same matrix-free Newton interface.

Supplied Jv and J.Tw give the efficient path. Omit both for a finite-difference
fallback that recomputes matching columns and stores no dense Jacobian.
"""
from cyroot import newton_krylov


def residual(x):
    a=x[0]*x[0]-2.
    b=x[1]*x[1]-3.
    return a,b,a+b


def jvp(x,v):
    a=2*x[0]*v[0]
    b=2*x[1]*v[1]
    return a,b,a+b


def vjp(x,w):
    return 2*x[0]*(w[0]+w[2]),2*x[1]*(w[1]+w[2])


if __name__=='__main__':
    tall=newton_krylov(residual,jvp,(.1,.2),vjp=vjp)
    wide=newton_krylov(lambda x:(x[0]+x[1]-3.,),None,(0.,0.))
    for name,result in (('Tall',tall),('Wide',wide)):
        print(f'{name}: root={result.root}, error={result.error:.3g}, success={result.success}')
