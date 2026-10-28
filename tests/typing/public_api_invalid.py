"""Each marked line must be rejected by a type checker."""
import cyroot
from cyroot import ahmadi  # error: attr-defined

def f(x: float) -> float:
    return x*x-2.

cyroot.newton(f, None, 1.5, max_iter='ten')  # error: arg-type
cyroot.bisect(f, 1., 2., None, None, 1)  # error: misc
cyroot.generalized_newton(lambda x: x, None, (0.,), history_bytes='large')  # error: arg-type
cyroot.newton(f, None, 1.5).iters = 5  # error: misc
snapshot = cyroot.as_root_snapshot(cyroot.newton(f, None, 1.5))
snapshot.root = 4.  # error: misc
bad_count: str = snapshot.iters  # error: assignment
cyroot.as_root_snapshot(cyroot.hybisect(f, lambda a,b: (a,b), 1., 2.))  # error: arg-type
