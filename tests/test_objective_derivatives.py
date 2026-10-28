"""Exact polynomial oracles for objective tensors, bounds and root/optimizer status."""
import importlib
from itertools import product
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose
from cyroot.native import ObjectiveDerivativePlan, LeastSquaresObjective
from cyroot.optimization import root_via_optimizer
from test_native_objective import multiply, derivative, evaluate


def nested_derivative(polynomial, x, n, order, indices=()):
    if len(indices)==order:
        for axis in indices:polynomial=derivative(polynomial,axis)
        return float(evaluate(polynomial,x))
    return tuple(nested_derivative(polynomial,x,n,order,indices+(i,)) for i in range(n))


class ObjectiveDerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='cyroot-objective-tensors-')
        cls.addClassCleanup(cls.directory.cleanup)
        directory=Path(cls.directory.name)
        (directory/'objective_derivatives_consumer.pyx').write_bytes(Path(__file__).with_name('objective_derivatives_consumer.pyx').read_bytes())
        (directory/'setup.py').write_text(
            'from setuptools import setup,Extension\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\nimport sys\n'
            'setup(ext_modules=cythonize([Extension("objective_derivatives_consumer",'
            '["objective_derivatives_consumer.pyx"],language="c++",include_dirs=[get_include()],'
            'extra_compile_args=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"])],'
            'language_level=3,include_path=[get_cython_include()]))\n')
        environment=os.environ.copy();environment.pop('PYTHONPATH',None)
        built=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
            env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if built.returncode:raise AssertionError(built.stdout)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('objective_derivatives_consumer')
        cls.addClassCleanup(sys.modules.pop,'objective_derivatives_consumer',None)

    def test_polynomial_product_oracle_all_partial_orders_and_native_layout(self):
        for n,m,highest in ((1,2,8),(2,3,6),(3,1,4)):
            x=tuple((i+1)/4 for i in range(n))
            polynomials=[];phi={}
            for o in range(m):
                poly={(0,)*n:o+1}
                for axis in range(n):
                    key=tuple(4 if j==axis else 0 for j in range(n));poly[key]=o+axis+1
                if n>1:poly[(1,1)+(0,)*(n-2)]=o+2
                polynomials.append(poly)
                for key,c in multiply(poly,poly).items():phi[key]=phi.get(key,0)+c/2
            for r in range(1,highest+1):
                tensors=[tuple(nested_derivative(p,x,n,k) for p in polynomials) for k in range(r+1)]
                plan=ObjectiveDerivativePlan(n,m,r)
                wanted=nested_derivative(phi,x,n,r)
                assert_allclose(plan.transform(tensors),wanted,rtol=2e-14,atol=1e-12)
                packed=[np.asarray(t).ravel(order='F').tolist() for t in tensors]
                status,flat,gil,guards=self.consumer.raw(plan,packed)
                self.assertEqual((status,gil,guards),(0,0,(1234.,1234.,5678.,5678.)))
                assert_allclose(np.asarray(flat).reshape((n,)*r,order='F'),wanted,rtol=2e-14,atol=1e-12)
                if r<=2:
                    f=lambda point:tuple(nested_derivative(p,point,n,0) for p in polynomials)
                    j=lambda point:tuple(nested_derivative(p,point,n,1) for p in polynomials)
                    h=lambda point:tuple(nested_derivative(p,point,n,2) for p in polynomials)
                    objective=LeastSquaresObjective(f,n,m,jacobian=j,hessian=h)
                    assert_allclose(objective.gradient(x) if r==1 else objective.hessian(x),wanted)

    def test_provider_counts_errors_resources_reentry_and_transactional_raw_output(self):
        plan=ObjectiveDerivativePlan(1,1,3)
        calls=[]
        providers=[lambda x,k=k:calls.append((k,x)) or ((x[0]**3,) if k==0 else
                    ((3*x[0]**2,),) if k==1 else (((6*x[0],),),) if k==2 else ((((6.,),),),)) for k in range(4)]
        self.assertEqual(plan.evaluate(providers[0],providers[1:],(2.,)),(((480.,),),))
        self.assertEqual(calls,[(k,(2.,)) for k in range(4)])
        shared=np.zeros((1,1))
        def residual(x):shared[0,0]=x[0]**2;return shared[0]
        def jacobian(x):shared[0,0]=2*x[0];return shared
        self.assertEqual(ObjectiveDerivativePlan(1,1,1).evaluate(residual,[jacobian],(3.,)),(54.,))
        error=KeyboardInterrupt('original')
        def fail(x):raise error
        with self.assertRaises(KeyboardInterrupt) as caught:plan.evaluate(fail,providers[1:],(2.,))
        self.assertIs(caught.exception,error)
        class Coordinate:
            def __float__(coordinate):
                with self.assertRaises(RuntimeError):plan.transform([])
                return 2.
        self.assertEqual(plan.evaluate(providers[0],providers[1:],(Coordinate(),)),(((480.,),),))
        for options in (dict(order=0),dict(order=64),dict(order=4,max_terms=1),dict(order=4,memory_bytes=0)):
            with self.assertRaises(ValueError):ObjectiveDerivativePlan(2,1,**options)
        with self.assertRaises(ValueError):plan.transform([(1.,)])
        raw=[[8.],[12.],[12.],[6.]]
        for options in (dict(output_capacity=0),dict(scratch_capacity=0),dict(derivative_count=0)):
            status,out,gil,guards=self.consumer.raw(plan,raw,**options)
            self.assertEqual((status,out,gil,guards),(2,(1234.,),0,(1234.,1234.,5678.,5678.)))
        for tensors,status in (([[8.],[12.],[],[6.]],2),([[math.nan],[12.],[12.],[6.]],1)):
            result=self.consumer.raw(plan,tensors)
            self.assertEqual((result[0],result[1]),(status,(1234.,)))
        self.assertEqual(ObjectiveDerivativePlan(1,1,2).transform([(1e308,),((1e308,),),(((-1e308,),),)]),((0.,),))
        tiny=ObjectiveDerivativePlan(1,3,1).transform([(1e308,1e308,1e-308),((2.,),(-2.,),(1.,))])
        self.assertEqual(tiny,(1e-308,))
        with self.assertRaises(ValueError):ObjectiveDerivativePlan(1,1,1).transform([(1.,),((math.nan,),)])
        with self.assertRaises(TypeError):ObjectiveDerivativePlan(1,1,1).transform([(1j,),((1.,),)])
        with self.assertRaises(TypeError):plan.evaluate(providers[0],providers[1:],(np.complex128(1j),))

    def test_external_optimizer_root_status_local_minima_and_copied_results(self):
        from scipy.optimize import minimize
        trap=LeastSquaresObjective(lambda x:(x[0]**3-2*x[0]+2.,x[1]),2,
            jacobian=lambda x:((3*x[0]**2-2,0.),(0.,1.)))
        result=root_via_optimizer(trap,minimize,(math.sqrt(2/3),0.),optimizer_options=dict(method='BFGS',jac=trap.gradient))
        self.assertTrue(result.optimizer_success)
        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason,'optimizer_stopped_without_root')
        self.assertGreater(result.error,.9)
        wide=LeastSquaresObjective(lambda x:(x[0]+x[1]-1.,),2,1,
            jacobian=lambda x:((1.,1.),))
        result=root_via_optimizer(wide,minimize,(0.,0.),optimizer_options=dict(method='BFGS',jac=wide.gradient))
        self.assertTrue(result.success)
        saved=result.root;result.optimizer_result.x[:]=99
        self.assertEqual(result.root,saved)
        def optimizer_failure(fun,x0):return SimpleNamespace(x=(.5,.5),success=False,status=-9)
        result=root_via_optimizer(wide,optimizer_failure,(0.,0.))
        self.assertTrue(result.success);self.assertFalse(result.optimizer_success)
        self.assertEqual(result.optimizer_status,-9)
        def unexpected(*args,**kwargs):self.fail('initial root must skip optimizer')
        result=root_via_optimizer(wide,unexpected,(.5,.5))
        self.assertTrue(result.success);self.assertIsNone(result.optimizer_result)
        class Coordinate:
            calls=0
            def __float__(coordinate):
                coordinate.calls+=1
                return .5 if coordinate.calls==1 else 3.
        coordinate=Coordinate()
        result=root_via_optimizer(wide,unexpected,(coordinate,.5))
        self.assertEqual((coordinate.calls,result.root,result.f_root),(1,(.5,.5),(0.,)))
        for opts in (dict(etol=-1),dict(ertol=1),dict(etol=math.nan)):
            with self.assertRaises(ValueError):root_via_optimizer(wide,minimize,(0.,0.),**opts)
        with self.assertRaises(TypeError):root_via_optimizer(wide,minimize,(np.complex128(1j),0.))
