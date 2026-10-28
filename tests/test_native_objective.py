"""Exact least-squares transforms, scaled arithmetic and compiled objective boundary."""
import gc
import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref
from fractions import Fraction as Q

from cyroot.native.objective import LeastSquaresObjective
from cyroot.native import (HessianCallback, NativeCallbackError, RectangularCallback,
                           RectangularHessianCallback, VectorCallback)


def multiply(left, right):
    out = {}
    for a, ca in left.items():
        for b, cb in right.items():
            exponent = tuple(x+y for x,y in zip(a,b))
            out[exponent] = out.get(exponent, Q(0))+ca*cb
    return out


def derivative(poly, axis):
    out = {}
    for powers, coefficient in poly.items():
        if powers[axis]:
            result = list(powers)
            result[axis] -= 1
            out[tuple(result)] = coefficient*powers[axis]
    return out


def evaluate(poly, point):
    return sum(c*math.prod(Q(x)**p for x,p in zip(point, powers))
               for powers,c in poly.items())


def fixture(n, m):
    polynomials = []
    objective = {}
    for o in range(m):
        poly = {(0,)*n: Q(o+1)}
        for i in range(n):
            linear = tuple(int(j == i) for j in range(n))
            square = tuple(2*int(j == i) for j in range(n))
            poly[linear], poly[square] = Q((o+2)*(i+1)), Q(o+i+1)
            for j in range(i):
                mixed = tuple(int(k == i)+int(k == j) for k in range(n))
                poly[mixed] = Q((o+1)*(i+j+1))
        polynomials.append(poly)
        for powers,coefficient in multiply(poly,poly).items():
            objective[powers] = objective.get(powers,Q(0))+coefficient/2
    def f(x):return tuple(float(evaluate(p,x)) for p in polynomials)
    def j(x):return tuple(tuple(float(evaluate(derivative(p,i),x)) for i in range(n))
                          for p in polynomials)
    def h(x):return tuple(tuple(tuple(float(evaluate(derivative(derivative(p,i),k),x))
                                     for k in range(n)) for i in range(n)) for p in polynomials)
    return f,j,h,objective


class NativeObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-objective-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        name = 'native_objective_consumer'
        (directory/(name+'.pyx')).write_bytes(Path(__file__).with_name(name+'.pyx').read_bytes())
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-objective-consumer",ext_modules=cythonize(\n'
            ' [Extension("native_objective_consumer",["native_objective_consumer.pyx"],\n'
            ' language="c++",include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH',None)
        build = subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                               env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:raise AssertionError('Objective external consumer failed:\n'+build.stdout)
        sys.path.insert(0,str(directory))
        cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer = importlib.import_module(name)
        cls.addClassCleanup(sys.modules.pop,name,None)

    def test_exact_scalar_square_tall_wide_polynomial_derivatives_and_stationary_nonroot(self):
        for n,m in ((1,1),(2,2),(2,3),(3,2)):
            f,j,h,polynomial = fixture(n,m)
            point = tuple(Q(i+1,4) for i in range(n))
            objective = LeastSquaresObjective(f,n,m,jacobian=j,hessian=h)
            expected_value = float(evaluate(polynomial,point))
            expected_gradient = tuple(float(evaluate(derivative(polynomial,i),point)) for i in range(n))
            expected_hessian = tuple(tuple(float(evaluate(derivative(derivative(polynomial,i),k),point))
                                          for k in range(n)) for i in range(n))
            with self.subTest(n=n,m=m):
                self.assertEqual(objective(point),expected_value)
                self.assertEqual(objective.value(point),expected_value)
                self.assertEqual(objective.residual(point),f(point))
                self.assertEqual(objective.gradient(point),expected_gradient)
                self.assertEqual(objective.hessian(point),expected_hessian)
                self.assertIsInstance(objective.gradient(point),tuple)
                self.assertIsInstance(objective.hessian(point)[0],tuple)
                # Copied results survive subsequent reuse.
                retained = objective.gradient(point)
                objective.gradient(tuple(x+1 for x in point))
                self.assertEqual(retained,expected_gradient)
                self.assertEqual((objective.input_dimension,objective.output_dimension),(n,m))
                count=n+m+max(m,n,n*n)+n*m+n*n*m
                self.assertEqual(objective.memory_bytes,8*count)
        # A minimum of phi can have a nonzero residual; this adapter returns no
        # root-success claim. Exact F*H term is indispensable away from a root.
        stationary = LeastSquaresObjective(lambda x:(x[0]**2+1.,),1,
            jacobian=lambda x:((2*x[0],),),hessian=lambda x:(((2.,),),))
        self.assertEqual((stationary.value((0.,)),stationary.gradient((0.,)),
                          stationary.hessian((0.,)),stationary.residual((0.,))),
                         (.5,(0.,),((2.,),),(1.,)))

    def test_scaled_final_values_missing_derivatives_and_finite_validation(self):
        objective = LeastSquaresObjective(lambda x:(1e154,1e154),1,2)
        self.assertEqual(objective.value((0.,)),1e308)
        huge = LeastSquaresObjective(lambda x:(1.4e154,1.4e154),1,2)
        with self.assertRaises(FloatingPointError):huge.value((0.,))
        cancel = LeastSquaresObjective(lambda x:(1e300,-1e300,1.),1,3,
                                     jacobian=lambda x:((1e300,),(1e300,),(3.,)))
        self.assertEqual(cancel.gradient((0.,)),(3.,))
        hcancel = LeastSquaresObjective(lambda x:(1e300,),1,
            jacobian=lambda x:((1e300,),),hessian=lambda x:(((-1e300,),),))
        self.assertEqual(hcancel.hessian((0.,)),((0.,),))
        tiny = math.ulp(0.)
        underflow = LeastSquaresObjective(lambda x:(tiny,tiny),1,2,
                                         jacobian=lambda x:((.5,),(.5,)))
        self.assertEqual(underflow.gradient((0.,)),(tiny,))
        self.assertEqual(objective.residual((0.,)),(1e154,1e154))
        for method in (objective.gradient,objective.hessian,cancel.hessian):
            with self.assertRaisesRegex(ValueError,'explicit'):method((0.,))
        for value in (math.nan,math.inf,-math.inf):
            with self.assertRaises(FloatingPointError):objective.value((value,))
            bad = LeastSquaresObjective(lambda x:(value,),1)
            with self.assertRaises(FloatingPointError):bad.value((0.,))
        with self.assertRaises(ValueError):LeastSquaresObjective(lambda x:(0.,),0)
        with self.assertRaises(ValueError):LeastSquaresObjective(lambda x:(0.,),True)
        with self.assertRaises(ValueError):LeastSquaresObjective(lambda x:(0.,),1,workspace_bytes=23)
        with self.assertRaises(ValueError):objective.value((0.,1.))
        with self.assertRaises(TypeError):objective.value('0')
        with self.assertRaises(RuntimeError):objective.__init__(lambda x:(0.,),1)
        uninitialized = LeastSquaresObjective.__new__(LeastSquaresObjective)
        with self.assertRaises(RuntimeError):uninitialized.value((0.,))

    def test_compiled_nogil_status_transactional_output_and_context_lifetime(self):
        for m,square in ((1,False),(2,True),(3,False)):
            owner = self.consumer.Owner(m)
            f,j,h = owner.callbacks(square=square)
            objective = LeastSquaresObjective(f,2,m,jacobian=j,hessian=h)
            point=(.5,.25)
            expected=(objective.residual(point),objective.value(point),
                      objective.gradient(point),objective.hessian(point))
            # All Python evaluation entry points also release GIL for native callbacks.
            self.assertEqual(owner.gil_held,0)
            previous=owner.calls
            for order,value in zip((-1,0,1,2),expected):
                status,result,code,stage,guards = self.consumer.raw(objective,order,point,repeats=3)
                reference=(value,) if order == 0 else value if order != 2 else tuple(value[i][k] for k in range(2) for i in range(2))
                self.assertEqual((status,code,stage,guards),(0,0,0,(1234.,1234.)))
                self.assertEqual(result,reference)
            self.assertEqual(owner.gil_held,0)
            self.assertEqual(owner.calls,tuple(a+b for a,b in zip(previous,(12,6,3))))
            for reserve,capacity,order in ((False,None,0),(True,0,0),(True,None,3)):
                status,result,_,_,guards=self.consumer.raw(objective,order,reserve=reserve,capacity=capacity)
                self.assertEqual(status,2)
                self.assertTrue(all(v==1234. for v in result))
                self.assertEqual(guards,(1234.,1234.))
            self.assertEqual(owner.calls,tuple(a+b for a,b in zip(previous,(12,6,3))))
            with self.assertRaises(RuntimeError):self.consumer.shared_callback_reserved(objective,f)
            reference=weakref.ref(owner)
            del owner,f,j,h
            gc.collect()
            self.assertIsNotNone(reference())
            self.assertEqual(objective.value(point),expected[1])
            del objective
            gc.collect()
            self.assertIsNone(reference())
        for stage in (1,2,3):
            owner=self.consumer.Owner(fail_stage=stage)
            f,j,h=owner.callbacks()
            objective=LeastSquaresObjective(f,2,3,jacobian=j,hessian=h)
            status,result,code,actual_stage,guards=self.consumer.raw(objective,2)
            self.assertEqual((status,code,actual_stage,guards),(3,-23-stage,stage,(1234.,1234.)))
            self.assertEqual(result,(1234.,)*4)
            with self.assertRaises(NativeCallbackError) as raised:objective.hessian((.5,.25))
            self.assertEqual((raised.exception.native_status,raised.exception.callback_stage),
                             (-23-stage,('function','jacobian','hessian')[stage-1]))
            owner=self.consumer.Owner(partial_stage=stage)
            f,j,h=owner.callbacks()
            objective=LeastSquaresObjective(f,2,3,jacobian=j,hessian=h)
            self.assertEqual(self.consumer.raw(objective,2)[:2],(1,(1234.,)*4))

    def test_python_exception_identity_shape_provider_and_reentrancy_guards(self):
        marker=ValueError('original callback error')
        for stage in (1,2,3):
            def f(x):
                self.assertIsInstance(x,tuple)
                if stage==1:raise marker
                return (x[0]**2+1.,)
            def j(x):
                if stage==2:raise marker
                return ((2*x[0],),)
            def h(x):
                if stage==3:raise marker
                return (((2.,),),)
            objective=LeastSquaresObjective(f,1,jacobian=j,hessian=h)
            with self.assertRaises(ValueError) as raised:objective.hessian((0.,))
            self.assertIs(raised.exception,marker)
            self.assertEqual((objective.callback_code,objective.callback_stage),
                             (1,('function','jacobian','hessian')[stage-1]))
        for residual in (lambda x:(),lambda x:(1.,2.)):
            with self.assertRaises(ValueError):LeastSquaresObjective(residual,1).value((0.,))
        for kwargs in (dict(function=VectorCallback(lambda x:(0.,0.),2),input_dimension=2,output_dimension=3),
                       dict(function=RectangularCallback(lambda x:(0.,0.),2,2,jacobian=True),input_dimension=2),
                       dict(function=lambda x:(0.,),input_dimension=1,jacobian=VectorCallback(lambda x:(0.,),1)),
                       dict(function=lambda x:(0.,),input_dimension=1,jacobian=lambda x:((0.,),),hessian=HessianCallback(lambda x:(((0.,),),),2))):
            with self.assertRaises(ValueError):LeastSquaresObjective(**kwargs)
        with self.assertRaisesRegex(ValueError,'Jacobian'):
            LeastSquaresObjective(lambda x:(0.,),1,hessian=lambda x:(((0.,),),))
        callback=RectangularCallback(lambda x:(x[0],),1,1)
        self_outer=self
        class Dimension:
            def __index__(self):
                with self_outer.assertRaises(RuntimeError):callback.evaluate((0.,))
                return 1
        objective=LeastSquaresObjective(callback,Dimension())
        self.assertEqual(objective.value((2.,)),2.)
        # Failed construction must release a reservation held before conversion.
        class BadDimension:
            def __index__(self):
                with self_outer.assertRaises(RuntimeError):callback.evaluate((0.,))
                raise marker
        with self.assertRaises(ValueError) as raised:LeastSquaresObjective(callback,BadDimension())
        self.assertIs(raised.exception,marker)
        self.assertEqual(callback.evaluate((2.,)),(2.,))
        holder={}
        def reentrant(x):
            with self.assertRaises(RuntimeError):holder['objective'].value(x)
            with self.assertRaises(RuntimeError):_ = holder['objective'].memory_bytes
            return (x[0],)
        objective=LeastSquaresObjective(reentrant,1)
        holder['objective']=objective
        self.assertEqual(objective.value((2.,)),2.)
        class ReentrantFloat:
            def __float__(self):
                with self_outer.assertRaises(RuntimeError):objective.value((0.,))
                return 2.
        self_outer=self
        self.assertEqual(objective.value((ReentrantFloat(),)),2.)
        # Reservations are released after conversion errors and original exceptions.
        with self.assertRaises(TypeError):objective.value((object(),))
        self.assertEqual(objective.value((2.,)),2.)
