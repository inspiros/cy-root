"""Rectangular Hessian shapes share the square stencil and compiled consumer."""
import gc
import importlib
import math
import struct
import traceback
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from cyroot.native import (HessianPlan, NativeCallbackError, RectangularCallback,
                           RectangularHessianPlan, VectorCallback)
import test_native_hessian as square_tests
from test_native_hessian import all_nan

hessians = importlib.import_module('cyroot.native.hessians')


def quadratic(point, m):
    return tuple(1.+o+sum((10*(o+1)+j+1)*x*x for j,x in enumerate(point))+
                 sum((10*(o+1)+j+k+1)*point[j]*point[k]
                     for j in range(len(point)) for k in range(j+1,len(point)))
                 for o in range(m))


def tensor(n, m):
    # Diagonal differentiation doubles its quadratic coefficient; mixed terms
    # have one coefficient and are symmetric in the two input axes.
    return tuple(tuple(tuple(2*(10*(o+1)+j+1) if j == k else
                             10*(o+1)+j+k+1 for k in range(n))
                       for j in range(n)) for o in range(m))


class RectangularHessianTests(unittest.TestCase):
    setUpClass = classmethod(square_tests.InstalledHessianConsumerTests.setUpClass.__func__)

    def test_numerical_shapes_signed_steps_counts_and_square_equivalence(self):
        self.assertEqual(tensor(2,3), (((22.,12.),(12.,24.)),
                                     ((42.,22.),(22.,44.)),
                                     ((62.,32.),(32.,64.))))
        for n,m in ((2,1),(2,3),(1,3),(3,2),(2,2)):
            point = tuple(.5*(j+1) for j in range(n))
            steps = tuple((-1)**j*.125 for j in range(n))
            function = lambda x: quadratic(x,m)
            for scheme in ('central','forward','backward'):
                plan = RectangularHessianPlan(n,m,scheme,steps)
                for cached in (False,True):
                    seen = []
                    result = plan.evaluate(lambda x: seen.append(x) or function(x), point,
                                           f_x=function(point) if cached else None)
                    with self.subTest(n=n,m=m,scheme=scheme,cached=cached):
                        self.assertEqual((result.status,result.callback_code),('ok',0))
                        self.assertEqual(result.hessian,tensor(n,m))
                        self.assertEqual(result.f_calls,plan.sample_count+(not cached))
                        self.assertEqual(len(seen),result.f_calls)
                        self.assertTrue(all(type(x) is tuple for x in seen))
                        for plane in result.hessian:
                            for j in range(n):
                                for k in range(n):
                                    self.assertEqual(struct.pack('d',plane[j][k]),
                                                     struct.pack('d',plane[k][j]))
                        if n == m:
                            square = HessianPlan(n,scheme,steps).evaluate(
                                VectorCallback(function,n),point,
                                f_x=function(point) if cached else None)
                            self.assertEqual(result,square)
                            vector = plan.evaluate(VectorCallback(function,n),point,
                                f_x=function(point) if cached else None)
                            self.assertEqual(result,vector)

    def test_properties_budget_steps_dimensions_and_constructor_guards(self):
        steps = [.125,-.25]
        n,m = 2,3
        work = 2*n*m+8*n+2*m
        evaluation = n+m+m*n*n+work
        plan = RectangularHessianPlan(n,m,'forward',steps,memory_bytes=(evaluation+n)*8)
        steps[:] = [7.,8.]
        self.assertEqual((plan.input_dimension,plan.output_dimension,plan.shape),(2,3,(3,2,2)))
        self.assertEqual((plan.scheme,plan.step,plan.sample_count),('forward',(.125,-.25),5))
        self.assertEqual((plan.workspace_size,plan.workspace_bytes),(work,work*8))
        self.assertEqual((plan.storage_bytes,plan.output_bytes,plan.evaluation_bytes,
                          plan.memory_bytes),(16,96,evaluation*8,(evaluation+2)*8))
        for name in ('input_dimension','output_dimension','shape','scheme','step',
                     'sample_count','storage_bytes','workspace_size','workspace_bytes',
                     'output_bytes','evaluation_bytes','memory_bytes'):
            with self.subTest(name=name),self.assertRaises((AttributeError,TypeError)):
                setattr(plan,name,None)
        for a,b in ((2,1),(2,3),(1,3),(3,2)):
            for explicit in (False,True):
                total = (a+b+b*a*a+2*a*b+8*a+2*b+(a if explicit else 0))*8
                options = {'step':.125} if explicit else {}
                self.assertEqual(RectangularHessianPlan(a,b,memory_bytes=total,**options).memory_bytes,total)
                with self.assertRaises(ValueError):
                    RectangularHessianPlan(a,b,memory_bytes=total-1,**options)
        for bad in (0,-1,True,np.bool_(False),1.5,2**31,1<<100):
            for a,b in ((bad,2),(2,bad)):
                with self.subTest(n=a,m=b),self.assertRaises((TypeError,ValueError,OverflowError)):
                    RectangularHessianPlan(a,b)
        with self.assertRaises((ValueError,OverflowError)):
            RectangularHessianPlan(2**31-1,2**31-1,memory_bytes=64)
        for step in (True,0.,math.nan,math.inf,(.1,),(.1,0.),(.1,True),np.ones(2,dtype=np.int64)):
            with self.subTest(step=step),self.assertRaises((ValueError,TypeError)):
                RectangularHessianPlan(2,3,step=step)
        automatic = RectangularHessianPlan(2,3)
        self.assertEqual((automatic.step,automatic.storage_bytes),(None,0))
        for scheme in (None,True,2,'unknown'):
            with self.assertRaises((ValueError,TypeError)):
                RectangularHessianPlan(2,3,scheme)
        for budget in (True,-1,1.5,1<<100):
            with self.assertRaises((ValueError,TypeError,OverflowError)):
                RectangularHessianPlan(2,3,memory_bytes=budget)
        partial = RectangularHessianPlan.__new__(RectangularHessianPlan)
        with self.assertRaises(RuntimeError):
            partial.evaluate(lambda x:quadratic(x,3),(0.,0.))
        case = self
        class Step:
            def __float__(self):
                with case.assertRaises(RuntimeError):
                    partial.__init__(2,3,step=.1)
                return .125
        partial.__init__(2,3,step=Step())
        with self.assertRaises(RuntimeError):
            partial.__init__(2,3)

    def test_native_pod_layout_reuse_invalid_configs_and_failure_poisoning(self):
        guards = (1234567.,1234567.,9876543.,9876543.)
        for n,m in ((2,1),(2,3),(3,2)):
            point = tuple(.5*(j+1) for j in range(n))
            for scheme in ('central','forward','backward'):
                plan = RectangularHessianPlan(n,m,scheme,step=.125)
                allocation = self.consumer.rectangular_allocation(plan)
                self.assertEqual(allocation[1:],(n,m,n*m,m*n*n,plan.workspace_size))
                for direct in (False,True):
                    result = self.consumer.compute_rectangular_direct(plan,point,use_config=direct,repeats=2)
                    self.assertEqual((result['status'],result['calls']),(0,2*plan.sample_count))
                    self.assertEqual(result['hessian'],tensor(n,m))
                    self.assertEqual(result['flat'],tuple(tensor(n,m)[o][j][k]
                        for k in range(n) for j in range(n) for o in range(m)))
                    self.assertEqual((result['guards'],result['restored_point']),(guards,point))
                    self.assertTrue(result['stable'])
                    self.assertEqual(result['gil_held'],0)
                self.assertEqual(self.consumer.rectangular_allocation(plan),allocation)
        plan = RectangularHessianPlan(2,3,step=.125)
        invalid = [{'capacity':plan.workspace_size-1},{'output_capacity':11}]
        invalid += [dict(use_config=True,config_overrides=value) for value in (
            {'input_dimension':0},{'output_dimension':0},{'input_dimension':2**31},
            {'output_dimension':2**31},{'matrix_size':5},{'tensor_size':11},
            {'work':plan.workspace_size-1},{'scheme':-1},{'scheme':3},
            {'input_dimension':2**31-1,'output_dimension':2**31-1})]
        for options in invalid:
            with self.subTest(options=options):
                result = self.consumer.compute_rectangular_direct(plan,(.5,1.),initial_calls=7,**options)
                self.assertEqual((result['status'],result['calls'],result['callback_code']),(2,7,0))
                self.assertTrue(result['untouched_output'])
                self.assertTrue(result['untouched_workspace'])
                self.assertEqual(result['guards'],guards)
        for options,status,calls in (({'fail_after':0},3,1),({'fail_after':7},3,8),
                ({'partial_after':7},1,8),({'base':(0.,math.nan,0.)},1,0),
                ({'point':(math.inf,1.)},1,0)):
            values = dict(point=(.5,1.)); values.update(options)
            result = self.consumer.compute_rectangular_direct(plan,**values)
            self.assertEqual((result['status'],result['calls']),(status,calls))
            self.assertEqual(result['callback_code'],-19 if status == 3 else 0)
            self.assertTrue(all_nan(result['hessian']))
            self.assertEqual(result['restored_point'],values['point'])
            self.assertEqual(result['guards'],guards)
        for steps,status in (((.125,1e-320),2),((.125,1e308),1)):
            result = self.consumer.compute_rectangular_direct(
                RectangularHessianPlan(2,3,'forward',step=steps),(1.,1.))
            self.assertEqual((result['status'],result['calls']),(status,0))
            self.assertTrue(all_nan(result['hessian']))
            self.assertEqual(result['restored_point'],(1.,1.))

    def test_compiled_callbacks_nogil_lifetime_and_native_diagnostics(self):
        plan = RectangularHessianPlan(2,3,step=.125)
        owner = self.consumer.RectangularContextOwner(2,3)
        reference = weakref.ref(owner)
        adapter = owner.callback()
        del owner; gc.collect()
        result = plan.evaluate(adapter,(.5,1.))
        self.assertEqual(result.hessian,tensor(2,3))
        self.assertEqual((reference().calls,reference().gil_held),(result.f_calls,0))
        del adapter; gc.collect()
        self.assertIsNone(reference())
        for after,stage in ((0,'function'),(1,'hessian'),(8,'hessian')):
            owner = self.consumer.RectangularContextOwner(2,3,fail_after=after,failure_code=-43)
            with self.assertRaises(NativeCallbackError) as caught:
                plan.evaluate(owner.callback(),(.5,1.))
            failure = caught.exception
            self.assertEqual((failure.native_status,failure.callback_stage),(-43,stage))
            self.assertEqual((failure.result.status,failure.result.callback_code,
                              failure.result.f_calls),('callback_failure',-43,after+1))
            self.assertTrue(all_nan(failure.result.hessian))
            self.assertEqual(owner.gil_held,0)
        owner = self.consumer.RectangularContextOwner(2,3,partial_after=8)
        result = plan.evaluate(owner.callback(),(.5,1.))
        self.assertEqual((result.status,result.f_calls),('nonfinite',9))
        self.assertTrue(all_nan(result.hessian))

    def test_python_exception_identity_adapter_checks_and_reservation_boundaries(self):
        plan = RectangularHessianPlan(2,3,step=.125)
        for failed_call in (1,9):
            failure = KeyboardInterrupt('rectangular Hessian sample')
            calls = 0
            def residual(x):
                nonlocal calls
                calls += 1
                if calls == failed_call:
                    raise failure
                return quadratic(x,3)
            adapter = RectangularCallback(residual,2,3)
            try:
                plan.evaluate(adapter,(.5,1.))
            except KeyboardInterrupt as caught:
                self.assertIs(caught,failure)
                self.assertIn('residual',[f.name for f in traceback.extract_tb(caught.__traceback__)])
            else:
                self.fail('Python exception swallowed')
            self.assertEqual(failure.__dict__,{})
            self.assertEqual(plan.evaluate(adapter,(.5,1.)).hessian,tensor(2,3))
        def unexpected(x):
            self.fail('Invalid adapter reached callback')
        class Unconverted:
            def __len__(self):
                raise AssertionError('Invalid adapter coerced input')
        for invalid in (RectangularCallback(unexpected,3,2),
                        RectangularCallback(unexpected,2,3,jacobian=True),
                        VectorCallback(unexpected,2),
                        RectangularCallback.__new__(RectangularCallback)):
            with self.assertRaises((ValueError,TypeError,RuntimeError)):
                plan.evaluate(invalid,Unconverted())
        adapter = RectangularCallback(lambda x:quadratic(x,3),2,3)
        case = self
        attempts = []
        class Sequence:
            def __init__(self,values):
                self.values = values
            def __len__(self):
                attempts.append(1)
                with case.assertRaises(RuntimeError):
                    plan.evaluate(adapter,(0.,0.))
                return len(self.values)
            def __getitem__(self,index):
                return self.values[index]
        result = plan.evaluate(adapter,Sequence((.5,1.)),f_x=Sequence(quadratic((.5,1.),3)))
        self.assertEqual(result.hessian,tensor(2,3))
        self.assertTrue(attempts)
        original = hessians.HessianEstimate
        def estimate(*args):
            with self.assertRaises(RuntimeError):
                plan.evaluate(adapter,(0.,0.))
            return original(*args)
        with patch.object(hessians,'HessianEstimate',estimate):
            self.assertEqual(plan.evaluate(adapter,(.5,1.)).status,'ok')
        failure = ArithmeticError('cached component')
        class BadFloat:
            def __float__(self):
                raise failure
        with self.assertRaises(ArithmeticError) as caught:
            plan.evaluate(adapter,(.5,1.),f_x=(0.,BadFloat(),0.))
        self.assertIs(caught.exception,failure)
        self.assertEqual(plan.evaluate(adapter,(.5,1.)).status,'ok')
        for point,base in (((0.,),None),((0.,0.),(1.,)),((0.,0.),(1.,2.,3.,4.))):
            with self.assertRaises((TypeError,ValueError)):
                plan.evaluate(adapter,point,f_x=base)
        for point,base in (((math.inf,0.),None),((0.,0.),(0.,math.nan,0.))):
            result = plan.evaluate(unexpected,point,f_x=base)
            self.assertEqual((result.status,result.f_calls),('nonfinite',0))
            self.assertTrue(all_nan(result.hessian))


if __name__ == '__main__':
    unittest.main()
