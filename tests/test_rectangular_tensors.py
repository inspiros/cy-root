"""Analytic rectangular Hessian adapters and native ordered contractions."""
import gc
import importlib
import math
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from cyroot.native import NativeCallbackError, RectangularHessianCallback
import test_native_vector_chebyshev as square_fixture


def tensor(n,m):
    return tuple(tuple(tuple(100*(o+1)+10*(j+1)+k+1. for k in range(n))
                       for j in range(n)) for o in range(m))


def all_nan(values):
    return all(all_nan(value) if isinstance(value,(tuple,list)) else math.isnan(value)
               for value in values)


class RectangularTensorTests(unittest.TestCase):
    setUpClass=classmethod(square_fixture.InstalledVectorChebyshevConsumerTests.setUpClass.__func__)

    def test_tall_wide_ordered_contractions_and_square_equivalence(self):
        for n,m in ((2,3),(3,2),(2,2),(1,1)):
            h=tensor(n,m);s=(2.,-3.,1.)[:n]
            matrix=tuple(tuple(10*(o+1)+k+1. for k in range(n)) for o in range(m))
            expected={
                'quadratic':tuple(sum(h[o][j][k]*s[j]*s[k] for j in range(n) for k in range(n)) for o in range(m)),
                'direction':tuple(sum(h[o][j][k]*s[j] for j in range(n)) for k in range(n) for o in range(m)),
                'matrix':tuple(sum(matrix[o][k]*s[k] for k in range(n)) for o in range(m)),
            }
            for kind in expected:
                with self.subTest(n=n,m=m,kind=kind):
                    answer=self.consumer.rectangular_contract(matrix if kind=='matrix' else h,s,kind)
                    self.assertEqual((answer['status'],answer['value']),(0,expected[kind]))
                    self.assertEqual(answer['guards'],(1234567.,7654321.))
                    self.assertEqual(answer['direction'],s)
                    self.assertEqual(answer['flat'],tuple(matrix[o][j] for j in range(n) for o in range(m))
                        if kind=='matrix' else tuple(h[o][j][k] for k in range(n) for j in range(n) for o in range(m)))
                    if n==m:self.assertEqual((answer['square_status'],answer['square']),(0,expected[kind]))
        for kind in ('quadratic','direction','matrix'):
            h=[list(map(list,plane)) for plane in tensor(2,3)]
            h[2][1][1]=math.inf
            values=tuple(tuple(1. for j in range(2)) for o in range(3)) if kind=='matrix' else h
            if kind=='matrix':values=values[:2]+((1.,math.inf),)
            answer=self.consumer.rectangular_contract(values,(1.,0.),kind)
            self.assertEqual(answer['status'],1);self.assertTrue(all_nan(answer['value']))
            for options in ({'n_override':0},{'m_override':0},{'n_override':2**63}):
                answer=self.consumer.rectangular_contract(values,(1.,0.),kind,**options)
                self.assertEqual(answer['status'],2)
                self.assertEqual(answer['value'],(5555555.,)*len(answer['value']))
                self.assertEqual(answer['guards'],(1234567.,7654321.))

    def test_python_shapes_copies_metadata_and_workspace(self):
        for n,m in ((2,3),(3,2)):
            seen=[];source=np.arange(float(m*n*n)).reshape(m,n,n)[:,::-1,::-1]
            def function(point):seen.append(point);return source
            callback=RectangularHessianCallback(function,n,m)
            point=np.arange(float(n*2))[::2];point.flags.writeable=False
            expected=tuple(tuple(tuple(row) for row in plane) for plane in source)
            result=callback.evaluate(memoryview(point),workspace_bytes=8*(n+m*n*n))
            self.assertEqual(result,expected);self.assertEqual(seen,[tuple(point)])
            self.assertIsInstance(seen[0],tuple);self.assertIsInstance(result,tuple)
            source[:]=0.;self.assertEqual(result,expected)
            self.assertEqual((callback.input_dimension,callback.output_dimension,callback.shape,callback.native),
                             (n,m,(m,n,n),False))
            self.assertFalse(callable(callback))
            for name in ('input_dimension','output_dimension','shape','native'):
                with self.assertRaises((AttributeError,TypeError)):setattr(callback,name,None)
            with self.assertRaises(ValueError):callback.evaluate(point,workspace_bytes=8*(n+m*n*n)-1)
            for bad in ((1.,),((1.,),(2.,)),'12',np.ones(n,dtype='f4')):
                with self.assertRaises((ValueError,TypeError)):callback.evaluate(bad)
            for bad in ((),tensor(n,m)[:-1],tuple(((1.,),)*n for _ in range(m)),
                        tuple(('bad',)*n for _ in range(m)),np.ones((m,n,n),dtype=complex)):
                with self.assertRaises((ValueError,TypeError)):
                    RectangularHessianCallback(lambda x:bad,n,m).evaluate(point)
        partial=RectangularHessianCallback(lambda x:(((math.nan,),),),1,1)
        self.assertTrue(all_nan(partial.evaluate((1.,))))
        for budget in (0,-1,True,1.5,'8'):
            with self.assertRaises((ValueError,TypeError,OverflowError)):
                partial.evaluate((1.,),workspace_bytes=budget)

    def test_dimensions_capsules_factories_and_constructor_reservation(self):
        for value in (0,-1,True,1.5,'2',2**31,2**64):
            for n,m in ((value,2),(2,value)):
                with self.assertRaises((ValueError,TypeError,OverflowError)):
                    RectangularHessianCallback(lambda x:(),n,m)
                # Native size_t parameters cannot distinguish bools or
                # fractional Python values after fixture argument conversion.
                if not isinstance(value, (bool, float)):
                    with self.assertRaises((ValueError,TypeError,OverflowError)):
                        self.consumer.RectangularHessianOwner().callback(n,m)
        owner=self.consumer.RectangularHessianOwner()
        for options in ({'version':2},{'struct_size':1},{'n':0},{'m':0},
                        {'n':2**31},{'m':2**31},{'null_function':True},{'wrong_name':True}):
            with self.subTest(options=options),self.assertRaises((ValueError,TypeError,OverflowError)):
                RectangularHessianCallback.from_capsule(owner.capsule(**options),keepalive=owner)
        with self.assertRaises(ValueError):self.consumer.rectangular_hessian_null_factory()
        with self.assertRaises(ValueError):RectangularHessianCallback.from_capsule(object())
        callback=RectangularHessianCallback.__new__(RectangularHessianCallback)
        with self.assertRaises(RuntimeError):callback.evaluate((1.,2.))
        case=self
        class Dimension:
            def __index__(self):
                with case.assertRaises(RuntimeError):callback.__init__(lambda x:tensor(2,3),2,3)
                with case.assertRaises(RuntimeError):callback.evaluate((1.,2.))
                return 2
        callback.__init__(lambda x:tensor(2,3),Dimension(),3)
        with self.assertRaises(RuntimeError):callback.__init__(lambda x:(),2,3)
        retry=RectangularHessianCallback.__new__(RectangularHessianCallback)
        with self.assertRaises(TypeError):retry.__init__(None,2,3)
        retry.__init__(lambda x:tensor(2,3),2,3)
        self.assertEqual(retry.evaluate((1.,2.)),tensor(2,3))
        class Number:
            def __float__(self):
                with case.assertRaises(RuntimeError):callback.evaluate((1.,2.))
                with case.assertRaises(RuntimeError):callback.__init__(lambda x:(),2,3)
                return 1.
        self.assertEqual(callback.evaluate((Number(),2.)),tensor(2,3))
        class Budget:
            def __index__(self):
                with case.assertRaises(RuntimeError):callback.evaluate((1.,2.))
                return 112
        self.assertEqual(callback.evaluate((1.,2.),workspace_bytes=Budget()),tensor(2,3))
        with self.assertRaises(RuntimeError):
            self.consumer.rectangular_hessian_reserved(callback,lambda:callback.evaluate((1.,2.)))
        self.assertEqual(callback.evaluate((1.,2.)),tensor(2,3))

    def test_native_capsule_lifetime_no_gil_and_raw_poisoning(self):
        for n,m in ((2,3),(3,2)):
            for capsule in (False,True):
                owner=self.consumer.RectangularHessianOwner();reference=weakref.ref(owner)
                callback=(RectangularHessianCallback.from_capsule(owner.capsule(n,m),keepalive=owner)
                          if capsule else owner.callback(n,m))
                del owner;gc.collect();self.assertIsNotNone(reference())
                self.assertTrue(callback.native);self.assertEqual(callback.evaluate((1.,)*n),tensor(n,m))
                raw=self.consumer.rectangular_hessian_raw(callback,(1.,)*n)
                self.assertEqual(raw['status'],0)
                self.assertEqual(raw['value'],tuple(tensor(n,m)[o][j][k] for k in range(n) for j in range(n) for o in range(m)))
                self.assertEqual(raw['guards'],(1234567.,7654321.))
                self.assertEqual(reference().counters,(2,0))
                del callback;gc.collect();self.assertIsNone(reference())
        partial=self.consumer.RectangularHessianOwner(partial=True).callback(2,3)
        raw=self.consumer.rectangular_hessian_raw(partial,(1.,2.))
        self.assertEqual((raw['status'],raw['value'][0]),(0,111.))
        self.assertTrue(all_nan(raw['value'][1:]));self.assertEqual(raw['guards'],(1234567.,7654321.))
        copied=partial.evaluate((1.,2.));self.assertEqual(copied[0][0][0],111.)
        self.assertTrue(math.isnan(copied[2][1][1]))
        for code in (-23,19):
            failed=self.consumer.RectangularHessianOwner(code=code).callback(2,3)
            raw=self.consumer.rectangular_hessian_raw(failed,(1.,2.))
            self.assertEqual(raw['status'],code);self.assertTrue(all_nan(raw['value']))
            self.assertEqual(raw['guards'],(1234567.,7654321.))
            with self.assertRaises(NativeCallbackError) as caught:failed.evaluate((1.,2.))
            self.assertEqual((caught.exception.native_status,caught.exception.callback_stage),(code,'hessian'))
            self.assertTrue(all_nan(caught.exception.result))

    def test_python_exception_identity_partial_cleanup_and_error_export_reservation(self):
        failure=KeyboardInterrupt('rectangular tensor bridge');fails=[True]
        def function(point):
            with self.assertRaises(RuntimeError):callback.evaluate(point)
            if fails[0]:raise failure
            return tensor(2,3)
        callback=RectangularHessianCallback(function,2,3)
        for action in (callback.evaluate,lambda point:self.consumer.rectangular_hessian_raw(callback,point)):
            with self.assertRaises(KeyboardInterrupt) as caught:action((1.,2.))
            self.assertIs(caught.exception,failure)
        fails[0]=False;self.assertEqual(callback.evaluate((1.,2.)),tensor(2,3))
        raw=self.consumer.rectangular_hessian_raw(callback,(1.,2.))
        self.assertEqual(raw['status'],0)
        case=self
        class Entry:
            def __float__(self):
                with case.assertRaises(RuntimeError):packing.evaluate((1.,2.))
                return 7.
        packing=RectangularHessianCallback(lambda point:(((Entry(),2.),(3.,4.)),)*3,2,3)
        self.assertEqual(packing.evaluate((1.,2.))[0][0][0],7.)
        failed=self.consumer.RectangularHessianOwner(code=-41).callback(2,3)
        module=importlib.import_module('cyroot.native.tensor_callbacks')
        def guarded_error(code,result,stage):
            with self.assertRaises(RuntimeError):failed.evaluate((1.,2.))
            self.assertTrue(all_nan(result));self.assertEqual(stage,'hessian')
            return NativeCallbackError(code,result,stage)
        with patch.object(module,'NativeCallbackError',guarded_error):
            with self.assertRaises(NativeCallbackError) as caught:failed.evaluate((1.,2.))
        self.assertEqual(caught.exception.native_status,-41)
        with self.assertRaises(NativeCallbackError):failed.evaluate((1.,2.))


if __name__=='__main__':unittest.main()
