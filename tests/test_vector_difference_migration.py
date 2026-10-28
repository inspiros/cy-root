"""Compatibility boundary of the generalized native vector differences."""
import importlib
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from cyroot import GeneralizedFiniteDifference, generalized_finite_difference
from cyroot.fptr import PyNdArrayFPtr
from cyroot.native import (NativeCallbackError, RectangularCallback,
                           TensorDifferencePlan, VectorCallback)
import test_native_higher_mixed as mixed_fixture


def polynomial(point, m=3):
    x,y=point
    value=x**3+2*x*x*y+3*x*y*y+5*y**3
    return tuple(11.+o+(o+1)*value for o in range(m))


def expected_central(order, m):
    # Derivatives of the cubic at (.5,1), including the central first
    # difference's exact h^2 truncation term for h=(.25,-.25).
    if order == 1:
        entries=(5.75+.25**2,18.5+5*.25**2)
    elif order == 2:
        entries=(7.,8.,8.,33.)
    else:
        entries=tuple((6.,4.,6.,30.)[sum(axes)]
                      for axes in itertools.product(range(2),repeat=3))
    return np.asarray([[(o+1)*value for value in entries] for o in range(m)]).reshape(
        (m,)+(2,)*order)


class VectorDifferenceMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mixed_fixture.HigherMixedDifferenceTests.setUpClass.__func__(cls)
        cls.native=cls.consumer
        cls.legacy_workspace=tempfile.TemporaryDirectory(prefix='cyroot-vector-difference-')
        cls.addClassCleanup(cls.legacy_workspace.cleanup)
        directory=Path(cls.legacy_workspace.name)
        name='vector_difference_consumer'
        (directory/(name+'.pyx')).write_bytes(Path(__file__).with_name(name+'.pyx').read_bytes())
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\n'
            'from Cython.Build import cythonize\n'
            'import numpy as np\n'
            'import sys\n'
            'flags=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'setup(name="vector-difference-consumer",ext_modules=cythonize(\n'
            ' [Extension("vector_difference_consumer",["vector_difference_consumer.pyx"],\n'
            ' language="c++",include_dirs=[get_include(),np.get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        environment=os.environ.copy()
        environment.pop('PYTHONPATH',None)
        build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                             env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('Legacy derivative consumer failed:\n'+build.stdout)
        sys.path.insert(0,str(directory))
        cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module(name)
        cls.addClassCleanup(sys.modules.pop,name,None)

    def test_native_orders_shapes_cached_counts_and_released_gil(self):
        for m in (1,2,3):
            for order,samples in ((1,4),(2,8),(3,20)):
                if order == 3:
                    samples=TensorDifferencePlan(2,m,3,step=(.25,-.25)).sample_count
                for cached in (False,True):
                    with self.subTest(m=m,order=order,cached=cached):
                        owner=self.native.ContextOwner(m=m)
                        callback=owner.callback()
                        options=dict(h=(.25,-.25),order=order)
                        answer=generalized_finite_difference(
                            callback,(.5,1.),F_x=polynomial((.5,1.),m) if cached else None,
                            **options)
                        assert_allclose(answer,expected_central(order,m),atol=1e-10,rtol=1e-12)
                        self.assertEqual(answer.shape,(m,)+(2,)*order)
                        self.assertEqual(answer.dtype,np.dtype('float64'))
                        self.assertTrue(answer.flags.c_contiguous)
                        self.assertEqual((owner.calls,owner.gil_held),(samples+(not cached),0))
                        derivative=GeneralizedFiniteDifference(callback,**options)
                        assert_allclose(derivative((.5,1.)),answer,atol=1e-10,rtol=1e-12)
                        self.assertEqual(derivative.n_f_calls,1)
                        self.assertEqual(owner.gil_held,0)
        square=VectorCallback(lambda x:polynomial(x,2),2)
        assert_allclose(generalized_finite_difference(square,(.5,1.),h=(.25,-.25),order=3),
                        expected_central(3,2),atol=1e-10)
        module=importlib.import_module('cyroot.vector_derivative_approximation')
        owner=self.native.ContextOwner(m=3)
        wrapper=GeneralizedFiniteDifference(owner.callback(),h=(.25,-.25),order=3)
        with patch.object(module,'TensorDifferencePlan',wraps=module.TensorDifferencePlan) as plans:
            for point in ((.5,1.),(1.,1.5)):
                assert_allclose(wrapper(point),expected_central(3,3),atol=1e-10)
            assert_allclose(wrapper.eval_with_f_val(np.array((.5,1.)),
                np.array(polynomial((.5,1.),3))),expected_central(3,3),atol=1e-10)
            self.assertEqual(plans.call_count,1)
            GeneralizedFiniteDifference(owner.callback(),h=(.25,-.25),order=3)((.5,1.))
            for _ in range(2):
                generalized_finite_difference(owner.callback(),(.5,1.),h=(.25,-.25),order=3)
            self.assertEqual(plans.call_count,4)
        self.assertEqual(owner.gil_held,0)

    def test_original_adapter_reservation_error_identity_and_native_status(self):
        case=self
        for square in (False,True):
            function=lambda x:polynomial(x,2)
            adapter=VectorCallback(function,2) if square else RectangularCallback(function,2,2)
            attempts=[]
            def reserved():
                with case.assertRaises(RuntimeError):
                    generalized_finite_difference(adapter,(.5,1.),h=.25,order=3)
                attempts.append(True)
            class Point:
                def __array__(self,dtype=None,copy=None):
                    reserved()
                    return np.array((.5,1.),dtype=dtype)
            generalized_finite_difference(adapter,Point(),h=.25,order=3)
            self.assertTrue(attempts)
            module=importlib.import_module('cyroot.vector_derivative_approximation')
            original=module.np.asarray
            boxed=[]
            def boxing(value,*args,**kwargs):
                if isinstance(value,tuple) and value and isinstance(value[0],tuple):
                    reserved()
                    boxed.append(True)
                return original(value,*args,**kwargs)
            with patch.object(module.np,'asarray',boxing):
                generalized_finite_difference(adapter,(.5,1.),h=.25,order=3)
            self.assertTrue(boxed)
            export_failure=KeyboardInterrupt('export')
            def failed_boxing(value,*args,**kwargs):
                if isinstance(value,tuple) and value and isinstance(value[0],tuple):
                    reserved()
                    raise export_failure
                return original(value,*args,**kwargs)
            with patch.object(module.np,'asarray',failed_boxing):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    generalized_finite_difference(adapter,(.5,1.),h=.25,order=3)
            self.assertIs(caught.exception,export_failure)
            failure=KeyboardInterrupt('coercion')
            class BadPoint:
                def __array__(self,dtype=None,copy=None):
                    reserved()
                    raise failure
            with self.assertRaises(KeyboardInterrupt) as caught:
                generalized_finite_difference(adapter,BadPoint(),order=3)
            self.assertIs(caught.exception,failure)
            for options in ({'h':0.},{'F_x':(1.,)},{'order':32}):
                with self.assertRaises((ValueError,TypeError)):
                    generalized_finite_difference(adapter,(.5,1.),**options)
            # Reservation is released even when setup or output conversion fails.
            generalized_finite_difference(adapter,(.5,1.),h=.25,order=3)
            failure=KeyboardInterrupt('original residual')
            def exploding(point):
                if tuple(point) != (.5,1.):raise failure
                return polynomial(point,2)
            bad=VectorCallback(exploding,2) if square else RectangularCallback(exploding,2,2)
            with self.assertRaises(KeyboardInterrupt) as caught:
                generalized_finite_difference(bad,(.5,1.),h=.25,order=3)
            self.assertIs(caught.exception,failure)
        for order in (1,2,3):
            for after,stage in ((0,'function'),(1,{1:'jacobian',2:'hessian',3:'derivative'}[order])):
                owner=self.native.ContextOwner(fail_after=after,failure_code=-67)
                with self.assertRaises(NativeCallbackError) as caught:
                    generalized_finite_difference(owner.callback(),(.5,1.),h=.25,order=order)
                error=caught.exception
                self.assertEqual((error.native_status,error.callback_stage),(-67,stage))
                self.assertEqual((error.result.status,error.result.f_calls),
                                 ('callback_failure',after+1))
                self.assertEqual(owner.gil_held,0)

        # An unsuccessful new configuration must not replace the last good plan.
        module=importlib.import_module('cyroot.vector_derivative_approximation')
        fail_sample=[False]
        sample_error=KeyboardInterrupt('sample during new plan')
        def sometimes_exploding(point):
            if fail_sample[0] and tuple(point) != (.5,1.):raise sample_error
            return polynomial(point,2)
        wrapper=GeneralizedFiniteDifference(VectorCallback(sometimes_exploding,2),h=.25)
        with patch.object(module,'RectangularJacobianPlan',wraps=module.RectangularJacobianPlan) as jacobians, \
             patch.object(module,'RectangularHessianPlan',wraps=module.RectangularHessianPlan) as hessians, \
             patch.object(module,'TensorDifferencePlan',wraps=module.TensorDifferencePlan) as tensors:
            wrapper((.5,1.))
            wrapper.order=2
            fail_sample[0]=True
            with self.assertRaises(KeyboardInterrupt) as caught:wrapper((.5,1.))
            self.assertIs(caught.exception,sample_error)
            fail_sample[0]=False
            wrapper.order=1
            wrapper((.5,1.))
            self.assertEqual(jacobians.call_count,1)
            wrapper.order=2
            wrapper((.5,1.))
            self.assertEqual(hessians.call_count,2)

            wrapper.order=3
            original=module.np.asarray
            export_error=KeyboardInterrupt('new plan export')
            def fail_export(value,*args,**kwargs):
                if isinstance(value,tuple) and value and isinstance(value[0],tuple):
                    raise export_error
                return original(value,*args,**kwargs)
            with patch.object(module.np,'asarray',fail_export):
                with self.assertRaises(KeyboardInterrupt) as caught:wrapper((.5,1.))
            self.assertIs(caught.exception,export_error)
            wrapper.order=2
            wrapper((.5,1.))
            self.assertEqual(hessians.call_count,2)
            wrapper.order=3
            wrapper((.5,1.))
            self.assertEqual(tensors.call_count,2)

            wrapper.order=2
            with patch.object(module,'_MAX_WORK_BYTES',1):
                with self.assertRaises(ValueError):wrapper((.5,1.))
            wrapper.order=3
            wrapper((.5,1.))
            self.assertEqual(tensors.call_count,2)
            wrapper.order=2
            wrapper((.5,1.))
            self.assertEqual(hessians.call_count,4)

    def test_legacy_callbacks_mutable_options_aliases_and_counts(self):
        for kind in ('central',0,'forward',1,'backward',-1):
            seen=[]
            callback=PyNdArrayFPtr(lambda x:seen.append(tuple(x)) or polynomial(x,1))
            wrapper=GeneralizedFiniteDifference(callback,h=.25,order=3,kind=kind)
            result=wrapper((.5,1.))
            assert_allclose(result,expected_central(3,1),atol=1e-10)
            self.assertEqual(wrapper.n_f_calls,1)
            self.assertEqual(callback.n_f_calls,len(seen))
            self.assertEqual(len(seen),len(set(seen)))
            count=len(seen)
            seen.clear()
            assert_allclose(wrapper.eval_with_f_val(np.array((.5,1.)),
                np.array(polynomial((.5,1.),1))),result,atol=1e-10)
            self.assertEqual(len(seen),count-1)
            self.assertNotIn((.5,1.),seen)
            self.assertEqual((wrapper.n_f_calls,callback.n_f_calls),(2,2*count-1))
            wrapper.order,wrapper.kind,wrapper.h=2,0,(.25,-.25)
            assert_allclose(wrapper((.5,1.)),expected_central(2,1),atol=1e-10)
            for attribute,invalid in (('h',0.),('h',(1.,)),('order',0),('kind',7)):
                previous=getattr(wrapper,attribute)
                setattr(wrapper,attribute,invalid)
                before=callback.n_f_calls
                with self.assertRaises((ValueError,TypeError)):
                    wrapper((.5,1.))
                self.assertEqual(callback.n_f_calls,before)
                setattr(wrapper,attribute,previous)
            assert_allclose(wrapper((.5,1.)),expected_central(2,1),atol=1e-10)

        module=importlib.import_module('cyroot.vector_derivative_approximation')
        outputs=[1]
        def changing_shape(point):
            return np.full(outputs[0],sum(value**3 for value in point))
        wrapper=GeneralizedFiniteDifference(changing_shape,h=(.25,.5))
        with patch.object(module,'RectangularJacobianPlan',wraps=module.RectangularJacobianPlan) as jacobians, \
             patch.object(module,'RectangularHessianPlan',wraps=module.RectangularHessianPlan) as hessians, \
             patch.object(module,'TensorDifferencePlan',wraps=module.TensorDifferencePlan) as tensors:
            assert_allclose(wrapper((.5,1.)),((.8125,3.25),))
            assert_allclose(wrapper((1.,2.)),((3.0625,12.25),))
            self.assertEqual(jacobians.call_count,1)
            wrapper.h[0]=.5
            assert_allclose(wrapper((.5,1.)),((1.,3.25),))
            self.assertEqual(jacobians.call_count,2)
            wrapper.kind=1
            assert_allclose(wrapper((.5,1.)),((1.75,4.75),))
            self.assertEqual(jacobians.call_count,3)
            wrapper.order=2
            self.assertEqual(wrapper((.5,1.)).shape,(1,2,2))
            self.assertEqual(hessians.call_count,1)
            wrapper.order=3
            result=wrapper((.5,1.))
            expected=np.zeros((1,2,2,2))
            expected[0,0,0,0]=expected[0,1,1,1]=6.
            assert_allclose(result,expected,atol=1e-10)
            self.assertEqual(tensors.call_count,1)
            wrapper.h=.25
            self.assertEqual(wrapper((.5,)).shape,(1,1,1,1))
            outputs[0]=2
            assert_allclose(wrapper((.5,)),np.full((2,1,1,1),6.),atol=1e-10)
            self.assertEqual(tensors.call_count,3)
            with patch.object(module,'_MAX_STENCIL_TERMS',module._MAX_STENCIL_TERMS+1):
                wrapper((.5,))
                self.assertEqual(tensors.call_count,4)
                with patch.object(module,'_MAX_WORK_BYTES',module._MAX_WORK_BYTES+8):
                    wrapper((.5,))
                    self.assertEqual(tensors.call_count,5)

    def test_retained_cython_kernel_eval_eval2_and_base_class_abi(self):
        point=np.array((.5,1.))
        base=np.array(polynomial(point,3))
        steps=np.array((.25,-.25))
        for order in (1,2,3):
            callback=PyNdArrayFPtr(lambda x:polynomial(x,3))
            result=self.consumer.kernel(callback,point,base,steps,order,0)
            assert_allclose(result,expected_central(order,3),atol=1e-10)
            count=callback.n_f_calls
            module=importlib.import_module('cyroot.vector_derivative_approximation')
            plan_name={1:'RectangularJacobianPlan',2:'RectangularHessianPlan',3:'TensorDifferencePlan'}[order]
            with patch.object(module,plan_name,wraps=getattr(module,plan_name)) as plans:
                uncached,cached,exposed,calls=self.consumer.compiled_evals(
                    callback,point,base,steps,order,0)
                self.assertEqual(plans.call_count,1)
            self.assertEqual(calls,3)
            self.assertEqual(callback.n_f_calls,4*count+1)
            for answer in (uncached,cached,exposed):
                assert_allclose(answer,result,atol=1e-10)
                self.assertTrue(answer.flags.c_contiguous)
            np.testing.assert_array_equal(point,(.5,1.))
            np.testing.assert_array_equal(base,polynomial(point,3))


if __name__ == '__main__':
    unittest.main()
