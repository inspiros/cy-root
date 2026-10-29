"""Independent vector Hessian arithmetic, ownership and installed native ABI."""
import gc
import importlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import weakref
from fractions import Fraction
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cyroot.native import HessianPlan, NativeCallbackError, VectorCallback

hessians = importlib.import_module('cyroot.native.hessians')


EXPECTED = (((2., 3.), (3., 4.)), ((10., -7.), (-7., 22.)))


def quadratic(point):
    x, y = point
    return (x*x+3*x*y+2*y*y+13., 5*x*x-7*x*y+11*y*y-17.)


def all_nan(tensor):
    return all(math.isnan(value) for plane in tensor for row in plane for value in row)


def exact_weights(nodes, derivative):
    """Solve exact polynomial moment equations, independent of native formulas."""
    count = len(nodes)
    rows = [[Fraction(node)**power for node in nodes] +
            [Fraction(math.factorial(power) if power == derivative else 0)]
            for power in range(count)]
    for column in range(count):
        pivot = next(row for row in range(column, count) if rows[row][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value/divisor for value in rows[column]]
        for row in range(count):
            if row != column:
                multiplier = rows[row][column]
                rows[row] = [a-multiplier*b for a, b in zip(rows[row], rows[column])]
    return [row[-1] for row in rows]


class HessianConfigurationTests(unittest.TestCase):
    def test_properties_copy_steps_and_account_for_full_native_evaluation(self):
        steps = [.125, -.25]
        plan = HessianPlan(2, step=steps)
        steps[:] = [10., 20.]
        self.assertEqual(plan.dimension, 2)
        self.assertEqual(plan.shape, (2, 2, 2))
        self.assertEqual(plan.scheme, 'central')
        self.assertEqual(plan.step, (.125, -.25))
        self.assertEqual(plan.storage_bytes, 16)
        self.assertEqual(plan.workspace_size, 28)
        self.assertEqual(plan.workspace_bytes, 224)
        self.assertEqual(plan.output_bytes, 64)
        self.assertEqual(plan.evaluation_bytes, (28+8+4)*8)
        self.assertEqual(plan.memory_bytes, (28+8+4+2)*8)
        self.assertEqual(plan.sample_count, 8)
        self.assertEqual(HessianPlan(2, 'forward').sample_count, 5)
        for name in ('dimension', 'scheme', 'step', 'shape', 'sample_count', 'storage_bytes',
                     'workspace_size', 'workspace_bytes', 'output_bytes', 'evaluation_bytes',
                     'memory_bytes'):
            with self.subTest(name=name), self.assertRaises((TypeError, AttributeError)):
                setattr(plan, name, None)
        scalar = HessianPlan(3, step=-.125)
        self.assertEqual(scalar.step, (-.125,)*3)
        automatic = HessianPlan(3)
        self.assertIsNone(automatic.step)
        self.assertEqual(automatic.storage_bytes, 0)
        self.assertEqual(automatic.memory_bytes, (27+48+6)*8)

    def test_dimensions_schemes_and_integer_budgets_are_validated(self):
        for dimension in (0, -1, True, np.bool_(True), 1.5, 2**31, 1 << 100):
            with self.subTest(dimension=dimension), self.assertRaises((ValueError, TypeError, OverflowError)):
                HessianPlan(dimension)
        for scheme in ('', 'unknown', 1, True, None):
            with self.subTest(scheme=scheme), self.assertRaises((TypeError, ValueError)):
                HessianPlan(2, scheme)
        for budget in (-1, True, np.bool_(False), 1.5, 1 << 100):
            with self.subTest(budget=budget), self.assertRaises((TypeError, ValueError, OverflowError)):
                HessianPlan(2, memory_bytes=budget)
        class Scheme(str):
            pass
        self.assertIs(type(HessianPlan(1, Scheme('central')).scheme), str)

    def test_combined_byte_budget_exact_boundaries_and_early_huge_orders(self):
        for n in (1, 2, 4):
            for explicit in (False, True):
                total = (n**3+2*n*n+12*n+(n if explicit else 0))*8
                options = {'step': .125} if explicit else {}
                with self.subTest(n=n, explicit=explicit):
                    self.assertEqual(HessianPlan(n, memory_bytes=total, **options).memory_bytes, total)
                    with self.assertRaises(ValueError):
                        HessianPlan(n, memory_bytes=total-1, **options)
        with self.assertRaises((ValueError, OverflowError)):
            HessianPlan(2**31-1, memory_bytes=64)

    def test_invalid_steps_include_bool_complex_text_shapes_and_formats(self):
        for step in (True, np.bool_(True), '1', b'1', 1j, 0., -0., math.nan, math.inf,
                     (), (.1,), (.1,.2,.3), (.1,0.), (.1,math.inf), (.1,True), (.1,np.bool_(True)),
                     (.1,1j), (.1,'1'), np.ones((2,1)),
                     np.ones(2, dtype=np.complex128), np.array([True,True])):
            with self.subTest(step=step), self.assertRaises((ValueError, TypeError)):
                HessianPlan(2, step=step)
        # Real non-float64 buffers convert element by element, like sequences.
        self.assertEqual(HessianPlan(2, step=np.ones(2, dtype=np.int64)).step, (1., 1.))

    def test_readonly_strided_zero_stride_and_unaligned_buffers_are_copied(self):
        readonly = np.array([.125, -.25]); readonly.flags.writeable = False
        reverse = np.array([-.25, .125])[::-1]
        zero = np.lib.stride_tricks.as_strided(np.array([.125]), (2,), (0,))
        unaligned = np.ndarray((2,), np.float64, buffer=bytearray(17), offset=1)
        unaligned[:] = [.125, -.25]
        for source in (readonly, reverse, zero, unaligned):
            with self.subTest(strides=source.strides):
                saved = tuple(source)
                plan = HessianPlan(2, step=source)
                self.assertEqual(plan.step, saved)
                assert_allclose(plan.evaluate(quadratic, (.75,-1.25)).hessian, EXPECTED, atol=1e-12)
                assert_array_equal(source, saved)

    def test_construction_guards_and_partial_objects(self):
        plan = HessianPlan.__new__(HessianPlan)
        case = self
        class Step:
            def __float__(self):
                with case.assertRaises(RuntimeError):
                    plan.__init__(2, step=.1)
                with case.assertRaises(RuntimeError):
                    plan.evaluate(quadratic, (0.,0.))
                return .125
        plan.__init__(2, step=Step())
        self.assertEqual(plan.step, (.125,.125))
        with self.assertRaises(RuntimeError):
            plan.__init__(2)
        partial = HessianPlan.__new__(HessianPlan)
        with self.assertRaises(RuntimeError):
            partial.evaluate(quadratic, (0.,0.))


class HessianNumericalTests(unittest.TestCase):
    def test_distinct_three_component_quadratics_define_tensor_layout_and_symmetry(self):
        expected = (((2.,3.,5.), (3.,4.,7.), (5.,7.,6.)),
                    ((8.,11.,13.), (11.,10.,17.), (13.,17.,12.)),
                    ((14.,19.,23.), (19.,16.,29.), (23.,29.,18.)))
        def function(point):
            return tuple(sum(plane[j][j]*point[j]*point[j]/2 for j in range(3))+
                         sum(plane[j][k]*point[j]*point[k] for j in range(3) for k in range(j+1,3))
                         for plane in expected)
        for scheme in ('central','forward','backward'):
            for steps in ((.125,-.25,.0625), (-.125,.25,-.0625)):
                with self.subTest(scheme=scheme, steps=steps):
                    result = HessianPlan(3, scheme, steps).evaluate(function, (0.,0.,0.))
                    self.assertEqual(result.status, 'ok')
                    self.assertEqual(result.hessian, expected)
                    self.assertEqual(result.f_calls, 19 if scheme == 'central' else 10)
                    for plane in result.hessian:
                        for j in range(3):
                            for k in range(3):
                                self.assertEqual(struct.pack('d',plane[j][k]), struct.pack('d',plane[k][j]))

    def test_sample_order_axis_cache_corners_and_base_reuse(self):
        point, h = (.75,-1.25), (.125,-.25)
        for scheme in ('central','forward','backward'):
            for cached in (False,True):
                samples = []
                result = HessianPlan(2, scheme, h).evaluate(
                    lambda x: samples.append(x) or quadratic(x), point,
                    f_x=quadratic(point) if cached else None)
                sign = -1 if scheme == 'backward' else 1
                first = [point[j]+sign*h[j] for j in range(2)]
                second = [point[j]-h[j] if scheme == 'central' else point[j]+sign*2*h[j] for j in range(2)]
                expected = [] if cached else [point]
                expected += [(first[0],point[1]), (second[0],point[1]),
                             (point[0],first[1]), (point[0],second[1])]
                expected += [(a,b) for a in ((first[0],second[0]) if scheme == 'central' else (first[0],))
                             for b in ((first[1],second[1]) if scheme == 'central' else (first[1],))]
                with self.subTest(scheme=scheme,cached=cached):
                    self.assertEqual(samples, expected)
                    self.assertEqual(result.f_calls, len(expected))
                    self.assertEqual(result.hessian, EXPECTED)

    def test_rounded_diagonal_matches_exact_moments_with_unequal_nodes(self):
        scale = 2**52
        for scheme, step in (('central',11*2**-55), ('forward',11*2**-55),
                             ('backward',-11*2**-55)):
            sign = -1 if scheme == 'backward' else 1
            nodes = [0., 1.+sign*step-1.,
                     1.-step-1. if scheme == 'central' else 1.+sign*2*step-1.]
            self.assertNotEqual(nodes[2], -nodes[1] if scheme == 'central' else 2*nodes[1])
            def function(point):
                u = (point[0]-1.)*scale
                return (u*u+u+8.,)
            weights = exact_weights(nodes,2)
            oracle = sum(weight*Fraction(function((1.+node,))[0]) for weight,node in zip(weights,nodes))
            result = HessianPlan(1,scheme,step).evaluate(function,(1.,))
            self.assertEqual(result.status,'ok')
            self.assertTrue(math.isclose(result.hessian[0][0][0],float(oracle),rel_tol=2e-15))
            self.assertEqual(oracle, 2*scale*scale)

    def test_unequal_central_biquadratic_mixed_weights_match_exact_tensor_moments(self):
        scale, step = 2**52, 11*2**-55
        nodes = [0.,1.+step-1.,1.-step-1.]
        self.assertNotEqual(nodes[1],-nodes[2])
        def function(point):
            u,v = ((point[j]-1.)*scale for j in range(2))
            product = (u+u*u)*(v+v*v)
            return (product, 2*product+u*u+3*v*v)
        weights = exact_weights(nodes,1)
        oracle = [sum(a*b*Fraction(function((1.+x,1.+y))[o])
                      for a,x in zip(weights,nodes) for b,y in zip(weights,nodes)) for o in range(2)]
        self.assertEqual(oracle, [scale*scale,2*scale*scale])
        result = HessianPlan(2,step=step).evaluate(function,(1.,1.))
        self.assertEqual(result.status,'ok')
        for o in range(2):
            self.assertTrue(math.isclose(result.hessian[o][0][1],float(oracle[o]),rel_tol=2e-15))
            self.assertEqual(result.hessian[o][0][1],result.hessian[o][1][0])

    def test_smooth_refinement_and_constant_offsets(self):
        point = (.3,-.4)
        function = lambda x: (math.exp(x[0]+x[1]),math.sin(x[0]+2*x[1]))
        e,s = math.exp(sum(point)),-math.sin(point[0]+2*point[1])
        expected = (((e,e),(e,e)),((s,2*s),(2*s,4*s)))
        for scheme,ratio in (('central',3.7),('forward',1.8),('backward',1.8)):
            errors = [np.max(np.abs(np.asarray(HessianPlan(2,scheme,h).evaluate(function,point).hessian)-expected))
                      for h in (.04,.02,.01)]
            with self.subTest(scheme=scheme):
                self.assertGreater(errors[0]/errors[1],ratio)
                self.assertGreater(errors[1]/errors[2],ratio)
        for offset in (0.,2**30):
            result = HessianPlan(2,step=.125).evaluate(
                lambda x: tuple(value+offset for value in quadratic(x)),(0.,0.))
            self.assertEqual(result.hessian,EXPECTED)

    def test_representable_subnormal_mixed_and_diagonal_results_survive_accumulation(self):
        minimum=math.ulp(0.)
        def function(point):
            x,y=point
            return (minimum*x*y,minimum*x*x)
        for scheme in ('central','forward','backward'):
            result=HessianPlan(2,scheme,1.).evaluate(function,(0.,0.))
            with self.subTest(scheme=scheme):
                self.assertEqual(result.status,'ok')
                self.assertEqual(result.hessian[0][0][1],minimum)
                self.assertEqual(result.hessian[0][1][0],minimum)
                self.assertEqual(result.hessian[1][0][0],2*minimum)
                for plane in result.hessian:
                    self.assertEqual(struct.pack('d',plane[0][1]),struct.pack('d',plane[1][0]))

    def test_extreme_anisotropic_scales_preserve_representable_mixed_results(self):
        for scheme in ('central', 'forward', 'backward'):
            for first_exponent, amplitude_exponent in ((1000, -400), (-1000, 400)):
                steps = (math.ldexp(1., first_exponent), math.ldexp(1., -first_exponent))
                amplitude = math.ldexp(1., amplitude_exponent)
                def residual(point):
                    return ((point[0]/steps[0])*(point[1]/steps[1])*amplitude, 0.)
                result = HessianPlan(2, scheme, steps).evaluate(residual, (0., 0.))
                with self.subTest(scheme=scheme, steps=steps):
                    self.assertEqual(result.status, 'ok')
                    self.assertEqual(result.hessian[0][0][1], amplitude)
                    self.assertEqual(result.hessian[0][1][0], amplitude)
                    self.assertEqual(result.hessian[0][0][0], 0.)
                    self.assertEqual(result.hessian[0][1][1], 0.)

    def test_automatic_steps_follow_scheme_and_coordinate_scale(self):
        point = (2**20,-.25)
        for scheme in ('central','forward','backward'):
            samples = []
            result = HessianPlan(2,scheme).evaluate(lambda x: samples.append(x) or (x[0],x[1]),point)
            self.assertEqual(result.status,'ok')
            factor = sys.float_info.epsilon**(.25 if scheme == 'central' else 1/3)
            sign = -1 if scheme == 'backward' else 1
            for j in range(2):
                requested = max(1.,abs(point[j]))*factor
                self.assertEqual(samples[1+2*j][j],point[j]+sign*requested)
                self.assertEqual(samples[2+2*j][j],point[j]-requested if scheme == 'central'
                                 else point[j]+sign*2*requested)


class HessianBoundaryTests(unittest.TestCase):
    def test_invalid_points_base_and_collapsed_stencils_do_not_sample(self):
        for point in ((math.inf,0.),(0.,math.nan)):
            result = HessianPlan(2).evaluate(lambda x:self.fail('invalid x sampled'),point)
            self.assertEqual((result.status,result.f_calls),('nonfinite',0))
            self.assertTrue(all_nan(result.hessian))
        for base in ((math.inf,0.),(0.,math.nan)):
            result = HessianPlan(2).evaluate(lambda x:self.fail('invalid base sampled'),(0.,0.),f_x=base)
            self.assertEqual((result.status,result.f_calls),('nonfinite',0))
            self.assertTrue(all_nan(result.hessian))
        for step,status in (((.1,1e-320),'breakdown'), ((.1,1e308),'nonfinite')):
            samples=[]
            result=HessianPlan(2,'forward',step).evaluate(lambda x:samples.append(x) or quadratic(x),(1.,1.))
            self.assertEqual(result.status,status)
            self.assertEqual(samples,[(1.,1.)])
            self.assertTrue(all_nan(result.hessian))
        for x,base in (((0.,),None),((0.,0.,0.),None),((0.,0.),(1.,)),((0.,0.),(1.,2.,3.))):
            with self.assertRaises((ValueError,TypeError)):
                HessianPlan(2).evaluate(quadratic,x,f_x=base)

    def test_callback_native_kind_and_dimension_are_checked_before_coercion(self):
        def unexpected(x):
            self.fail('invalid adapter reached callback')
        for adapter in (VectorCallback(unexpected,3), VectorCallback(unexpected,2,jacobian=True),
                        VectorCallback.__new__(VectorCallback)):
            with self.subTest(adapter=adapter), self.assertRaises((ValueError,TypeError,RuntimeError)):
                HessianPlan(2).evaluate(adapter,(0.,0.),f_x=(0.,0.))
        class Spoofed(HessianPlan):
            @property
            def dimension(self):
                return 2
        with self.assertRaises(ValueError):
            Spoofed(3).evaluate(VectorCallback(unexpected,2),(0.,0.),f_x=(0.,0.))

    def test_late_nonfinite_and_arithmetic_failure_poison_the_whole_tensor(self):
        for failed_call in (2,5,9):
            samples=[]
            def function(x):
                samples.append(x)
                return (math.nan,0.) if len(samples)==failed_call else quadratic(x)
            result=HessianPlan(2,step=.125).evaluate(function,(.75,-1.25))
            self.assertEqual((result.status,result.f_calls),('nonfinite',failed_call))
            self.assertTrue(all_nan(result.hessian))
        result=HessianPlan(1,step=.125).evaluate(lambda x:(-1e308 if x[0] else 1e308,),(0.,))
        self.assertEqual(result.status,'nonfinite')
        self.assertTrue(all_nan(result.hessian))

    def test_python_base_and_late_corner_exceptions_preserve_identity_traceback(self):
        for failed_call in (1,9):
            failure=KeyboardInterrupt('hessian callback failed')
            calls=0
            def residual(x):
                nonlocal calls
                calls+=1
                if calls==failed_call:
                    raise failure
                return quadratic(x)
            adapter=VectorCallback(residual,2)
            try:
                HessianPlan(2,step=.125).evaluate(adapter,(.75,-1.25))
            except KeyboardInterrupt as caught:
                self.assertIs(caught,failure)
                self.assertIn('residual',[frame.name for frame in traceback.extract_tb(caught.__traceback__)])
            else:
                self.fail('callback exception was swallowed')
            self.assertEqual(failure.__dict__,{})
            self.assertEqual(HessianPlan(2,step=.125).evaluate(adapter,(.75,-1.25)).status,'ok')

    def test_point_and_cache_coercion_and_boxing_hold_callback_reservations(self):
        plan=HessianPlan(2,step=.125)
        adapter=VectorCallback(quadratic,2)
        case=self
        attempts=[]
        class Sequence:
            def __init__(self,values):
                self.values=values
            def __len__(self):
                with case.assertRaises(RuntimeError):
                    plan.evaluate(adapter,(0.,0.))
                attempts.append(1)
                return 2
            def __getitem__(self,index):
                return self.values[index]
        result=plan.evaluate(adapter,Sequence((.75,-1.25)),f_x=Sequence(quadratic((.75,-1.25))))
        self.assertEqual(result.status,'ok')
        self.assertTrue(attempts)
        original=hessians.HessianEstimate
        def estimate(*args):
            with self.assertRaises(RuntimeError):
                plan.evaluate(adapter,(0.,0.))
            return original(*args)
        with patch.object(hessians,'HessianEstimate',estimate):
            self.assertEqual(plan.evaluate(adapter,(0.,0.)).status,'ok')

    def test_failed_cache_conversion_releases_callback_reservation(self):
        failure=ArithmeticError('invalid cached component')
        class Invalid:
            def __float__(self):
                raise failure
        adapter=VectorCallback(quadratic,2)
        plan=HessianPlan(2,step=.125)
        with self.assertRaises(ArithmeticError) as caught:
            plan.evaluate(adapter,(0.,0.),f_x=(0.,Invalid()))
        self.assertIs(caught.exception,failure)
        self.assertEqual(plan.evaluate(adapter,(0.,0.)).hessian,EXPECTED)

    def test_mutable_inputs_and_callback_outputs_are_private_snapshots(self):
        point=[.75,-1.25]
        base=list(quadratic(point))
        saved=tuple(point)
        retained=[]
        output=[0.,0.]
        def callback(x):
            retained.append(x)
            point[:]=[90.,91.]
            base[:]=[92.,93.]
            output[:]=quadratic(x)
            return output
        result=HessianPlan(2,step=.125).evaluate(callback,point,f_x=base)
        self.assertEqual(result.hessian,EXPECTED)
        self.assertTrue(all(type(x) is tuple for x in retained))
        self.assertEqual(retained[0],(saved[0]+.125,saved[1]))
        output[:]=[100.,100.]
        self.assertEqual(result.hessian,EXPECTED)
        with self.assertRaises((TypeError,AttributeError)):
            result.hessian=()

    def test_immutable_plan_allows_nested_evaluation_with_independent_callbacks(self):
        plan=HessianPlan(2,step=.125)
        nested=[]
        def outer(x):
            if not nested:
                nested.append(plan.evaluate(quadratic,(0.,0.)).hessian)
            return quadratic(x)
        self.assertEqual(plan.evaluate(outer,(.75,-1.25)).hessian,EXPECTED)
        self.assertEqual(nested,[EXPECTED])


@unittest.skipUnless(importlib.util.find_spec('Cython'),'installed consumer requires Cython')
class InstalledHessianConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-hessian-consumer-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory=Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_hessian_consumer.pyx'),directory/'native_hessian_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension, setup\nimport sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-hessian-consumer",ext_modules=cythonize(\n'
            '[Extension("native_hessian_consumer",["native_hessian_consumer.pyx"],language="c++",\n'
            'include_dirs=[get_include()],extra_compile_args=flags)],language_level=3,include_path=[get_cython_include()]))\n')
        environment=os.environ.copy(); environment.pop('PYTHONPATH',None)
        build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                             env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('Hessian external consumer build failed:\n'+build.stdout)
        sys.path.insert(0,str(directory)); cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('native_hessian_consumer')
        cls.addClassCleanup(sys.modules.pop,'native_hessian_consumer',None)

    def test_public_native_callback_context_lifetime_and_actual_nogil(self):
        self.assertIs(self.consumer.public_type(),HessianPlan)
        owner=self.consumer.ContextOwner()
        reference=weakref.ref(owner)
        adapter=owner.callback()
        del owner; gc.collect()
        self.assertIsNotNone(reference())
        result=HessianPlan(2,step=.125).evaluate(adapter,(.75,-1.25))
        self.assertEqual(result.hessian,EXPECTED)
        self.assertEqual(reference().calls,result.f_calls)
        self.assertEqual(reference().gil_held,0)
        del adapter; gc.collect()
        self.assertIsNone(reference())

    def test_native_and_pod_compute_layout_canaries_restoration_and_storage_reuse(self):
        for scheme in ('central','forward','backward'):
            plan=HessianPlan(2,scheme,(.125,-.25))
            allocation=self.consumer.plan_allocation(plan)
            self.assertNotEqual(allocation[0],0)
            for config in (False,True):
                result=self.consumer.compute_direct(plan,use_config=config,repeats=3)
                with self.subTest(scheme=scheme,config=config):
                    self.assertEqual(result['status'],0)
                    self.assertEqual(result['hessian'],EXPECTED)
                    self.assertEqual(result['flat'],(2.,10.,3.,-7.,3.,-7.,4.,22.))
                    self.assertEqual(result['calls'],3*plan.sample_count)
                    self.assertEqual(result['point'],(.75,-1.25))
                    self.assertEqual(result['restored_point'],result['point'])
                    self.assertEqual(result['guards'],(1234567.,1234567.,9876543.,9876543.))
                    self.assertTrue(result['stable'])
                    self.assertEqual(result['gil_held'],0)
            self.assertEqual(self.consumer.plan_allocation(plan),allocation)

    def test_short_capacities_and_invalid_configs_leave_spans_and_calls_untouched(self):
        plan=HessianPlan(2,step=.125)
        invalid=[{'capacity':plan.workspace_size-1},{'output_capacity':7}]
        invalid += [{'use_config':True,'config_overrides':override} for override in
                    ({'dimension':0},{'dimension':2**31},{'square':3},{'cube':7},
                     {'work':27},{'scheme':-1},{'scheme':3})]
        for options in invalid:
            with self.subTest(options=options):
                result=self.consumer.compute_direct(plan,initial_calls=7,**options)
                self.assertEqual(result['status'],2)
                self.assertEqual(result['calls'],7)
                self.assertEqual(result['callback_code'],0)
                self.assertTrue(result['untouched_output'])
                self.assertTrue(result['untouched_workspace'])

    def test_raw_failures_poison_all_outputs_and_restore_trial_point(self):
        plan=HessianPlan(2,step=.125)
        for options,status,calls in (({'fail_after':0},3,1),({'fail_after':7},3,8),
                                     ({'partial_after':7},1,8),({'base':(math.nan,0.)},1,0),
                                     ({'point':(math.inf,0.)},1,0)):
            with self.subTest(options=options):
                result=self.consumer.compute_direct(plan,**options)
                self.assertEqual(result['status'],status)
                self.assertEqual(result['calls'],calls)
                self.assertEqual(result['callback_code'],-19 if status==3 else 0)
                self.assertTrue(all_nan(result['hessian']))
                self.assertEqual(result['restored_point'],result['point'])
                self.assertEqual(result['guards'],(1234567.,1234567.,9876543.,9876543.))
                self.assertEqual(result['gil_held'],0)
        collapsed=self.consumer.compute_direct(HessianPlan(2,step=(.125,1e-320)),point=(1.,1.))
        self.assertEqual((collapsed['status'],collapsed['calls']),(2,0))
        self.assertTrue(all_nan(collapsed['hessian']))
        self.assertEqual(collapsed['restored_point'],(1.,1.))

    def test_native_failure_diagnostics_include_base_and_late_corner(self):
        for after,stage in ((0,'function'),(8,'hessian')):
            owner=self.consumer.ContextOwner(fail_after=after,failure_code=-43)
            with self.assertRaises(NativeCallbackError) as caught:
                HessianPlan(2,step=.125).evaluate(owner.callback(),(.75,-1.25))
            self.assertEqual(caught.exception.native_status,-43)
            self.assertEqual(caught.exception.callback_stage,stage)
            self.assertEqual(caught.exception.result.status,'callback_failure')
            self.assertEqual(caught.exception.result.f_calls,after+1)
            self.assertEqual(caught.exception.result.callback_code,-43)
            self.assertTrue(all_nan(caught.exception.result.hessian))
        owner=self.consumer.ContextOwner(partial_after=8)
        result=HessianPlan(2,step=.125).evaluate(owner.callback(),(.75,-1.25))
        self.assertEqual((result.status,result.f_calls),('nonfinite',9))
        self.assertTrue(all_nan(result.hessian))

    def test_actual_thread_progress_callback_guard_and_concurrent_plan_reuse(self):
        plan=HessianPlan(2,step=.125)
        owner=self.consumer.WaitingOwner()
        adapter=owner.callback()
        results,failures=[],[]
        def evaluate():
            try:
                results.append(plan.evaluate(adapter,(.75,-1.25),f_x=quadratic((.75,-1.25))))
            except BaseException as error:
                failures.append(error)
        worker=threading.Thread(target=evaluate)
        worker.start()
        try:
            deadline=time.monotonic()+4.
            while not owner.entered and time.monotonic()<deadline:
                time.sleep(.001)
            self.assertTrue(owner.entered)
            self.assertTrue(worker.is_alive())
            with self.assertRaises(RuntimeError):
                plan.evaluate(adapter,(0.,0.))
            independent=plan.evaluate(quadratic,(0.,0.))
            self.assertEqual(independent.hessian,EXPECTED)
        finally:
            owner.release(); worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures,[])
        self.assertEqual(results[0].hessian,EXPECTED)


if __name__=='__main__':
    unittest.main()
