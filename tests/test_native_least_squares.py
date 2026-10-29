"""Independent rectangular/rank-aware solve identities and installed Cython API."""
from concurrent.futures import ThreadPoolExecutor
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
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from cyroot.native import LeastSquaresPlan, LeastSquaresResult


TALL=((1.,0.),(0.,1.),(1.,1.))
WIDE=((1.,0.,1.),(0.,1.,1.))


def multiply(matrix, point):
    return tuple(sum(a*x for a,x in zip(row,point)) for row in matrix)


def assert_solution(test, answer, expected, rank):
    test.assertIsInstance(answer,LeastSquaresResult)
    test.assertEqual((answer.rank,answer.linear_status,answer.status),(rank,0,'ok'))
    test.assertIsInstance(answer.solution,tuple)
    assert_allclose(answer.solution,expected,rtol=2e-13,atol=2e-14)


class LeastSquaresTests(unittest.TestCase):
    def test_tall_consistent_exact_solution(self):
        assert_solution(self,LeastSquaresPlan(3,2).solve(TALL,(1.,2.,3.)),(1.,2.),2)

    def test_tall_inconsistent_solution_and_normal_residual_identity(self):
        answer=LeastSquaresPlan(3,2).solve(TALL,(1.,2.,4.))
        assert_solution(self,answer,(4/3,7/3),2)
        residual=tuple(x-y for x,y in zip(multiply(TALL,answer.solution),(1.,2.,4.)))
        for j in range(2):self.assertAlmostEqual(sum(row[j]*r for row,r in zip(TALL,residual)),0.)
        self.assertGreater(sum(r*r for r in residual),0.)

    def test_wide_minimum_norm_and_nullspace_orthogonality(self):
        answer=LeastSquaresPlan(2,3).solve(WIDE,(1.,2.))
        assert_solution(self,answer,(0.,1.,1.),2)
        assert_allclose(multiply(WIDE,answer.solution),(1.,2.),atol=1e-14)
        self.assertAlmostEqual(-answer.solution[0]-answer.solution[1]+answer.solution[2],0.)

    def test_wide_two_dimensional_nullspace(self):
        matrix=((1.,0.,1.,0.),(0.,1.,0.,1.))
        answer=LeastSquaresPlan(2,4).solve(matrix,(2.,4.))
        assert_solution(self,answer,(1.,2.,1.,2.),2)
        self.assertAlmostEqual(answer.solution[0]-answer.solution[2],0.)
        self.assertAlmostEqual(answer.solution[1]-answer.solution[3],0.)

    def test_rank_deficient_inconsistent_minimum_norm(self):
        matrix=((1.,2.),(2.,4.),(3.,6.))
        answer=LeastSquaresPlan(3,2).solve(matrix,(1.,0.,1.))
        assert_solution(self,answer,(2/35,4/35),1)
        self.assertAlmostEqual(2*answer.solution[0]-answer.solution[1],0.)
        residual=tuple(x-y for x,y in zip(multiply(matrix,answer.solution),(1.,0.,1.)))
        self.assertAlmostEqual(sum(row[0]*r for row,r in zip(matrix,residual)),0.)

    def test_zero_matrix_rank_zero_and_zero_minimum_norm(self):
        for m,n in ((3,2),(2,3),(1,1)):
            assert_solution(self,LeastSquaresPlan(m,n).solve([[0.]*n for _ in range(m)],[3.]*m),[0.]*n,0)

    def test_nonsymmetric_matrix_layout_and_pivot_permutation(self):
        matrix=((0.,2.),(1.,3.),(2.,-1.))
        assert_solution(self,LeastSquaresPlan(3,2).solve(matrix,(-4.,-3.,8.)),(3.,-2.),2)

    def test_square_nonsymmetric_solution(self):
        assert_solution(self,LeastSquaresPlan(2,2).solve(((0.,2.),(1.,3.)),(4.,7.)),(1.,2.),2)

    def test_single_row_and_single_column(self):
        assert_solution(self,LeastSquaresPlan(1,2).solve(((3.,4.),),(10.,)),(1.2,1.6),1)
        assert_solution(self,LeastSquaresPlan(2,1).solve(((3.,),(4.,)),(6.,8.)),(2.,),1)

    def test_rcond_controls_estimated_rank(self):
        matrix=((1.,0.),(0.,1e-8))
        assert_solution(self,LeastSquaresPlan(2,2,rcond=1e-12).solve(matrix,(1.,1e-8)),(1.,1.),2)
        assert_solution(self,LeastSquaresPlan(2,2,rcond=1e-5).solve(matrix,(1.,1e-8)),(1.,0.),1)

    def test_global_scaling_preserves_solution_and_rank(self):
        for scale in (1e-150,1e150):
            a=tuple(tuple(scale*x for x in row) for row in TALL)
            assert_solution(self,LeastSquaresPlan(3,2).solve(a,tuple(scale*x for x in (1.,2.,4.))),
                            (4/3,7/3),2)

    def test_native_double_strides_readonly_and_unaligned_inputs(self):
        source=np.array([[99.,1.,0.,99.],[99.,0.,1.,99.],[99.,1.,1.,99.]])
        matrix=source[:,1:3];rhs=np.array([99.,1.,99.,2.,99.,4.])[1::2]
        matrix.flags.writeable=False;rhs.flags.writeable=False
        before=source.copy()
        assert_solution(self,LeastSquaresPlan(3,2).solve(matrix,rhs),(4/3,7/3),2)
        np.testing.assert_array_equal(source,before)
        assert_solution(self,LeastSquaresPlan(3,2).solve(matrix[::-1,::-1],rhs[::-1]),(7/3,4/3),2)
        unaligned=np.ndarray((3,2),dtype='d',buffer=bytearray(49),offset=1)
        unaligned[:]=TALL
        assert_solution(self,LeastSquaresPlan(3,2).solve(unaligned,rhs),(4/3,7/3),2)

    def test_fortran_contiguous_and_memoryview_inputs(self):
        matrix=np.asfortranarray(TALL);rhs=np.array((1.,2.,3.))
        assert_solution(self,LeastSquaresPlan(3,2).solve(memoryview(matrix),memoryview(rhs)),(1.,2.),2)

    def test_input_arrays_and_retained_results_are_independent(self):
        matrix=np.array(TALL);rhs=np.array((1.,2.,3.));plan=LeastSquaresPlan(3,2)
        answer=plan.solve(matrix,rhs);matrix[:]=0.;rhs[:]=99.
        assert_solution(self,answer,(1.,2.),2)
        plan.solve(TALL,(4.,5.,9.));assert_solution(self,answer,(1.,2.),2)

    def test_exact_shapes_and_text_rejected(self):
        plan=LeastSquaresPlan(3,2)
        for matrix,rhs in ((((1.,2.),),[1.]*3),([[1.]*3]*3,[1.]*3),
                           ([[1.,2.],[1.],[1.,2.]],[1.]*3),(TALL,[1.,2.]),
                           (TALL,[[1.],[2.],[3.]]),('123',[1.]*3),(TALL,'123'),
                           ((('1',0.),(0.,1.),(1.,1.)),[1.]*3)):
            with self.subTest(matrix=matrix,rhs=rhs):
                with self.assertRaises((ValueError,TypeError)):plan.solve(matrix,rhs)

    def test_real_buffer_dtypes_convert_and_complex_is_rejected(self):
        plan=LeastSquaresPlan(3,2)
        expected=plan.solve(TALL,[1.]*3)
        for dtype in ('f4','i8','>f8'):
            self.assertEqual(plan.solve(np.array(TALL,dtype=dtype),[1.]*3),expected)
            self.assertEqual(plan.solve(TALL,np.array([1.]*3,dtype=dtype)),expected)
        for dtype in ('c16','?'):
            with self.assertRaises(TypeError):plan.solve(np.array(TALL,dtype=dtype),[1.]*3)
            with self.assertRaises(TypeError):plan.solve(TALL,np.array([1.]*3,dtype=dtype))

    def test_nonfinite_inputs_have_atomic_nan_result_and_no_rank(self):
        plan=LeastSquaresPlan(3,2)
        for value in (math.nan,math.inf,-math.inf):
            for a,b in ((((value,0.),(0.,1.),(1.,1.)),(1.,2.,3.)),(TALL,(1.,value,3.))):
                answer=plan.solve(a,b)
                self.assertEqual((answer.rank,answer.linear_status,answer.status),(None,0,'nonfinite'))
                self.assertTrue(all(math.isnan(x) for x in answer.solution))

    def test_finite_inputs_with_unrepresentable_factors_have_no_solution(self):
        # Both exact solutions exist, but sqrt(2)*1.3e308 cannot be stored in
        # the rescaled QR factor. This primitive conservatively rejects it.
        huge=1.3e308
        for matrix,rhs,m,n in ((((huge,),(huge,)),(huge,huge),2,1),
                                (((huge,huge),),(huge,),1,2)):
            answer=LeastSquaresPlan(m,n).solve(matrix,rhs)
            self.assertEqual((answer.status,answer.rank,answer.linear_status),('nonfinite',None,0))
            self.assertTrue(all(math.isnan(x) for x in answer.solution))

class LeastSquaresPlanValidationTests(unittest.TestCase):
    def test_default_properties_and_exact_managed_memory_budget(self):
        plan=LeastSquaresPlan(3,2)
        self.assertEqual((plan.rows,plan.columns,plan.rcond),(3,2,1e-12))
        minimum=max(2+3*2+1,2*2+1)
        self.assertGreaterEqual(plan.lwork,minimum)
        self.assertEqual(plan.work_size,6+3+plan.lwork)
        self.assertEqual(plan.pivot_size,2)
        expected=(6+3+2+plan.work_size)*8+2*struct.calcsize('i')
        self.assertEqual(plan.memory_bytes,expected)
        self.assertEqual(LeastSquaresPlan(3,2,workspace_bytes=expected).memory_bytes,expected)
        with self.assertRaises(ValueError):LeastSquaresPlan(3,2,workspace_bytes=expected-1)

    def test_positive_int_dimensions_and_abi_bounds(self):
        for value in (0,-1,2**31,2**64,True,1.5,'2'):
            for m,n in ((value,2),(2,value)):
                with self.assertRaises((ValueError,TypeError,OverflowError)):LeastSquaresPlan(m,n)
        self.assertEqual(LeastSquaresPlan(np.int64(2),np.int32(3)).columns,3)
        # Reject before DGELSY's LP64 optimal-query integer can wrap positive.
        for m,n in ((1,440_000_000),(440_000_000,1),(1,32768),(32768,1)):
            with self.assertRaises(ValueError):LeastSquaresPlan(m,n,workspace_bytes=1<<45)
        self.assertEqual(LeastSquaresPlan(1,32767,workspace_bytes=1<<30).columns,32767)


    def test_huge_checked_products_fail_without_allocating(self):
        for m,n in ((2**30,2**30),(1,2**30),(2**30,1)):
            with self.assertRaises((ValueError,OverflowError)):LeastSquaresPlan(m,n,workspace_bytes=0)

    def test_rcond_finite_strict_unit_interval(self):
        for value in (0.,-1.,1.,2.,math.nan,math.inf,-math.inf):
            with self.assertRaises(ValueError):LeastSquaresPlan(2,2,rcond=value)
        for value in (True,np.bool_(True),1+0j,np.array([.1]),'.1'):
            with self.assertRaises((TypeError,ValueError)):LeastSquaresPlan(2,2,rcond=value)

    def test_workspace_budget_validated(self):
        for value in (-1,True,1.5,'10000'):
            with self.assertRaises((TypeError,ValueError,OverflowError)):LeastSquaresPlan(2,2,workspace_bytes=value)

    def test_configuration_readonly_and_reinitialization_rejected(self):
        plan=LeastSquaresPlan(2,3)
        for attr in ('rows','columns','rcond','lwork','work_size','pivot_size','memory_bytes'):
            with self.assertRaises((AttributeError,TypeError)):setattr(plan,attr,99)
        with self.assertRaises(RuntimeError):LeastSquaresPlan.__init__(plan,3,2)
        assert_solution(self,plan.solve(WIDE,(1.,2.)),(0.,1.,1.),2)

    def test_new_only_plan_rejected_by_public_solve(self):
        with self.assertRaises(RuntimeError):LeastSquaresPlan.__new__(LeastSquaresPlan).solve(TALL,(1.,2.,3.))

    def test_constructor_guards_recursive_coercion_and_can_retry_after_failure(self):
        plan=LeastSquaresPlan.__new__(LeastSquaresPlan);outer=self
        class Dimension:
            def __index__(self):
                with outer.assertRaises(RuntimeError):LeastSquaresPlan.__init__(plan,2,2)
                return 2
        LeastSquaresPlan.__init__(plan,Dimension(),2)
        self.assertEqual(plan.rows,2)
        failed=LeastSquaresPlan.__new__(LeastSquaresPlan)
        with self.assertRaises(ValueError):LeastSquaresPlan.__init__(failed,0,2)
        LeastSquaresPlan.__init__(failed,2,2);self.assertEqual(failed.columns,2)

    def test_native_fields_do_not_use_subclass_property_overrides(self):
        class Plan(LeastSquaresPlan):
            @property
            def rows(self):raise AssertionError('Python rows override')
            @property
            def columns(self):raise AssertionError('Python columns override')
            @property
            def rcond(self):raise AssertionError('Python threshold override')
        assert_solution(self,Plan(3,2).solve(TALL,(1.,2.,3.)),(1.,2.),2)

    def test_result_boxing_exception_does_not_poison_reuse(self):
        plan=LeastSquaresPlan(3,2);error=KeyboardInterrupt('result allocation')
        module=importlib.import_module('cyroot.native.least_squares')
        with patch.object(module,'LeastSquaresResult',side_effect=error):
            with self.assertRaises(KeyboardInterrupt) as caught:plan.solve(TALL,(1.,2.,3.))
        self.assertIs(caught.exception,error)
        assert_solution(self,plan.solve(TALL,(1.,2.,3.)),(1.,2.),2)

    def test_workspace_query_validator_range_and_backend_gates(self):
        validate=importlib.import_module('cyroot.native.least_squares')._validate_workspace_query
        self.assertEqual(validate(32.,0,10),32)
        for value,info in ((math.nan,0),(math.inf,0),(-1.,0),(0.,0),(9.,0),(10.5,0),(2**31,0),(32.,-1),(32.,1)):
            with self.subTest(value=value,info=info):
                with self.assertRaises((ValueError,RuntimeError,OverflowError)):validate(value,info,10)


@unittest.skipUnless(importlib.util.find_spec('Cython'),'consumer requires Cython')
class NativeLeastSquaresConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-least-squares-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup);temp=Path(cls.workspace.name)
        shutil.copy2(Path(__file__).with_name('native_least_squares_consumer.pyx'),temp/'native_least_squares_consumer.pyx')
        (temp/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="least-squares-consumer",ext_modules=cythonize([Extension("native_least_squares_consumer",\n'
            ' ["native_least_squares_consumer.pyx"],language="c++",include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=temp,env=env,text=True,capture_output=True)
        if result.returncode:raise AssertionError(result.stdout+result.stderr)
        sys.path.insert(0,str(temp));cls.addClassCleanup(sys.path.remove,str(temp))
        cls.consumer=importlib.import_module('native_least_squares_consumer')
        cls.addClassCleanup(sys.modules.pop,'native_least_squares_consumer',None)

    def test_raw_layout_pivots_no_gil_and_input_preservation(self):
        matrix=((0.,2.),(1.,3.),(2.,-1.));rhs=(-4.,-3.,8.)
        answer=self.consumer.compute(LeastSquaresPlan(3,2),matrix,rhs,repeat=20)
        self.assertEqual(answer[:3],(0,2,0));assert_allclose(answer[3],(3.,-2.),atol=1e-14)
        self.assertEqual(answer[4:6],((987654.,987654.),0))
        self.assertEqual(answer[6:],(matrix,rhs))

    def test_wide_dirty_rhs_padding_and_pivots_reset_between_calls(self):
        values=((1.,2.),(0.,0.),(3.,-1.),(1.,2.))
        answers=self.consumer.reuse(LeastSquaresPlan(2,3),WIDE,values)
        for answer,b in zip(answers,values):
            # x0=(2b0-b1)/3, x1=(2b1-b0)/3, x2=(b0+b1)/3.
            expected=((2*b[0]-b[1])/3,(2*b[1]-b[0])/3,(b[0]+b[1])/3)
            self.assertEqual(answer[:3],(0,2,0));assert_allclose(answer[3],expected,atol=2e-14)

    def test_nonzero_pivot_flags_do_not_force_leading_zero_column(self):
        # Every JPVT is initially nonzero in the consumer. Leaving these flags
        # in place prevents pivoting the zero first column out of the lead.
        answer=self.consumer.compute(LeastSquaresPlan(3,2),((0.,1.),(0.,2.),(0.,3.)),(1.,2.,3.))
        self.assertEqual(answer[:3],(0,1,0));assert_allclose(answer[3],(0.,1.),atol=1e-14)

    def test_raw_nonfinite_failure_is_transactional(self):
        for matrix,rhs in ((((math.nan,0.),(0.,1.),(1.,1.)),(1.,2.,3.)),(TALL,(1.,math.inf,3.))):
            answer=self.consumer.compute(LeastSquaresPlan(3,2),matrix,rhs)
            self.assertEqual(answer[:3],(1,-1,0))
            self.assertEqual(answer[3],(987654.,987654.));self.assertEqual(answer[4],(987654.,987654.))
        # Backend-computed factor overflow must preserve output just as an
        # input rejected before DGELSY does.
        answer=self.consumer.compute(LeastSquaresPlan(2,1),((1.3e308,),(1.3e308,)),(1.3e308,1.3e308))
        self.assertEqual(answer[:3],(1,-1,0));self.assertEqual(answer[3],(987654.,))
        self.assertEqual(answer[4],(987654.,987654.))

    def test_raw_uninitialized_plan_keeps_output_untouched(self):
        answer=self.consumer.compute(LeastSquaresPlan.__new__(LeastSquaresPlan),TALL,(1.,2.,3.))
        self.assertEqual(answer[:3],(3,-1,0));self.assertEqual(answer[3],(987654.,987654.))

    def test_raw_null_spans_fail_without_touching_output(self):
        for null in ('matrix','rhs','output','scratch','pivots','rank','info'):
            answer=self.consumer.compute(LeastSquaresPlan(3,2),TALL,(1.,2.,3.),null=null)
            self.assertEqual(answer[0],3)
            self.assertEqual(answer[3],(987654.,987654.))
            self.assertEqual(answer[4],(987654.,987654.))

    def test_shared_immutable_plan_distinct_scratch_in_threads(self):
        plan=LeastSquaresPlan(2,3)
        def worker(rhs):return self.consumer.compute(plan,WIDE,rhs,repeat=1000)
        with ThreadPoolExecutor(max_workers=2) as workers:
            answers=list(workers.map(worker,((1.,2.),(3.,-1.))))
        for answer,expected in zip(answers,((0.,1.,1.),(7/3,-5/3,2/3))):
            self.assertEqual(answer[:3],(0,2,0));self.assertEqual(answer[5],0)
            assert_allclose(answer[3],expected,atol=2e-14)


if __name__=='__main__':unittest.main()
