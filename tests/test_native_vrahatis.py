"""Native vector bisection: signs, geometry, bounded state and compiled callbacks."""
import gc
import importlib
import itertools
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import weakref

import numpy as np
from numpy.testing import assert_allclose
from cyroot import vrahatis
from cyroot.native import NativeCallbackError, VectorCallback, RectangularCallback
# Import the new class in setUpClass so test discovery remains useful during build.


class NativeVrahatisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cyroot.native.vrahatis import Vrahatis
        cls.Solver=Vrahatis
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-vrahatis-consumer-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory=Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_vrahatis_consumer.pyx'),directory/'native_vrahatis_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\nimport sys\n'
            'setup(ext_modules=cythonize([Extension("native_vrahatis_consumer",'
            '["native_vrahatis_consumer.pyx"],language="c++",include_dirs=[get_include()],'
            'extra_compile_args=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"])],'
            'language_level=3,include_path=[get_cython_include()]))\n')
        environment=os.environ.copy();environment.pop('PYTHONPATH',None)
        build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
            env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:raise AssertionError(build.stdout)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('native_vrahatis_consumer')
        cls.addClassCleanup(sys.modules.pop,'native_vrahatis_consumer',None)

    def test_analytic_roots_sign_cells_geometry_and_counts(self):
        # Shift .7 avoids any exact initial zero sign and exercises edge sweeps.
        for n in (1,2,3):
            points=tuple(itertools.product((0.,2.),repeat=n))
            calls=[]
            def residual(x):
                self.assertIsInstance(x,tuple)
                calls.append(x)
                return tuple(t-.7 for t in x)
            result=self.Solver(residual,points,max_iter=100,etol=1e-10,ertol=0.).run()
            self.assertTrue(result.success,result)
            assert_allclose(result.root,(.7,)*n,atol=1e-9,rtol=0.)
            self.assertEqual(result.f_calls,len(calls))
            self.assertEqual(result.error,max(abs(t) for t in result.f_root))
            self.assertEqual(result.f_root,residual(result.root))
            a=np.asarray(result.bracket)
            self.assertAlmostEqual(result.precision,float(np.max(a.max(0)-a.min(0))))
        # Rectangular n=3,m=2 preserves input coordinates while bracketing signs.
        points=tuple((a,b,9.) for a,b in itertools.product((0.,2.),repeat=2))
        result=self.Solver(lambda x:(x[0]-1.,x[1]-1.),points).run()
        self.assertTrue(result.success)
        assert_allclose(result.root,(1.,1.,9.))
        # A sign-complete polygon can have no common root. Width cannot certify it.
        points=tuple(itertools.product((-1.,1.),(-2.,2.)))
        result=self.Solver(lambda x:(x[0],x[0]*x[1]-1.),points,
                           etol=0.,ertol=0.,ptol=10.,prtol=0.).run()
        self.assertEqual(result.termination_reason,'stagnation')
        self.assertFalse(result.success)
        with self.assertRaises(ValueError):self.Solver(lambda x:(1.,1.),points)

    def test_cache_copied_history_step_run_budgets_and_invalid_options(self):
        points=np.asarray(list(itertools.product((0.,2.),repeat=2)))
        values=points-.7
        saved_points=points.copy();saved_values=values.copy()
        callback=lambda x:(x[0]-.7,x[1]-.7)
        opts=dict(max_iter=3,etol=0.,ertol=0.,ptol=0.,prtol=0.,history=16)
        full=self.Solver(callback,points,**opts)
        stepped=self.Solver(callback,points,F_x0s=values,**opts)
        initial=stepped.history
        points[:]=100;values[:]=200
        first=stepped.step();actual=stepped.run();expected=full.run()
        self.assertEqual(actual.root,expected.root)
        self.assertEqual(actual.f_calls+4,expected.f_calls)
        self.assertEqual(actual.iters,3)
        self.assertEqual(actual.termination_reason,'max_iterations')
        self.assertEqual(initial[0].iters,0)
        self.assertEqual(stepped.history[-1].root,actual.root)
        with self.assertRaises((AttributeError,TypeError)):first.root=(0.,0.)
        del stepped;gc.collect();self.assertTrue(first.root)
        truncated=self.Solver(callback,saved_points,history=1,max_iter=3,
                              etol=0.,ertol=0.,ptol=0.,prtol=0.)
        truncated.run();self.assertTrue(truncated.history_truncated)
        for opts in (dict(history=1,history_bytes=0),dict(workspace_bytes=0),
                     dict(history_every=0),dict(max_iter=True),dict(max_iter=-1)):
            with self.assertRaises((ValueError,TypeError)):
                self.Solver(callback,saved_points,**opts)
        public=vrahatis(callback,saved_points,F_x0s=saved_values,max_iter=3,
                         etol=0.,ertol=0.,ptol=0.,prtol=0.)
        self.assertEqual(public.root,expected.root)

    def test_native_callbacks_gil_lifetime_failures_and_guards(self):
        points=tuple(itertools.product((0.,2.),repeat=2))
        for rectangular in (False,True):
            owner=self.consumer.Owner()
            callback=owner.rectangular(2,2) if rectangular else owner.callback()
            result=self.Solver(callback,points).run()
            self.assertTrue(result.success,result)
            self.assertEqual(owner.gil_held,0)
            self.assertEqual(result.f_calls,owner.calls)
        owner=self.consumer.Owner();reference=weakref.ref(owner)
        solver=self.Solver(owner.callback(),points)
        del owner;gc.collect();self.assertIsNotNone(reference())
        self.assertTrue(solver.run().success)
        del solver;gc.collect();self.assertIsNone(reference())
        for options,reason in ((dict(fail_after=4,code=-17),'callback_failure'),
                               (dict(partial_after=4),'nonfinite')):
            owner=self.consumer.Owner(**options)
            solver=self.Solver(owner.callback(),points)
            old=solver.result
            if reason=='callback_failure':
                with self.assertRaises(NativeCallbackError) as caught:solver.run()
                self.assertEqual(caught.exception.native_status,-17)
            else:solver.run()
            self.assertEqual(solver.result.termination_reason,reason)
            self.assertEqual(solver.result.root,old.root)
            self.assertEqual(owner.gil_held,0)
        error=KeyboardInterrupt('original')
        def explosive(x):
            if x not in points:raise error
            return tuple(t-.7 for t in x)
        solver=self.Solver(explosive,points)
        with self.assertRaises(KeyboardInterrupt) as caught:solver.run()
        self.assertIs(caught.exception,error)
        adapter=VectorCallback(lambda x:(x[0]-.7,x[1]-.7),2)
        case=self
        class Coordinate:
            def __float__(self):
                with case.assertRaisesRegex(RuntimeError,'active'):case.Solver(adapter,points)
                return 0.
        self.assertTrue(self.Solver(adapter,((Coordinate(),0.),(0.,2.),(2.,0.),(2.,2.))).run().success)

    def test_polygon_discovery_helpers_and_bounded_failures(self):
        from cyroot.vector_bracketing import (
            compute_admissible_n_polygon,get_M,_get_1_simplexes,sorted_by_vertices,
        )
        # Corners have only two signs; refining true x-axis edges in 3D
        # discovers the two missing signs. Zero components are never a pattern.
        calls=[]
        def curved(x):
            self.assertIsInstance(x,tuple)
            calls.append(x)
            return (x[0]-.5,x[1]-.5+4*(x[0]-.5)**2)
        vertices,signs=compute_admissible_n_polygon(curved,(0.,0.,0.),h=(1.,1.,1.))
        self.assertEqual(vertices.shape,(4,3))
        assert_allclose(signs,get_M(2,sign=True))
        for point,sign in zip(vertices,signs):assert_allclose(np.sign(curved(tuple(point))),sign)
        self.assertTrue(any(0.<p[0]<1. for p in vertices))
        cube=get_M(3)
        assert_allclose(cube,np.asarray(tuple(itertools.product((0.,1.),repeat=3))))
        pairs=_get_1_simplexes(cube)
        self.assertEqual(int(np.sum(pairs[:,2]==2)),12)
        sorted_values,sorted_signs=sorted_by_vertices(np.arange(4),S=signs[::-1])
        assert_allclose(sorted_signs,signs)
        assert_allclose(sorted_values,np.arange(4)[::-1])
        for options in (dict(eps=0.),dict(eps=math.inf),dict(h=(0.,0.,0.)),
                        dict(workspace_bytes=0)):
            with self.assertRaises(ValueError):
                compute_admissible_n_polygon(curved,(0.,0.,0.),**options)
        with self.assertRaises(ValueError):
            compute_admissible_n_polygon(lambda x:(1.,1.),(0.,0.),h=1.)
        # The solver retains a sampled root even if no full strict polygon
        # can exist for two dependent residual components.
        sampled=self.Solver(lambda x:(x[0]-.5,x[0]-.5),(0.,0.),h=(1.,1.)).run()
        self.assertTrue(sampled.success)
        self.assertEqual(sampled.f_root,(0.,0.))
        with self.assertRaises(ValueError):
            compute_admissible_n_polygon(lambda x:(x[0]-.5,x[0]-.5),(0.,0.),h=(1.,1.))
        # Solver root shortcuts do not establish the helper's strict signs.
        with self.assertRaises(ValueError):
            compute_admissible_n_polygon(lambda x:(x[0],),((1.,),(0.,)))
        self.assertTrue(self.Solver(lambda x:(x[0],),((1.,),(0.,))).run().success)
        huge=(sys.float_info.max,)*3
        initial=self.Solver(lambda x:(0.,0.),huge,F_x0s=(0.,0.)).run()
        self.assertTrue(initial.success)
        self.assertEqual(initial.f_calls,0)
        for bad in (math.nan,math.inf):
            invalid=self.Solver(lambda x:self.fail('nonfinite point sampled'),(bad,)).run()
            self.assertEqual(invalid.termination_reason,'nonfinite')
            self.assertEqual(invalid.f_calls,0)
        with self.assertRaises(ValueError):get_M(100)
        with self.assertRaises(ValueError):get_M(20,workspace_bytes=16)
        with self.assertRaises(ValueError):_get_1_simplexes(cube,workspace_bytes=1)
