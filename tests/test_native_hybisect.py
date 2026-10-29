"""Bounded native interval search, state, failure and compiled callback contracts."""
import gc
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

from cyroot import hybisect
from cyroot.native import Hybisect, IntervalCallback, ScalarCallback, NativeCallbackError


def enclosure(a,b):
    return (-1. if a<=0<=b else min(a*a,b*b)-1.,max(a*a,b*b)-1.)


class HybisectTests(unittest.TestCase):
    def test_roots_pruning_duplicates_events_and_limits(self):
        options=dict(etol=1e-13,ertol=0.,ptol=0.,prtol=0.)
        solver=Hybisect(lambda x:x*x-1.,enclosure,2.,-2.,history=64,**options)
        before=solver.history
        for _ in solver:pass
        result=solver.result
        self.assertTrue(result.success)
        self.assertEqual(result.root,[-1.,1.])
        self.assertEqual(result.split_iters,1)
        self.assertEqual(result.f_calls,(8,3))
        self.assertEqual(solver.history[-1].termination_reason,'completed')
        self.assertEqual(len(before),1)
        self.assertTrue(any(event.event=='split' for event in solver.history))
        self.assertEqual(hybisect(lambda x:x*x-1.,enclosure,-2.,2.,**options).root,result.root)
        self.assertEqual(Hybisect(lambda x:x,lambda a,b:(a,b),0.,0.).run().root,[0.])
        self.assertEqual(Hybisect(lambda x:x*x,lambda a,b:(0.,max(a*a,b*b)),-1.,1.).run().root,[0.])
        empty=Hybisect(lambda x:1.,lambda a,b:(1.,1.),-1.,1.).run()
        self.assertTrue(empty.success)
        self.assertEqual(empty.f_calls,(0,1))
        for options,reason in ((dict(max_split_iter=1),'max_split_iterations'),
                               (dict(max_pending=1),'breakdown'),
                               (dict(max_roots=1),'breakdown')):
            system=Hybisect(lambda x:x*x-1. if 'max_roots' in options else 1.,
                enclosure if 'max_roots' in options else lambda a,b:(-1.,1.),-2.,2.,**options)
            self.assertEqual(system.run().search_termination_reason,reason)
            self.assertFalse(system.result.success)
        self.assertEqual(Hybisect(lambda x:x*x-2.,lambda a,b:(-2.,2.),0.,2.,max_iter=1,
                                  etol=0.,ertol=0.,ptol=0.,prtol=0.).run().termination_reasons,['max_iterations'])

    def test_failure_identity_guards_and_nonfinite_enclosures(self):
        for bad,reason in (((math.nan,1.),'nonfinite'),((2.,1.),'breakdown')):
            result=Hybisect(lambda x:self.fail('must not sample'),lambda a,b:bad,-1.,1.).run()
            self.assertEqual(result.search_termination_reason,reason)
        for stage in ('interval','function'):
            error=KeyboardInterrupt(stage)
            def explode(*args):raise error
            f=ScalarCallback(explode if stage=='function' else lambda x:x)
            interval=IntervalCallback(explode if stage=='interval' else lambda a,b:(a,b))
            solver=Hybisect(f,interval,-1.,1.)
            with self.assertRaises(KeyboardInterrupt) as caught:solver.run()
            self.assertIs(caught.exception,error)
            self.assertEqual(solver.result.search_termination_reason,'callback_failure')
            # The reservation is released after the failed callback.
            self.assertFalse(Hybisect(f,interval,-1.,1.).done)
        f=ScalarCallback(lambda x:x)
        interval=IntervalCallback(lambda a,b:(a,b))
        class Reentrant:
            def __float__(self):
                with self_case.assertRaises(RuntimeError):Hybisect(f,interval,-1.,1.)
                return -1.
        self_case=self
        self.assertTrue(Hybisect(f,interval,Reentrant(),1.).run().success)
        for options in (dict(max_pending=0),dict(max_roots=0),dict(history_every=0),
                        dict(workspace_bytes=0),dict(history=1,history_bytes=0)):
            with self.assertRaises(ValueError):Hybisect(f,interval,-1.,1.,**options)
        uninitialized=Hybisect.__new__(Hybisect)
        for operation in (uninitialized.run,uninitialized.step,lambda:uninitialized.result):
            with self.assertRaises(RuntimeError):operation()
        # Result/history boxing may invoke Python: reserve the solver and both
        # original adapters until export finishes, then release on exceptions.
        f=ScalarCallback(lambda x:x*x-1.)
        interval=IntervalCallback(enclosure)
        solver=Hybisect(f,interval,-2.,2.,history=32)
        for module,operation in ((importlib.import_module('cyroot.return_types'),lambda:solver.result),
                                 (importlib.import_module('cyroot.native.hybisect'),lambda:solver.history)):
            if module.__name__ == 'cyroot.native.hybisect':solver.run()
            original=module._normalize_reason
            seen=[]
            def normalize(reason):
                seen.append(reason)
                with self.assertRaises(RuntimeError):solver.step()
                with self.assertRaises(RuntimeError):Hybisect(f,IntervalCallback(enclosure),-2.,2.)
                with self.assertRaises(RuntimeError):Hybisect(ScalarCallback(lambda x:x),interval,-2.,2.)
                return original(reason)
            with patch.object(module,'_normalize_reason',normalize):operation()
            self.assertTrue(seen)
            error=LookupError('export')
            with patch.object(module,'_normalize_reason',side_effect=error):
                with self.assertRaises(LookupError) as caught:operation()
            self.assertIs(caught.exception,error)
            self.assertFalse(Hybisect(f,interval,-2.,2.).done)
        self.assertTrue(solver.run().success)

    def test_installed_native_callbacks_no_gil_lifetimes_and_statuses(self):
        with tempfile.TemporaryDirectory(prefix='cyroot-hybisect-consumer-', ignore_cleanup_errors=(os.name == 'nt')) as name:
            directory=Path(name)
            shutil.copyfile(Path(__file__).with_name('native_hybisect_consumer.pyx'),directory/'native_hybisect_consumer.pyx')
            (directory/'setup.py').write_text(
                'from setuptools import setup,Extension\nfrom Cython.Build import cythonize\n'
                'from cyroot.native import get_include,get_cython_include\nimport sys\n'
                'setup(ext_modules=cythonize([Extension("native_hybisect_consumer",'
                '["native_hybisect_consumer.pyx"],language="c++",include_dirs=[get_include()],'
                'extra_compile_args=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"])],'
                'language_level=3,include_path=[get_cython_include()]))\n')
            environment=os.environ.copy();environment.pop('PYTHONPATH',None)
            build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertEqual(build.returncode,0,build.stdout)
            sys.path.insert(0,name)
            try:
                consumer=importlib.import_module('native_hybisect_consumer')
                owner=consumer.Owner();reference=weakref.ref(owner)
                f,interval=owner.residual(),owner.enclosure()
                self.assertTrue(f.native and interval.native)
                solver=Hybisect(f,interval,-2.,2.,history=32)
                self.assertTrue(solver.run().success)
                self.assertEqual(solver.result.f_calls,(owner.calls,owner.interval_calls))
                self.assertEqual(owner.gil_held,0)
                del owner,f,interval;gc.collect();self.assertIsNotNone(reference())
                del solver;gc.collect();self.assertIsNone(reference())
                for failure,status,stage in ((1,-17,'function'),(2,23,'interval')):
                    owner=consumer.Owner();owner.failure=failure
                    solver=Hybisect(owner.residual(),owner.enclosure(),-2.,2.)
                    with self.assertRaises(NativeCallbackError) as caught:solver.run()
                    self.assertEqual(caught.exception.native_status,status)
                    self.assertEqual(caught.exception.callback_stage,stage)
                    self.assertEqual(owner.gil_held,0)
            finally:
                sys.path.remove(name);sys.modules.pop('native_hybisect_consumer',None)
