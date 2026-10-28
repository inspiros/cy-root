"""Established Householder names use the native engine without symbolic imports."""
import inspect
import math
import subprocess
import sys
import unittest

import cyroot
from cyroot import scalar_newton
from cyroot.native import householder


class HouseholderMigrationTests(unittest.TestCase):
    def test_established_entry_points_forward_runtime_order_and_caches(self):
        derivatives = [lambda x: 2*x, lambda x: 2.] + [lambda x: 0.]*10
        options = dict(f_x0=.25, dfs_x0=[3., 2.]+[0.]*10, max_iter=1,
                       etol=0., ertol=0., ptol=0., prtol=0.)
        expected = householder(lambda x: x*x-2., derivatives, 1.5, **options)
        for entry in (cyroot.householder, scalar_newton.householder):
            with self.subTest(entry=entry):
                actual = entry(lambda x: x*x-2., derivatives, 1.5, d=12, **options)
                self.assertEqual(actual.root, expected.root)
                self.assertEqual(actual.f_calls, (1,)+(0,)*12)
                self.assertEqual(actual.termination_reason, expected.termination_reason)
                self.assertTrue(all(math.isnan(value) for value in actual.df_root))

    def test_established_numerical_options_and_obsolete_keyword(self):
        actual = cyroot.householder(lambda x: x*x-2., None, 1.5, d=3, h=.125)
        self.assertTrue(actual.success)
        self.assertAlmostEqual(actual.root, math.sqrt(2.), places=10)
        with self.assertRaises(ValueError):
            cyroot.householder(lambda x: x-1., [lambda x: 1.], 0., d=2)
        with self.assertRaises(TypeError):
            cyroot.householder(lambda x: x-1., [lambda x: 1.], 0., c_code=True)
        self.assertNotIn('c_code', inspect.signature(cyroot.householder).parameters)
        for obsolete in ('ReciprocalDerivativeFuncFactory', 'CyExprEvaluator',
                         'R0DFPtr', 'R10DFPtr', 'householder_kernel'):
            self.assertFalse(hasattr(scalar_newton, obsolete), obsolete)

    def test_import_and_native_and_forwarded_solves_with_sympy_blocked(self):
        script = r'''
import importlib.abc
import math
import sys
class BlockSympy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sympy' or fullname.startswith('sympy.'):
            raise AssertionError('unexpected symbolic dependency: ' + fullname)
sys.meta_path.insert(0, BlockSympy())
import cyroot
from cyroot.native import Householder
roles = (lambda x: 2*x, lambda x: 2.) + (lambda x: 0.,)*10
for result in (Householder(lambda x: x*x-2., roles, 1.5).run(),
               cyroot.householder(lambda x: x*x-2., roles, 1.5)):
    assert result.success
    assert abs(result.root-math.sqrt(2.)) < 1e-12
assert not any(name == 'sympy' or name.startswith('sympy.') for name in sys.modules)
'''
        completed = subprocess.run([sys.executable, '-c', script], text=True,
                                   capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)


if __name__ == '__main__':
    unittest.main()
