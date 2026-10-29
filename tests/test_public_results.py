"""Public copied-state semantics and the packaged Python typing contract."""
import ast
import importlib
import inspect
import math
from pathlib import Path
import unittest

import numpy as np

import cyroot
from cyroot import RootSnapshot, TerminationReason, as_root_snapshot
from cyroot import native
from cyroot.return_types import RootReturnType


class PublicResultTests(unittest.TestCase):
    def check_state(self, solver):
        initial = as_root_snapshot(solver.result)
        self.assertFalse(initial.success)
        self.assertIsNone(initial.termination_reason)
        raw = solver.step()
        copied = as_root_snapshot(raw)
        self.assertIs(copied.diagnostics, raw)
        self.assertEqual(copied.iters, raw.iters)
        self.assertEqual(copied.root, raw.root)
        self.assertEqual(copied.f_root, raw.f_root)
        final = as_root_snapshot(solver.run())
        self.assertTrue(final.success, final)
        self.assertTrue(final.residual_converged)
        history = tuple(map(as_root_snapshot, solver.history))
        self.assertEqual(history[0].root, initial.root)
        self.assertEqual(history[-1].root, final.root)
        self.assertIs(as_root_snapshot(copied), copied)
        with self.assertRaises(AttributeError):
            copied.root = 0.
        return final

    def test_scalar_states_copy_without_evaluation_and_keep_success_meaning(self):
        calls = []
        def function(x):
            calls.append(x)
            return x*x-2.
        first = lambda x: 2*x
        second = lambda x: 2.
        factories = (
            lambda: native.Bisection(function, 1., 2., history=128),
            lambda: native.BracketSolver(function, 1., 2., history=128),
            lambda: native.Newton(function, first, 1.5, history=128),
            lambda: native.Homeier(function, first, 1.5, history=128),
            lambda: native.Secant(function, 1., 2., history=128),
            lambda: native.Halley(function, first, second, 1.5, history=128),
            lambda: native.Householder(function, (first, second), 1.5, history=128),
            lambda: native.InverseQuadraticInterpolation(function, 1., 1.5, 2., history=128),
        )
        for factory in factories:
            solver = factory()
            with self.subTest(solver=type(solver).__name__):
                final = self.check_state(solver)
                self.assertIsInstance(final.root, float)
                self.assertAlmostEqual(final.root, math.sqrt(2.), places=9)
                before = len(calls)
                as_root_snapshot(solver.result)
                tuple(map(as_root_snapshot, solver.history))
                self.assertEqual(len(calls), before)
        complex_solver = native.Muller(lambda z: z*z+1., -1., 0., 1., history=128)
        final = self.check_state(complex_solver)
        self.assertIsInstance(final.root, complex)
        self.assertLess(abs(final.f_root), 1e-10)
        bracket = native.Bisection(function, 1., 2., etol=0., ertol=0., ptol=2., history=2)
        for raw in (bracket.result, bracket.history[0]):
            copied = as_root_snapshot(raw)
            self.assertTrue(copied.success)
            self.assertFalse(copied.residual_converged)
            self.assertEqual(copied.termination_reason, TerminationReason.BRACKET_TOLERANCE)
        source = RootReturnType(np.array([1., 2.]), np.array([3.]), f_calls=(7, 2))
        copied = as_root_snapshot(source)
        source.root[:] = 99.
        source.f_root[:] = 88.
        self.assertEqual((copied.root, copied.f_root, copied.f_calls), ((1., 2.), (3.,), (7, 2)))
        self.assertIs(copied.diagnostics, source)
        self.assertIsNone(copied.termination_reason)
        self.assertTrue(math.isnan(copied.error))

    def test_vector_optimizer_shapes_counts_and_search_separation(self):
        function = lambda x: (x[0]-.7,)
        jacobian = lambda x: ((1.,),)
        hessian = lambda x: (((0.,),),)
        factories = (
            lambda: native.VectorNewton(function, jacobian, (0.,), history=128),
            lambda: native.VectorChebyshev(function, jacobian, hessian, (0.,), history=128),
            lambda: native.VectorHalley(function, jacobian, hessian, (0.,), history=128),
            lambda: native.BroydenGood(function, ((1.,),), (0.,), history=128),
            lambda: native.WolfeBittner(function, ((0.,), (2.,)), history=128),
            lambda: native.Vrahatis(function, ((0.,), (2.,)), history=128),
        )
        for factory in factories:
            solver = factory()
            with self.subTest(solver=type(solver).__name__):
                final = self.check_state(solver)
                self.assertIsInstance(final.root, tuple)
                self.assertEqual(len(final.root), 1)
        for method in cyroot.OPTIMIZATION_ROOT_FINDING_METHODS:
            for scalar in (False, True):
                with self.subTest(method=method, scalar=scalar):
                    solver = cyroot.OptimizationSolver(method,
                        (lambda x: x-.7) if scalar else function,
                        0. if scalar else (0.,), scalar=scalar, history=128)
                    common = self.check_state(solver)
                    raw = as_root_snapshot(solver.native_state.result)
                    self.assertEqual(common.f_calls, raw.f_calls)
                    self.assertEqual(raw.f_calls, (raw.diagnostics.f_calls, raw.diagnostics.jac_calls))
                    self.assertIsInstance(common.root, float if scalar else tuple)
                    self.assertIsInstance(raw.root, tuple)
            stopped = as_root_snapshot(cyroot.OptimizationSolver(method,
                lambda x: x*x+1., 0., scalar=True).run())
            self.assertFalse(stopped.success)
            self.assertFalse(stopped.residual_converged)
            self.assertEqual(stopped.termination_reason, TerminationReason.STATIONARY_NONROOT)
        search = native.Hybisect(lambda x: x-.7, lambda a, b: (a-.7, b-.7), 0., 2., history=128)
        for raw in (search.history[0], search.run()):
            with self.assertRaisesRegex(TypeError, 'search'):
                as_root_snapshot(raw)

    def test_packaged_stubs_preserve_exports_and_public_argument_binding(self):
        root = Path(cyroot.__file__).parent
        self.assertTrue((root/'py.typed').is_file())
        for name in ('scalar_bracketing', 'scalar_quasi_newton', 'scalar_newton',
                     'vector_bracketing', 'vector_quasi_newton', 'vector_newton',
                     'return_types'):
            module = importlib.import_module('cyroot.'+name)
            tree = ast.parse((root/(name+'.pyi')).read_text())
            declarations = {node.name for node in tree.body
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            exports = next(ast.literal_eval(node.value) for node in tree.body
                           if isinstance(node, ast.Assign) and any(
                               isinstance(target, ast.Name) and target.id == '__all__'
                               for target in node.targets))
            self.assertEqual(set(exports), set(module.__all__))
            self.assertLessEqual(set(module.__all__), declarations)
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name not in module.__all__:
                    continue
                actual = inspect.signature(getattr(module, node.name))
                expected = [(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                            for arg in node.args.args]
                if node.args.vararg:
                    expected.append((node.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
                expected.extend((arg.arg, inspect.Parameter.KEYWORD_ONLY) for arg in node.args.kwonlyargs)
                if node.args.kwarg:
                    expected.append((node.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
                self.assertEqual(expected, [(p.name, p.kind) for p in actual.parameters.values()], node.name)
        lengths = dict(RootReturnType=8, BracketingMethodsReturnType=10,
                       NewtonMethodsReturnType=9, MultiRootsReturnType=9,
                       SplittingBracketingMethodsReturnType=11,
                       SplittingNewtonMethodsReturnType=10)
        for name, length in lengths.items():
            cls = getattr(cyroot.return_types, name)
            multiple = issubclass(cls, cyroot.return_types.MultiRootsReturnType)
            result = cls(precision=(.25,) if multiple else .25,
                         error=(.5,) if multiple else .5)
            values = tuple(result)
            self.assertEqual(len(values), length)
            self.assertEqual(list(iter(result)), list(values))
            for i, value in enumerate(values):
                self.assertEqual(value, result[i])
            with self.assertRaises(IndexError):
                result[len(values)]
        self.assertFalse(hasattr(cyroot, 'ahmadi'))
        self.assertFalse(hasattr(cyroot, 'mishchenko'))


if __name__ == '__main__':
    unittest.main()
