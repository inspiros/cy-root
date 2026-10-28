"""A callback failure ends a solve; terminal inspection cannot invent errors."""
import math
import unittest

from cyroot.native import (
    Bisection, Chebyshev, Halley, ModifiedHalley, SuperHalley,
    TangentHyperbolas, Newton, Secant, VectorNewton,
)


class TerminalCallbackFailureTests(unittest.TestCase):
    def test_python_failure_is_delivered_once_and_terminal_calls_are_passive(self):
        for method in ('bisection', 'newton', 'numerical_newton', 'secant',
                       'halley', 'chebyshev', 'modified_halley', 'super_halley',
                       'tangent_hyperbolas', 'vector', 'numerical_vector'):
            for during_setup in (False, True):
                with self.subTest(method=method, during_setup=during_setup):
                    error = KeyboardInterrupt('original callback failure')
                    calls = []
                    initial_calls = (3 if method.startswith('numerical') else
                                     2 if method in ('bisection', 'secant') else 1)

                    def function(x):
                        calls.append(x)
                        if during_setup or len(calls) > initial_calls:
                            raise error
                        return (x[0]*x[0]-2.,) if 'vector' in method else x*x-2.

                    if method == 'bisection':
                        cls, args = Bisection, (function, 1., 2.)
                    elif method == 'secant':
                        cls, args = Secant, (function, 1., 2.)
                    elif method in ('newton', 'numerical_newton'):
                        cls = Newton
                        args = (function, None if method == 'numerical_newton' else lambda x: 2*x, 1.5)
                    elif 'vector' in method:
                        cls = VectorNewton
                        args = (function, None if method == 'numerical_vector' else lambda x: ((2*x[0],),), (1.5,))
                    else:
                        cls = {'halley': Halley, 'chebyshev': Chebyshev,
                               'modified_halley': ModifiedHalley,
                               'super_halley': SuperHalley,
                               'tangent_hyperbolas': TangentHyperbolas}[method]
                        args = (function, lambda x: 2*x, lambda x: 2., 1.5)

                    solver = cls.__new__(cls)
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        solver.__init__(*args, history=8)
                        solver.run()
                    self.assertIs(caught.exception, error)
                    self.assertTrue(solver.done)
                    before = solver.result
                    count = len(calls)
                    history_length = len(solver.history)
                    for _ in range(2):
                        terminal = solver.run()
                        step = solver.step()
                        self.assertEqual(terminal.root, before.root)
                        if math.isfinite(before.error):
                            self.assertEqual(terminal.error, before.error)
                        self.assertEqual(terminal.f_calls, before.f_calls)
                        self.assertEqual(step.f_calls, before.f_calls)
                        self.assertEqual(terminal.iters, before.iters)
                        self.assertEqual(terminal.termination_reason, 'callback_failure')
                        self.assertEqual(step.termination_reason, 'callback_failure')
                        self.assertEqual(len(solver.history), history_length)
                        self.assertEqual(len(calls), count)
                    self.assertEqual(list(solver), [])
