"""Review-derived final correction rounding at the binary64 subnormal boundary."""
from fractions import Fraction
import math
import unittest

from cyroot.native import VectorModifiedHalley


class ModifiedVectorFinalRoundingTests(unittest.TestCase):
    def test_general_path_keeps_half_of_subnormal_correction_until_final_rounding(self):
        minimum = math.ulp(0.)
        for sign in (-1., 1.):
            # A=M=2^-51, v=+/-2^-1023 and t=s=+/-minsub exactly.
            # Rounding the half correction first yields one minsub instead of
            # the correctly rounded two. Compute the oracle in exact rationals.
            base = sign*minimum
            expected = float(-Fraction(base)-Fraction(base)/2)
            points = []
            def residual(point):
                points.append(point)
                return (base,)
            solver = VectorModifiedHalley(
                residual, lambda x: ((1.,),), lambda x: (((sign*2.**1023,),),),
                (0.,), alpha=2.**51-1, max_iter=1,
                etol=0., ertol=0., ptol=0., prtol=0.)
            with self.subTest(sign=sign):
                result = solver.run()
                self.assertEqual(result.root, (expected,))
                self.assertEqual(points, [(0.,), (expected,)])
                self.assertEqual(result.f_calls, (2, 1, 1))
                self.assertEqual(result.iters, 1)
                self.assertEqual(result.termination_reason, 'max_iterations')
                self.assertEqual(solver.linear_stage, 'correction')
                self.assertEqual(solver.linear_status, 0)


if __name__ == '__main__':
    unittest.main()
