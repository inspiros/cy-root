import unittest

from cyroot import TerminationReason
from cyroot.return_types import (
    RootReturnType, BracketingMethodsReturnType, NewtonMethodsReturnType,
    SplittingBracketingMethodsReturnType,
)


class TerminationMetadataTests(unittest.TestCase):
    def test_only_documented_root_reasons_are_successful(self):
        for cls in (RootReturnType, BracketingMethodsReturnType, NewtonMethodsReturnType):
            for reason in TerminationReason:
                with self.subTest(cls=cls, reason=reason):
                    result = cls(converged=True, optimal=True, termination_reason=reason)
                    success = reason in (TerminationReason.RESIDUAL_TOLERANCE,
                                         TerminationReason.BRACKET_TOLERANCE)
                    self.assertEqual(result.success, success)
                    self.assertEqual(result.converged, success)
                    self.assertEqual(result.optimal, reason == TerminationReason.RESIDUAL_TOLERANCE)
                    self.assertEqual(result.termination_reason, str(reason))
                    self.assertIn(str(reason), repr(result))

    def test_reason_is_read_only_and_rejects_unknown_values(self):
        result = RootReturnType(termination_reason="stagnation")
        with self.assertRaises(AttributeError):
            result.termination_reason = TerminationReason.RESIDUAL_TOLERANCE
        for reason in ("arbitrary", -1, 999, True, object()):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                RootReturnType(termination_reason=reason)

    def test_split_search_and_per_root_reasons_are_distinct(self):
        result = SplittingBracketingMethodsReturnType(
            root=[-1., 1.], f_root=[0., .5],
            termination_reasons=["residual_tolerance", "max_iterations"],
            search_termination_reason="completed",
        )
        self.assertEqual(result.converged, [True, False])
        self.assertEqual(result.optimal, [True, False])
        self.assertEqual(result.search_termination_reason, TerminationReason.COMPLETED)
        self.assertFalse(result.success)
        partial = SplittingBracketingMethodsReturnType(
            root=[1.], f_root=[0.], termination_reasons=["residual_tolerance"],
            search_termination_reason="max_split_iterations",
        )
        self.assertEqual(partial.converged, [True])
        self.assertFalse(partial.success)
        empty_completed = SplittingBracketingMethodsReturnType(search_termination_reason="completed")
        self.assertTrue(empty_completed.success)

    def test_one_reason_is_required_per_root(self):
        with self.assertRaises(ValueError):
            SplittingBracketingMethodsReturnType(root=[1.], termination_reasons=[])

    def test_incomplete_manual_flags_cannot_make_search_successful(self):
        for roots, flags in (([1.], []), ([1., 2.], [True]), ([], [True])):
            with self.subTest(roots=roots, flags=flags):
                result = SplittingBracketingMethodsReturnType(
                    root=roots, converged=flags, search_termination_reason="completed")
                self.assertFalse(result.success)
