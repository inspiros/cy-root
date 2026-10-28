"""Coverage for function tag predicates used by solver discovery."""
import unittest

from cyroot.utils._function_registering import (
    register,
    is_tagged_with_any_startswith_any,
    is_tagged_with_any_endswith_any,
)


class TagPredicateTests(unittest.TestCase):
    def test_multiple_prefixes_and_suffixes(self):
        @register('cyroot.vector.newton', 'private.scalar.helper')
        def function():
            pass

        self.assertTrue(is_tagged_with_any_startswith_any(function, 'absent', 'cyroot.vector'))
        self.assertFalse(is_tagged_with_any_startswith_any(function, 'absent', 'unknown'))
        self.assertTrue(is_tagged_with_any_endswith_any(function, '.absent', '.newton'))
        self.assertFalse(is_tagged_with_any_endswith_any(function, '.absent', '.unknown'))

    def test_empty_patterns_and_untagged_objects(self):
        @register('cyroot.vector.newton')
        def function():
            pass

        for predicate in (is_tagged_with_any_startswith_any, is_tagged_with_any_endswith_any):
            with self.subTest(predicate=predicate.__name__):
                self.assertFalse(predicate(function))
                self.assertFalse(predicate(object(), 'cyroot'))
                self.assertTrue(predicate(function, ''))


if __name__ == '__main__':
    unittest.main()
