import unittest

import mt.geo as g


class TestFalseNotRaise(unittest.TestCase):
    def test_return_false(self):
        self.assertIs(g.approximable(3, str), False)
        self.assertIs(g.upper_boundable(3, str), False)
        self.assertIs(g.lower_boundable(3, str), False)
        self.assertIs(g.transformable(g.Transformer(), 3.0), False)


if __name__ == "__main__":
    unittest.main()
