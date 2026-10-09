import os
import unittest

import mt.geo
from mt.geo import join_volume, register_join_volume

assert os.path.abspath(mt.geo.__file__).startswith("/home/minhtri/gitcentre/mtgeo/")


class TestRegisterJoinVolume(unittest.TestCase):
    def test_duplicate_message(self):
        class A:
            pass

        register_join_volume(A, A, lambda a, b: (0, 0, 0, 0))
        with self.assertRaises(SyntaxError) as cm:
            register_join_volume(A, A, lambda a, b: (0, 0, 0, 0))
        msg = str(cm.exception)
        self.assertNotIn("Volumeimate", msg)
        self.assertIn("Join_volume function", msg)
        self.assertIn("has been registered before", msg)
        self.assertEqual(join_volume(A(), A()), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
