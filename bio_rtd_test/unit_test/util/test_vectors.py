import unittest

import numpy as np

from bio_rtd.utils import vectors


class BooleanIntervalTest(unittest.TestCase):

    def test_true_start(self):
        v = np.ones(100, dtype=bool)
        self.assertEqual(vectors.true_start(v), 0)
        v[:10] = False
        self.assertEqual(vectors.true_start(v), 10)
        v[-10:] = False
        self.assertEqual(vectors.true_start(v), 10)
        v[:] = False
        with self.assertRaises(ValueError):
            vectors.true_start(v)

    def test_true_end(self):
        v = np.ones(100, dtype=bool)
        self.assertEqual(vectors.true_end(v), v.size)
        v[:10] = False
        self.assertEqual(vectors.true_end(v), v.size)
        v[-10:] = False
        self.assertEqual(vectors.true_end(v), v.size - 10)
        v[:] = False
        with self.assertRaises(ValueError):
            vectors.true_end(v)

    def test_true_start_and_end(self):
        v = np.ones(100, dtype=bool)
        self.assertEqual(vectors.true_start_and_end(v), (0, v.size))
        v[:10] = False
        self.assertEqual(vectors.true_start_and_end(v), (10, v.size))
        v[-10:] = False
        self.assertEqual(vectors.true_start_and_end(v), (10, v.size - 10))
        v[:] = False
        with self.assertRaises(ValueError):
            vectors.true_start_and_end(v)
