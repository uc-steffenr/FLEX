"""Tests membership functions."""
import unittest

import jax.numpy as jnp

# from flex.mfs.functions import gaussian, triangle, trapezoid, left_shoulder, right_shoulder


# class TestMFs(unittest.TestCase):
#     def _generate_test_points(self):
#         return jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

#     def test_gaussian(self):
#         sig, mu = 2.0, 0.0

#         xs = self._generate_test_points()
#         ys = jnp.exp(-(xs - mu)**2 / (2.0*sig**2))
#         vals = gaussian(xs, sig, mu)

#         assert jnp.all(ys == vals)

#     def test_triangle(self):
#         a, b, c = -2.0, 0.0, 2.0

#         xs = self._generate_test_points()
#         ys = jnp.array([0.0, 0.5, 1.0, 0.5, 0.0])
#         vals = triangle(xs, a, b, c)

#         assert jnp.all(ys == vals)

#     def test_trapezoid(self):
#         a, b, c, d = -2.0, -1.0, 1.0, 2.0

#         xs = self._generate_test_points()
#         ys = jnp.array([0.0, 1.0, 1.0, 1.0, 0.0])
#         vals = trapezoid(xs, a, b, c, d)

#         assert jnp.all(ys == vals)

#     def test_left_shoulder(self):
#         c, d = 0.0, 2.0

#         xs = self._generate_test_points()
#         ys = jnp.array([1.0, 1.0, 1.0, 0.5, 0.0])
#         vals = left_shoulder(xs, c, d)

#         assert jnp.all(ys == vals)

#     def test_right_shoulder(self):
#         a, b = -2.0, 0.0

#         xs = self._generate_test_points()
#         ys = jnp.array([0.0, 0.5, 1.0, 1.0, 1.0])
#         vals = right_shoulder(xs, a, b)

#         assert jnp.all(ys == vals)
