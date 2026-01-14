"""Tests membership functions."""
import unittest

import jax.numpy as jnp

from flex.mfs import Gaussian, Triangle, Trapezoid, LeftShoulder, RightShoulder
from flex.mfs.functions import gaussian, triangle, trapezoid, left_shoulder, right_shoulder


class TestMFs(unittest.TestCase):
    def _generate_nodes(self):
        return jnp.linspace(0.0, 1.0, 10)

    def _generate_sigs(self):
        return jnp.array([0.5, 0.6, 0.4])

    def test_gaussian(self):
        nodes = self._generate_nodes()
        sigs = self._generate_sigs()

        sig_idx = 1
        idx = 2
        name = "mf"
        
        params = jnp.array([sigs[sig_idx], nodes[idx]])
        
        mf = Gaussian(idx=idx, sig_idx=sig_idx, name=name)
        
        assert mf.name == name
        assert jnp.all(mf.get_params(nodes, sigs) == params)

    def test_triangle(self):
        nodes = self._generate_nodes()

        idx = 1
        name = "mf"
        
        params = jnp.array([nodes[idx-1], nodes[idx], nodes[idx+1]])
        
        mf = Triangle(idx=idx, name=name)
        
        assert mf.name == name
        assert jnp.all(mf.get_params(nodes) == params)

    def test_trapezoid(self):
        nodes = self._generate_nodes()

        idx = 1
        name = "mf"
        
        params = jnp.array([nodes[idx-1], nodes[idx], nodes[idx+1], nodes[idx+2]])
        
        mf = Trapezoid(idx=idx, name=name)
        
        assert mf.name == name
        assert jnp.all(mf.get_params(nodes) == params)

    def test_left_shoulder(self):
        nodes = self._generate_nodes()

        idx = 0
        name = "mf"
        
        params = jnp.array([nodes[idx], nodes[idx+1]])
        
        mf = LeftShoulder(idx=idx, name=name)
        
        assert mf.name == name
        assert jnp.all(mf.get_params(nodes) == params)

    def test_right_shoulder(self):
        nodes = self._generate_nodes()

        idx = 9
        name = "mf"
        
        params = jnp.array([nodes[idx-1], nodes[idx]])
        
        mf = RightShoulder(idx=idx, name=name)
        
        assert mf.name == name
        assert jnp.all(mf.get_params(nodes) == params)

class TestMFFunctions(unittest.TestCase):
    def _generate_test_points(self):
        return jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    def test_gaussian(self):
        sig, mu = 2.0, 0.0

        xs = self._generate_test_points()
        ys = jnp.exp(-(xs - mu)**2 / (2.0*sig**2))
        vals = gaussian(xs, sig, mu)

        assert jnp.all(ys == vals)

    def test_triangle(self):
        a, b, c = -2.0, 0.0, 2.0

        xs = self._generate_test_points()
        ys = jnp.array([0.0, 0.5, 1.0, 0.5, 0.0])
        vals = triangle(xs, a, b, c)

        assert jnp.all(ys == vals)

    def test_trapezoid(self):
        a, b, c, d = -2.0, -1.0, 1.0, 2.0

        xs = self._generate_test_points()
        ys = jnp.array([0.0, 1.0, 1.0, 1.0, 0.0])
        vals = trapezoid(xs, a, b, c, d)

        assert jnp.all(ys == vals)

    def test_left_shoulder(self):
        c, d = 0.0, 2.0

        xs = self._generate_test_points()
        ys = jnp.array([1.0, 1.0, 1.0, 0.5, 0.0])
        vals = left_shoulder(xs, c, d)

        assert jnp.all(ys == vals)

    def test_right_shoulder(self):
        a, b = -2.0, 0.0

        xs = self._generate_test_points()
        ys = jnp.array([0.0, 0.5, 1.0, 1.0, 1.0])
        vals = right_shoulder(xs, a, b)

        assert jnp.all(ys == vals)
