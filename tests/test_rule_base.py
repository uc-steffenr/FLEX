"""Tests fuzzy variable class."""
import unittest

import numpy as np

import jax
import equinox as eqx
import jax.numpy as jnp

from flex.fiss import RuleBase, RuleStats, update_rule_stats, mf_usage_from_stats
from flex.fiss.rule_stats import mf_usage_from_rule_values


def _np(x):
    """Convert JAX arrays to NumPy arrays for assertions."""
    return np.asarray(x)


class TestRuleBaseDense(unittest.TestCase):
    def test_dense_shapes_and_enumeration(self):
        # n_mfs_per_var = [2, 3] => R=6, V=2
        rb = RuleBase.dense([2, 3], tnorm="prod")
        self.assertEqual(rb.n_vars, 2)
        self.assertEqual(rb.n_rules, 6)

        ants = _np(rb.antecedents)
        self.assertEqual(ants.shape, (6, 2))

        # var0 fastest-changing digit
        expected = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 2],
                [1, 2],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(ants, expected)


class TestRuleBaseSparse(unittest.TestCase):
    def test_sparse_construction_uses_dont_care(self):
        rb = RuleBase.sparse(
            n_vars=3,
            rules=[
                [(0, 2), (2, 1)],  # x0 MF2 AND x2 MF1
                [(1, 0)],          # x1 MF0
            ],
            tnorm="prod",
        )
        ants = _np(rb.antecedents)
        self.assertEqual(ants.shape, (2, 3))

        expected = np.array(
            [
                [2, -1, 1],
                [-1, 0, -1],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(ants, expected)

    def test_sparse_raises_on_bad_var_index(self):
        with self.assertRaises(ValueError):
            _ = RuleBase.sparse(
                n_vars=2,
                rules=[[(2, 0)]],  # invalid var index
                tnorm="prod",
            )

    def test_sparse_raises_on_negative_mf_index(self):
        with self.assertRaises(ValueError):
            _ = RuleBase.sparse(
                n_vars=2,
                rules=[[(0, -3)]],  # invalid mf index
                tnorm="prod",
            )


class TestRuleBaseFire(unittest.TestCase):
    def test_fire_unbatched_prod_with_dont_care(self):
        # V=2, M=3
        mu = jnp.array(
            [
                [0.2, 0.6, 0.9],  # var0
                [0.1, 0.5, 0.8],  # var1
            ],
            dtype=jnp.float32,
        )  # (V, M)

        # r0: var0->MF2, var1->MF1 => 0.9 * 0.5 = 0.45
        # r1: var0->MF0, var1 don't-care => 0.2 * 1.0 = 0.2
        rb = RuleBase(jnp.array([[2, 1], [0, -1]], dtype=jnp.int32), tnorm="prod")
        w = rb.fire(mu)  # (R,)

        np.testing.assert_allclose(
            _np(w), np.array([0.45, 0.2], dtype=np.float32), rtol=0, atol=1e-6
        )

    def test_fire_batched_min_with_dont_care(self):
        mu = jnp.array(
            [
                # sample 0
                [
                    [0.2, 0.6, 0.9],  # var0
                    [0.1, 0.5, 0.8],  # var1
                ],
                # sample 1
                [
                    [0.7, 0.4, 0.3],
                    [0.9, 0.2, 0.1],
                ],
            ],
            dtype=jnp.float32,
        )  # (B=2, V=2, M=3)

        rb = RuleBase(jnp.array([[2, 1], [0, -1]], dtype=jnp.int32), tnorm="min")
        w = rb.fire(mu)  # (B, R)

        expected = np.array(
            [
                [0.5, 0.2],
                [0.2, 0.7],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(_np(w), expected, rtol=0, atol=1e-6)

    def test_fire_raises_on_wrong_nvars(self):
        rb = RuleBase.dense([2, 2], tnorm="prod")  # V=2
        mu_wrong = jnp.zeros((3, 2), dtype=jnp.float32)  # (V=3, M=2)
        with self.assertRaises(ValueError):
            _ = rb.fire(mu_wrong)

    def test_fire_jittable_for_batched_and_unbatched(self):
        rb = RuleBase(jnp.array([[1, -1], [0, 2]], dtype=jnp.int32), tnorm="prod")

        mu_unbatched = jnp.array(
            [
                [0.2, 0.6, 0.9],
                [0.1, 0.5, 0.8],
            ],
            dtype=jnp.float32,
        )  # (V=2, M=3)
        mu_batched = mu_unbatched[None, :, :]  # (B=1, V=2, M=3)

        f = eqx.filter_jit(rb.fire)
        w1 = f(mu_unbatched)
        w2 = f(mu_batched)

        self.assertEqual(w1.shape, (2,))
        self.assertEqual(w2.shape, (1, 2))

    def test_fire_accepts_ellipsis_batch_dims_nonjit(self):
        # R=2, V=2, M=3
        rb = RuleBase(jnp.array([[2, 1], [0, -1]], dtype=jnp.int32), tnorm="prod")

        # Build mu with shape (T, B, V, M) = (2, 3, 2, 3)
        # We'll make values easy to reason about.
        T, B, V, M = 2, 3, 2, 3
        mu = jnp.array(
            [
                # t=0
                [
                    [[0.2, 0.6, 0.9], [0.1, 0.5, 0.8]],  # b=0
                    [[0.3, 0.4, 0.5], [0.9, 0.2, 0.1]],  # b=1
                    [[0.7, 0.2, 0.1], [0.4, 0.4, 0.4]],  # b=2
                ],
                # t=1
                [
                    [[0.9, 0.1, 0.0], [0.6, 0.6, 0.6]],
                    [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
                    [[0.5, 0.5, 0.5], [0.8, 0.1, 0.0]],
                ],
            ],
            dtype=jnp.float32,
        )
        self.assertEqual(mu.shape, (T, B, V, M))

        w = rb.fire(mu)
        self.assertEqual(w.shape, (T, B, rb.n_rules))

        # Spot-check a couple values:
        # rule0: (v0 MF2) * (v1 MF1)
        # rule1: (v0 MF0) * (dont-care -> 1)
        # t=0, b=0: v0 MF2=0.9, v1 MF1=0.5 => 0.45; rule1=v0 MF0=0.2
        w_np = np.asarray(w)
        self.assertAlmostEqual(float(w_np[0, 0, 0]), 0.9 * 0.5, places=6)
        self.assertAlmostEqual(float(w_np[0, 0, 1]), 0.2, places=6)

        # t=1, b=2: v0 MF2=0.5, v1 MF1=0.1 => 0.05; rule1=v0 MF0=0.5
        self.assertAlmostEqual(float(w_np[1, 2, 0]), 0.5 * 0.1, places=6)
        self.assertAlmostEqual(float(w_np[1, 2, 1]), 0.5, places=6)

    def test_fire_accepts_ellipsis_batch_dims_jittable(self):
        rb = RuleBase(jnp.array([[2, 1], [0, -1]], dtype=jnp.int32), tnorm="prod")

        T, B, V, M = 2, 3, 2, 3
        mu = jnp.ones((T, B, V, M), dtype=jnp.float32) * 0.5

        # Use eqx.filter_jit for methods closing over eqx.Modules.
        f = eqx.filter_jit(rb.fire)
        w = f(mu)

        self.assertEqual(w.shape, (T, B, rb.n_rules))
        # With all memberships 0.5:
        # rule0 (prod): 0.5 * 0.5 = 0.25
        # rule1 (prod): 0.5 * 1.0 = 0.5
        w_np = np.asarray(w)
        self.assertAlmostEqual(float(w_np[0, 0, 0]), 0.25, places=6)
        self.assertAlmostEqual(float(w_np[0, 0, 1]), 0.5, places=6)


class TestRuleStats(unittest.TestCase):
    def test_init_shapes_and_zeros(self):
        stats = RuleStats.init(4, dtype=jnp.float32)
        np.testing.assert_array_equal(_np(stats.mass), np.zeros((4,), dtype=np.float32))
        np.testing.assert_array_equal(_np(stats.count), np.zeros((4,), dtype=np.float32))
        np.testing.assert_array_equal(_np(stats.ema_mass), np.zeros((4,), dtype=np.float32))

    def test_update_sum_reduction(self):
        stats = RuleStats.init(3, dtype=jnp.float32)

        w = jnp.array(
            [
                [0.0, 0.2, 0.5],
                [0.1, 0.0, 0.4],
            ],
            dtype=jnp.float32,
        )  # (B=2, R=3)

        # tau=0.15: fires are > 0.15
        # batch_mass sum: [0.1, 0.2, 0.9]
        # batch_count sum: [0, 1, 2]
        stats2 = update_rule_stats(stats, w=w, tau=0.15, ema_alpha=1.0, reduce="sum")

        np.testing.assert_allclose(_np(stats2.mass), np.array([0.1, 0.2, 0.9], np.float32), atol=1e-6)
        np.testing.assert_allclose(_np(stats2.count), np.array([0.0, 1.0, 2.0], np.float32), atol=1e-6)
        np.testing.assert_allclose(_np(stats2.ema_mass), np.array([0.1, 0.2, 0.9], np.float32), atol=1e-6)

    def test_update_mean_reduction(self):
        stats = RuleStats.init(3, dtype=jnp.float32)

        w = jnp.array(
            [
                [0.0, 0.2, 0.5],
                [0.1, 0.0, 0.4],
            ],
            dtype=jnp.float32,
        )

        # mean:
        # batch_mass: [0.05, 0.1, 0.45]
        # batch_count (tau=0.15): [0, 0.5, 1.0]
        stats2 = update_rule_stats(stats, w=w, tau=0.15, ema_alpha=1.0, reduce="mean")

        np.testing.assert_allclose(_np(stats2.mass), np.array([0.05, 0.1, 0.45], np.float32), atol=1e-6)
        np.testing.assert_allclose(_np(stats2.count), np.array([0.0, 0.5, 1.0], np.float32), atol=1e-6)
        np.testing.assert_allclose(_np(stats2.ema_mass), np.array([0.05, 0.1, 0.45], np.float32), atol=1e-6)

    def test_update_unbatched_input(self):
        stats = RuleStats.init(2, dtype=jnp.float32)
        w = jnp.array([0.2, 0.0], dtype=jnp.float32)  # (R,)
        stats2 = update_rule_stats(stats, w=w, tau=0.15, ema_alpha=1.0, reduce="sum")

        np.testing.assert_allclose(_np(stats2.mass), np.array([0.2, 0.0], np.float32), atol=1e-6)
        np.testing.assert_allclose(_np(stats2.count), np.array([1.0, 0.0], np.float32), atol=1e-6)

    def test_update_jittable(self):
        stats = RuleStats.init(3, dtype=jnp.float32)
        w = jnp.array([[0.0, 0.2, 0.5], [0.1, 0.0, 0.4]], dtype=jnp.float32)

        f = jax.jit(lambda s, ww: update_rule_stats(s, w=ww, tau=0.15, ema_alpha=0.5, reduce="sum"))
        stats2 = f(stats, w)

        self.assertEqual(stats2.mass.shape, (3,))
        self.assertEqual(stats2.count.shape, (3,))
        self.assertEqual(stats2.ema_mass.shape, (3,))


class TestMFUsageAggregation(unittest.TestCase):
    def test_mf_usage_from_rule_values_respects_dont_care(self):
        # R=3, V=2, max_mfs=3
        ants = jnp.array(
            [
                [0, 1],
                [2, -1],
                [0, 2],
            ],
            dtype=jnp.int32,
        )
        rule_vals = jnp.array([10.0, 1.0, 2.0], dtype=jnp.float32)

        out = mf_usage_from_rule_values(antecedents=ants, rule_values=rule_vals, max_mfs=3, normalize=False)

        expected = np.array(
            [
                [12.0, 0.0, 1.0],
                [0.0, 10.0, 2.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(_np(out), expected, atol=1e-6)

    def test_mf_usage_from_stats_wrapper(self):
        ants = jnp.array([[0, 1], [2, -1], [0, 2]], dtype=jnp.int32)
        stats = RuleStats(
            mass=jnp.array([10.0, 1.0, 2.0], dtype=jnp.float32),
            count=jnp.array([3.0, 5.0, 7.0], dtype=jnp.float32),
            ema_mass=jnp.array([4.0, 0.5, 1.0], dtype=jnp.float32),
        )

        out_mass = mf_usage_from_stats(antecedents=ants, stats=stats, max_mfs=3, which="mass", normalize=False)
        out_ema = mf_usage_from_stats(antecedents=ants, stats=stats, max_mfs=3, which="ema_mass", normalize=False)

        expected_mass = np.array([[12.0, 0.0, 1.0], [0.0, 10.0, 2.0]], dtype=np.float32)
        expected_ema = np.array([[5.0, 0.0, 0.5], [0.0, 4.0, 1.0]], dtype=np.float32)

        np.testing.assert_allclose(_np(out_mass), expected_mass, atol=1e-6)
        np.testing.assert_allclose(_np(out_ema), expected_ema, atol=1e-6)

    def test_mf_usage_normalize_rows_sum_to_one_when_nonzero(self):
        ants = jnp.array([[0, 1], [2, -1], [0, 2]], dtype=jnp.int32)
        rule_vals = jnp.array([10.0, 1.0, 2.0], dtype=jnp.float32)

        out = mf_usage_from_rule_values(antecedents=ants, rule_values=rule_vals, max_mfs=3, normalize=True)
        out_np = _np(out)

        np.testing.assert_allclose(out_np.sum(axis=1), np.ones((2,), dtype=np.float32), atol=1e-6)
        self.assertTrue(np.all(out_np >= -1e-7))

    def test_mf_usage_functions_jittable(self):
        ants = jnp.array([[0, 1], [2, -1], [0, 2]], dtype=jnp.int32)
        rule_vals = jnp.array([10.0, 1.0, 2.0], dtype=jnp.float32)

        f = jax.jit(lambda a, v: mf_usage_from_rule_values(antecedents=a, rule_values=v, max_mfs=3, normalize=False))
        out = f(ants, rule_vals)
        self.assertEqual(out.shape, (2, 3))
