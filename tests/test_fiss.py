"""Tests FIS classes."""
import unittest

import jax
import jax.numpy as jnp
import equinox as eqx

from flex.fuzzy_variable import FuzzyVariable
from flex.fiss.rule_base import RuleBase
from flex.fiss.tsk import TSK


class TestTSK(unittest.TestCase):
    def _make_input_vars(self):
        # Different MF counts to exercise padding in BaseFIS.fuzzify
        # Use ruspini triangles to keep things deterministic and simple.
        fv2 = FuzzyVariable.ruspini(n_mfs=2, kind="triangle", minval=0.0, maxval=1.0, name="x1")
        fv3 = FuzzyVariable.ruspini(n_mfs=3, kind="triangle", minval=0.0, maxval=1.0, name="x2")
        return (fv2, fv3)

    def test_init_consequents_shape_multioutput_orders_dense(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(0)

        n_out = 4
        n_inps = len(input_vars)

        # Dense rulebase: n_rules = product of MF counts
        # Here: 2 * 3 = 6
        expected_n_rules = 2 * 3

        for order in (0, 1, 2):
            with self.subTest(order=order):
                fis = TSK.init(
                    input_vars=input_vars,
                    n_out=n_out,
                    order=order,
                    antecedents=None,   # dense
                    key=key,
                )

                self.assertEqual(fis.rb.n_rules, expected_n_rules)

                if order == 0:
                    expected_n_features = 1
                elif order == 1:
                    expected_n_features = 1 + n_inps
                else:
                    expected_n_features = 1 + n_inps + n_inps * (n_inps + 1) // 2

                self.assertEqual(
                    fis.consequents.shape,
                    (n_out, expected_n_rules, expected_n_features),
                )
                self.assertEqual(fis.n_out, n_out)
                self.assertEqual(fis.n_features, expected_n_features)

    def test_phi_shapes(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(1)

        # x shape (..., n_inps)
        x = jnp.zeros((5, len(input_vars)), dtype=jnp.float32)

        for order in (0, 1, 2):
            with self.subTest(order=order):
                fis = TSK.init(input_vars=input_vars, n_out=2, order=order, key=key)
                phi = fis.phi(x)

                if order == 0:
                    expected_f = 1
                elif order == 1:
                    expected_f = 1 + fis.n_inps
                else:
                    expected_f = 1 + fis.n_inps + fis.n_inps * (fis.n_inps + 1) // 2

                self.assertEqual(phi.shape, (5, expected_f))

    def test_fuzzify_padding(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(2)

        fis = TSK.init(input_vars=input_vars, n_out=2, order=1, key=key)

        # Arbitrary batch dims, last dim is n_inps (=2)
        x = jnp.array([[[0.25, 0.75],
                        [0.50, 0.50],
                        [0.75, 0.25]]], dtype=jnp.float32)  # shape (1, 3, 2)

        mus = fis.fuzzify(x)  # expected shape (1, 3, n_vars=2, n_mfs_max=3)
        self.assertEqual(mus.shape, (1, 3, 2, 3))

        # First variable has n_mfs=2, so last MF entry should be padding zero always.
        pad_slice = mus[..., 0, 2]  # (... batch ..., var0, mf_idx=2)
        self.assertTrue(jnp.all(pad_slice == 0.0))

        # Second variable has n_mfs=3, no padding.
        # We can't assert "nonzero" everywhere, but we can assert it's not *forced* to 0 by padding.
        # At least check that its 3rd entry equals the computed membership, not padding;
        # practical proxy: for some x values, membership should be > 0.
        var1_third = mus[..., 1, 2]
        self.assertTrue(jnp.any(var1_third > 0.0))

    def test_call_arbitrary_batch_dims_output_shape(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(3)

        n_out = 3
        fis = TSK.init(input_vars=input_vars, n_out=n_out, order=2, key=key)

        # x shape (..., n_inps)
        x = jax.random.uniform(jax.random.PRNGKey(33), (2, 4, 7, fis.n_inps), minval=0.0, maxval=1.0)
        y = fis(x)

        self.assertEqual(y.shape, (2, 4, 7, n_out))
        self.assertTrue(jnp.all(jnp.isfinite(y)))

    def test_wrong_input_last_dim_raises(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(4)
        fis = TSK.init(input_vars=input_vars, n_out=1, order=1, key=key)

        # last dim should be n_inps=2, but here it's 3
        x_bad = jnp.zeros((5, 3), dtype=jnp.float32)

        with self.assertRaises(eqx.EquinoxTracetimeError):
            _ = fis.fuzzify(x_bad)

    def test_rulebase_sparse_rule_count(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(6)

        # Sparse rules use (var_idx, mf_idx). Here n_vars=2.
        rules = [
            [(0, 0), (1, 1)],
            [(1, 2)],
            [(0, 1)],
        ]

        fis = TSK.init(
            input_vars=input_vars,
            n_out=1,
            order=1,
            antecedents=rules,
            key=key,
        )

        self.assertEqual(fis.rb.n_rules, len(rules))

    def test_order0_output_independent_of_x(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(10)

        fis = TSK.init(input_vars=input_vars, n_out=3, order=0, key=key)

        # Same w, different x -> same y for order 0
        B = 8
        x1 = jax.random.uniform(jax.random.PRNGKey(11), (B, fis.n_inps), minval=0.0, maxval=1.0)
        x2 = jax.random.uniform(jax.random.PRNGKey(12), (B, fis.n_inps), minval=0.0, maxval=1.0)

        w = jnp.abs(jax.random.normal(jax.random.PRNGKey(13), (B, fis.rb.n_rules))) + 0.2

        y1 = fis.defuzzify(w, x1)
        y2 = fis.defuzzify(w, x2)

        self.assertTrue(jnp.allclose(y1, y2, rtol=1e-6, atol=1e-6))

    def test_weight_scaling_invariance(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(20)
        fis = TSK.init(input_vars=input_vars, n_out=2, order=2, key=key, eps=1e-6)

        B = 10
        x = jax.random.uniform(jax.random.PRNGKey(21), (B, fis.n_inps), minval=0.0, maxval=1.0)
        w = jnp.abs(jax.random.normal(jax.random.PRNGKey(22), (B, fis.rb.n_rules))) + 0.1

        y = fis.defuzzify(w, x)
        y_scaled = fis.defuzzify(7.5 * w, x)

        self.assertTrue(jnp.allclose(y, y_scaled, rtol=1e-5, atol=1e-6))

    def test_defuzzify_all_weights_zero_is_finite(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(30)
        fis = TSK.init(input_vars=input_vars, n_out=2, order=1, key=key, eps=1e-6)

        B = 6
        x = jnp.zeros((B, fis.n_inps), dtype=jnp.float32)

        # exactly zero weights
        w = jnp.zeros((B, fis.rb.n_rules), dtype=jnp.float32)

        y = fis.defuzzify(w, x)
        self.assertTrue(jnp.all(jnp.isfinite(y)))
        self.assertEqual(y.shape, (B, fis.n_out))

    def test_call_jittable(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(40)
        fis = TSK.init(input_vars=input_vars, n_out=2, order=2, key=key)

        x = jax.random.uniform(jax.random.PRNGKey(41), (3, 5, fis.n_inps), minval=0.0, maxval=1.0)

        f = eqx.filter_jit(fis)
        y = f(x)

        self.assertEqual(y.shape, (3, 5, fis.n_out))
        self.assertTrue(jnp.all(jnp.isfinite(y)))

    def test_grad_wrt_consequents_nonzero(self):
        input_vars = self._make_input_vars()
        key = jax.random.PRNGKey(50)
        fis = TSK.init(input_vars=input_vars, n_out=1, order=1, key=key, eps=1e-6)

        x = jax.random.uniform(jax.random.PRNGKey(51), (4, fis.n_inps), minval=0.0, maxval=1.0)
        w = jnp.abs(jax.random.normal(jax.random.PRNGKey(52), (4, fis.rb.n_rules))) + 0.1

        def loss_fn(model):
            y = model.defuzzify(w, x)  # (B,1)
            return jnp.mean(y**2)

        grads = eqx.filter_grad(loss_fn)(fis)

        # Consequents should have a gradient tensor of same shape and not be all zeros.
        self.assertIsNotNone(grads.consequents)
        self.assertEqual(grads.consequents.shape, fis.consequents.shape)
        self.assertTrue(jnp.any(grads.consequents != 0.0))

    def test_phi2_feature_ordering_n2(self):
        # build 2-input model
        fv2 = FuzzyVariable.ruspini(n_mfs=2, kind="triangle", minval=0.0, maxval=1.0, name="x1")
        fv3 = FuzzyVariable.ruspini(n_mfs=3, kind="triangle", minval=0.0, maxval=1.0, name="x2")
        input_vars = (fv2, fv3)

        fis = TSK.init(input_vars=input_vars, n_out=1, order=2, key=jax.random.PRNGKey(60))

        x = jnp.array([[2.0, 3.0]], dtype=jnp.float32)  # (1,2)
        phi = fis.phi(x)[0]  # (F,)

        expected = jnp.array([1.0, 2.0, 3.0, 4.0, 6.0, 9.0], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(phi, expected, rtol=0.0, atol=0.0))

