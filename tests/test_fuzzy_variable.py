"""Tests fuzzy variable class."""
import unittest

import jax
import jax.numpy as jnp
import equinox as eqx

from flex import FuzzyVariable
from flex.mfs import LeftShoulder, RightShoulder, Triangle, Trapezoid, Gaussian


class TestRuspini(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            FuzzyVariable.ruspini(n_mfs=1, kind="triangle")
        with self.assertRaises(ValueError):
            FuzzyVariable.ruspini(n_mfs=3, kind="bad_kind")

        with self.assertRaises(ValueError):
            FuzzyVariable.ruspini(n_mfs=3, kind="triangle", minval=1.0, maxval=1.0)
        with self.assertRaises(ValueError):
            FuzzyVariable.ruspini(n_mfs=3, kind="triangle", minval=2.0, maxval=1.0)

        with self.assertRaises(ValueError):
            FuzzyVariable.ruspini(n_mfs=3, kind="triangle", init="bad_init")
        with self.assertRaises(ValueError):
            FuzzyVariable.ruspini(n_mfs=3, kind="triangle", init="noisy", key=None)

        with self.assertRaises(ValueError):
            FuzzyVariable.ruspini(n_mfs=4, kind="triangle", mf_names=["a", "b"])

    def test_triangle_wiring(self):
        n_mfs = 5
        names = [f"mf_{i}" for i in range(n_mfs)]
        fv = FuzzyVariable.ruspini(n_mfs=n_mfs, kind="triangle", mf_names=names)

        self.assertEqual(tuple(fv.params.gaps.shape), (n_mfs - 1,))

        self.assertIsInstance(fv.mfs[0], LeftShoulder)
        self.assertIsInstance(fv.mfs[-1], RightShoulder)
        self.assertTrue(all(isinstance(m, Triangle) for m in fv.mfs[1:-1]))

        self.assertEqual(fv.mfs[0].idx, 0)
        self.assertEqual(fv.mfs[-1].idx, n_mfs - 1)
        self.assertEqual([m.idx for m in fv.mfs[1:-1]], list(range(1, n_mfs - 1)))

        self.assertEqual(fv.mf_names, names)

        self.assertEqual(tuple(fv(0.2).shape), (n_mfs,))
        x = jnp.linspace(0.0, 1.0, 11)
        self.assertEqual(tuple(fv(x).shape), (x.shape[0], n_mfs))

    def test_trapezoid_wiring(self):
        n_mfs = 6
        names = [f"mf_{i}" for i in range(n_mfs)]
        fv = FuzzyVariable.ruspini(n_mfs=n_mfs, kind="trapezoid", mf_names=names)

        self.assertEqual(tuple(fv.params.gaps.shape), (2 * n_mfs - 1,))

        self.assertIsInstance(fv.mfs[0], LeftShoulder)
        self.assertIsInstance(fv.mfs[-1], RightShoulder)
        self.assertTrue(all(isinstance(m, Trapezoid) for m in fv.mfs[1:-1]))

        self.assertEqual(fv.mfs[0].idx, 0)
        self.assertEqual(fv.mfs[-1].idx, 2 * n_mfs - 3)
        self.assertEqual([m.idx for m in fv.mfs[1:-1]], [2 * i - 1 for i in range(1, n_mfs - 1)])

        self.assertEqual(fv.mf_names, names)

        self.assertEqual(tuple(fv(0.7).shape), (n_mfs,))
        x = jnp.linspace(0.0, 1.0, 9)
        self.assertEqual(tuple(fv(x).shape), (x.shape[0], n_mfs))

    def test_default_names(self):
        n_mfs = 4
        fv = FuzzyVariable.ruspini(n_mfs=n_mfs, kind="triangle")
        self.assertEqual(fv.mf_names, [f"mf_{i+1}" for i in range(n_mfs)])

    def test_noisy_determinism(self):
        key = jax.random.PRNGKey(0)
        fv1 = FuzzyVariable.ruspini(n_mfs=5, kind="triangle", init="noisy", key=key, noise_scaler=0.1)
        fv2 = FuzzyVariable.ruspini(n_mfs=5, kind="triangle", init="noisy", key=key, noise_scaler=0.1)
        self.assertTrue(jnp.array_equal(fv1.params.gaps, fv2.params.gaps))

        fv3 = FuzzyVariable.ruspini(
            n_mfs=5, kind="triangle", init="noisy", key=jax.random.PRNGKey(1), noise_scaler=0.1
        )
        self.assertFalse(jnp.array_equal(fv1.params.gaps, fv3.params.gaps))

    def test_jax_jit_vmap_grad(self):
        fv = FuzzyVariable.ruspini(n_mfs=5, kind="triangle")

        # jit
        f_jit = jax.jit(lambda x: fv(x))
        x = jnp.linspace(0.0, 1.0, 7)
        self.assertEqual(tuple(f_jit(x).shape), (x.shape[0], 5))

        # vmap over scalar inputs
        xs = jnp.linspace(0.0, 1.0, 13)
        y = jax.vmap(fv)(xs)
        self.assertEqual(tuple(y.shape), (xs.shape[0], 5))

        # grad w.r.t. gaps
        def loss_fn(gaps):
            fv2 = eqx.tree_at(lambda v: v.params.gaps, fv, gaps)
            out = fv2(xs)  # (B, n_mfs)
            return jnp.mean(out)

        g = jax.grad(loss_fn)(fv.params.gaps)
        self.assertEqual(tuple(g.shape), tuple(fv.params.gaps.shape))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))


class TestGaussian(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            FuzzyVariable.gaussian(n_mfs=0)

        with self.assertRaises(ValueError):
            FuzzyVariable.gaussian(n_mfs=2, minval=1.0, maxval=1.0)

        with self.assertRaises(ValueError):
            FuzzyVariable.gaussian(n_mfs=2, minval=2.0, maxval=1.0)

        with self.assertRaises(ValueError):
            FuzzyVariable.gaussian(n_mfs=2, init="bad_init")

        with self.assertRaises(ValueError):
            FuzzyVariable.gaussian(n_mfs=2, init="noisy", key=None)

        with self.assertRaises(ValueError):
            FuzzyVariable.gaussian(n_mfs=3, mf_names=["a", "b"])  # wrong length

    def test_wiring_and_param_shapes(self):
        n_mfs = 4
        names = [f"g_{i}" for i in range(n_mfs)]
        fv = FuzzyVariable.gaussian(n_mfs=n_mfs, mf_names=names, minval=-1.0, maxval=2.0)

        # params shapes per your constructor
        self.assertEqual(tuple(fv.params.gaps.shape), (n_mfs + 1,))
        self.assertIsNotNone(fv.params.raw_sigmas)
        self.assertEqual(tuple(fv.params.raw_sigmas.shape), (n_mfs,))

        # MF list
        self.assertEqual(len(fv.mfs), n_mfs)
        self.assertTrue(all(isinstance(m, Gaussian) for m in fv.mfs))

        # Indices: idx=i+1, sig_idx=i
        self.assertEqual([m.idx for m in fv.mfs], [i + 1 for i in range(n_mfs)])
        self.assertEqual([m.sig_idx for m in fv.mfs], list(range(n_mfs)))

        # Names
        self.assertEqual(fv.mf_names, names)

    def test_default_names(self):
        n_mfs = 3
        fv = FuzzyVariable.gaussian(n_mfs=n_mfs)
        self.assertEqual(fv.mf_names, [f"mf_{i+1}" for i in range(n_mfs)])

    def test_forward_shapes(self):
        n_mfs = 5
        fv = FuzzyVariable.gaussian(n_mfs=n_mfs)

        # scalar -> (n_mfs,)
        y0 = fv(0.2)
        self.assertEqual(tuple(y0.shape), (n_mfs,))

        # vector -> (N, n_mfs)
        x = jnp.linspace(0.0, 1.0, 11)
        y = fv(x)
        self.assertEqual(tuple(y.shape), (x.shape[0], n_mfs))

    def test_noisy_determinism(self):
        key = jax.random.PRNGKey(0)
        fv1 = FuzzyVariable.gaussian(n_mfs=4, init="noisy", key=key, noise_scaler=0.1)
        fv2 = FuzzyVariable.gaussian(n_mfs=4, init="noisy", key=key, noise_scaler=0.1)

        self.assertTrue(jnp.array_equal(fv1.params.gaps, fv2.params.gaps))
        self.assertTrue(jnp.array_equal(fv1.params.raw_sigmas, fv2.params.raw_sigmas))

        fv3 = FuzzyVariable.gaussian(n_mfs=4, init="noisy", key=jax.random.PRNGKey(1), noise_scaler=0.1)
        self.assertFalse(jnp.array_equal(fv1.params.gaps, fv3.params.gaps))
        self.assertFalse(jnp.array_equal(fv1.params.raw_sigmas, fv3.params.raw_sigmas))

    def test_jax_jit_vmap_grad(self):
        n_mfs = 4
        fv = FuzzyVariable.gaussian(n_mfs=n_mfs)

        # jit
        f_jit = jax.jit(lambda x: fv(x))
        x = jnp.linspace(0.0, 1.0, 7)
        self.assertEqual(tuple(f_jit(x).shape), (x.shape[0], n_mfs))

        # vmap over scalar inputs
        xs = jnp.linspace(0.0, 1.0, 13)
        y = jax.vmap(fv)(xs)
        self.assertEqual(tuple(y.shape), (xs.shape[0], n_mfs))

        # grad w.r.t. gaps and raw_sigmas (two separate checks)
        def loss_wrt_gaps(gaps):
            fv2 = eqx.tree_at(lambda v: v.params.gaps, fv, gaps)
            out = fv2(xs)
            return jnp.mean(out**2)

        def loss_wrt_raw_sigmas(raw_sigmas):
            fv2 = eqx.tree_at(lambda v: v.params.raw_sigmas, fv, raw_sigmas)
            out = fv2(xs)
            return jnp.mean(out**2)

        g_gaps = jax.grad(loss_wrt_gaps)(fv.params.gaps)
        self.assertEqual(tuple(g_gaps.shape), tuple(fv.params.gaps.shape))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g_gaps))))

        g_sigs = jax.grad(loss_wrt_raw_sigmas)(fv.params.raw_sigmas)
        self.assertEqual(tuple(g_sigs.shape), tuple(fv.params.raw_sigmas.shape))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g_sigs))))


class TestFuzzyVariableManual(unittest.TestCase):
    def test_validation(self):
        # invalid MF
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["triangle", "bad_mf"])

        # bounds
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["triangle"], minval=1.0, maxval=1.0)
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["triangle"], minval=2.0, maxval=1.0)

        # init
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["triangle"], init="bad_init")

        # noisy requires key
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["triangle"], init="noisy", key=None)

        # mf_names length
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["triangle", "trapezoid"], mf_names=["only_one"])

        # manual mode: shoulders placement rules
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["triangle", "left_shoulder"])  # left not first
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["left_shoulder", "triangle", "left_shoulder"])  # left twice
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["right_shoulder", "triangle"])  # right not last
        with self.assertRaises(ValueError):
            FuzzyVariable.manual(mfs=["triangle", "right_shoulder", "right_shoulder"])  # right twice

        # params specification not yet implemented
        with self.assertRaises(NotImplementedError):
            FuzzyVariable.manual(mfs=["triangle"], params=[[0.0, 0.5, 1.0]])

    def test_wiring_mixed_types_and_param_shapes(self):
        # Mixed case: LS, tri, gauss, trap, RS
        mfs = ["left_shoulder", "triangle", "gaussian", "trapezoid", "right_shoulder"]
        names = ["LS", "T", "G", "TZ", "RS"]
        fv = FuzzyVariable.manual(mfs=mfs, mf_names=names, minval=-1.0, maxval=2.0)

        # Param shapes: nn = tri(1) + gauss(1) + trap(2) = 4 => gaps shape (nn+1,) = (5,)
        # ns = gauss(1) => raw_sigmas shape (1,)
        self.assertEqual(tuple(fv.params.gaps.shape), (5,))
        self.assertIsNotNone(fv.params.raw_sigmas)
        self.assertEqual(tuple(fv.params.raw_sigmas.shape), (1,))

        # MF types in order
        self.assertIsInstance(fv.mfs[0], LeftShoulder)
        self.assertIsInstance(fv.mfs[1], Triangle)
        self.assertIsInstance(fv.mfs[2], Gaussian)
        self.assertIsInstance(fv.mfs[3], Trapezoid)
        self.assertIsInstance(fv.mfs[4], RightShoulder)

        # Names preserved
        self.assertEqual(fv.mf_names, names)

        # Index wiring (from your manual() logic):
        # LS idx=0
        # Triangle after LS uses n=0 -> idx=0 then n=1
        # Gaussian uses idx=1, sig_idx=0 then n=2
        # Trapezoid uses idx=2 then n=4
        # RS uses idx=4
        self.assertEqual(fv.mfs[0].idx, 0)
        self.assertEqual(fv.mfs[1].idx, 0)
        self.assertEqual(fv.mfs[2].idx, 1)
        self.assertEqual(getattr(fv.mfs[2], "sig_idx"), 0)
        self.assertEqual(fv.mfs[3].idx, 2)
        self.assertEqual(fv.mfs[4].idx, 4)

    def test_first_not_left_shoulder_offsets_indexing(self):
        # If first mf is not left_shoulder, your code does n += 1 before creating it.
        mfs = ["triangle", "right_shoulder"]
        fv = FuzzyVariable.manual(mfs=mfs)

        # nn = tri(1) => gaps shape (nn+1,) = (2,)
        self.assertEqual(tuple(fv.params.gaps.shape), (2,))

        # First triangle should start at idx=1 (because of n += 1 when no left_shoulder)
        self.assertIsInstance(fv.mfs[0], Triangle)
        self.assertEqual(fv.mfs[0].idx, 1)

        # Right shoulder uses current n after triangle increments (triangle: n from 1 -> 2)
        self.assertIsInstance(fv.mfs[-1], RightShoulder)
        self.assertEqual(fv.mfs[-1].idx, 2)

    def test_forward_shapes(self):
        mfs = ["left_shoulder", "triangle", "right_shoulder"]
        fv = FuzzyVariable.manual(mfs=mfs)

        # scalar -> (n_mfs,)
        y0 = fv(0.3)
        self.assertEqual(tuple(y0.shape), (len(mfs),))

        # vector -> (N, n_mfs)
        x = jnp.linspace(0.0, 1.0, 10)
        y = fv(x)
        self.assertEqual(tuple(y.shape), (x.shape[0], len(mfs)))

    def test_noisy_determinism(self):
        mfs = ["left_shoulder", "gaussian", "right_shoulder"]  # includes sigmas
        key = jax.random.PRNGKey(0)

        fv1 = FuzzyVariable.manual(mfs=mfs, init="noisy", key=key, noise_scaler=0.1)
        fv2 = FuzzyVariable.manual(mfs=mfs, init="noisy", key=key, noise_scaler=0.1)

        self.assertTrue(jnp.array_equal(fv1.params.gaps, fv2.params.gaps))
        self.assertTrue(jnp.array_equal(fv1.params.raw_sigmas, fv2.params.raw_sigmas))

        fv3 = FuzzyVariable.manual(mfs=mfs, init="noisy", key=jax.random.PRNGKey(1), noise_scaler=0.1)
        self.assertFalse(jnp.array_equal(fv1.params.gaps, fv3.params.gaps))
        self.assertFalse(jnp.array_equal(fv1.params.raw_sigmas, fv3.params.raw_sigmas))

    def test_jax_jit_vmap_grad(self):
        mfs = ["left_shoulder", "triangle", "gaussian", "right_shoulder"]
        fv = FuzzyVariable.manual(mfs=mfs)

        # jit
        f_jit = jax.jit(lambda x: fv(x))
        x = jnp.linspace(0.0, 1.0, 7)
        self.assertEqual(tuple(f_jit(x).shape), (x.shape[0], len(mfs)))

        # vmap
        xs = jnp.linspace(0.0, 1.0, 13)
        y = jax.vmap(fv)(xs)
        self.assertEqual(tuple(y.shape), (xs.shape[0], len(mfs)))

        # grad w.r.t gaps
        def loss_wrt_gaps(gaps):
            fv2 = eqx.tree_at(lambda v: v.params.gaps, fv, gaps)
            out = fv2(xs)
            return jnp.mean(out**2)

        g = jax.grad(loss_wrt_gaps)(fv.params.gaps)
        self.assertEqual(tuple(g.shape), tuple(fv.params.gaps.shape))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

        # grad w.r.t raw_sigmas if present
        if fv.params.raw_sigmas is not None and fv.params.raw_sigmas.size > 0:
            def loss_wrt_raw_sigmas(raw_sigmas):
                fv2 = eqx.tree_at(lambda v: v.params.raw_sigmas, fv, raw_sigmas)
                out = fv2(xs)
                return jnp.mean(out**2)

            gs = jax.grad(loss_wrt_raw_sigmas)(fv.params.raw_sigmas)
            self.assertEqual(tuple(gs.shape), tuple(fv.params.raw_sigmas.shape))
            self.assertTrue(bool(jnp.all(jnp.isfinite(gs))))
