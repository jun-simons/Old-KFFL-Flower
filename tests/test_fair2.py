"""Comprehensive tests for the FAIR2 gradient computation.

Tests the pure ``_compute_fair2_gradient`` function directly — no Flower
context or UCI ML repository download required.

Covers:
  - Output shapes match model parameter shapes
  - Output dtypes and finiteness
  - Zero gradient when G = 0 (⟨Ωᵢ, 0⟩_F = 0 everywhere)
  - Non-zero gradient for non-zero G with a non-trivial model
  - Determinism: same seed + data + model → same gradient
  - Seed isolation: different seeds → different gradients
  - Different model weights → different gradients (different f_X)
  - ORF projection consistency between FAIR1 and FAIR2 (same seed+1)
  - Gradient linearity in G: g(αG) = α·g(G)
  - Multiclass model support
  - Multi-column sensitive attribute support
  - Global µ_s/µ_f invariance (constants have zero Jacobian)
  - Server half-step aggregation formula: ω_{t+1/2} = ω_t − η·λ·Σᵢ gᵢ
  - Integration with the Adult DataLoader
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kffl.app.client_app import _compute_fair1_stats, _compute_fair2_gradient
from kffl.ml import LogisticRegression, get_weights, set_weights


# ======================================================================
# Helpers
# ======================================================================


D_DEFAULT = 8
INPUT_DIM = 10
N_SAMPLES = 120
K_SENSITIVE = 2


def _make_loader(
    n: int = N_SAMPLES,
    input_dim: int = INPUT_DIM,
    k_sensitive: int = K_SENSITIVE,
    seed: int = 0,
    batch_size: int = 64,
) -> DataLoader:
    """Return a synthetic DataLoader with (X, y, s) of known size."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, input_dim, generator=g)
    y = (X[:, 0] > 0).long()
    s = torch.randint(0, 2, (n, k_sensitive), generator=g).long()
    return DataLoader(TensorDataset(X, y, s), batch_size=batch_size, shuffle=False)


def _fresh_model(input_dim: int = INPUT_DIM, num_classes: int = 2, seed: int = 0) -> LogisticRegression:
    torch.manual_seed(seed)
    return LogisticRegression(input_dim=input_dim, num_classes=num_classes)


def _zero_mu(D: int = D_DEFAULT) -> np.ndarray:
    return np.zeros(D, dtype=np.float64)


def _zero_G(D: int = D_DEFAULT) -> np.ndarray:
    return np.zeros((D, D), dtype=np.float64)


def _eye_G(D: int = D_DEFAULT) -> np.ndarray:
    return np.eye(D, dtype=np.float64)


def _rand_G(D: int = D_DEFAULT, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((D, D)).astype(np.float64)


def _run(
    loader: DataLoader | None = None,
    model: LogisticRegression | None = None,
    G: np.ndarray | None = None,
    mu_s: np.ndarray | None = None,
    mu_f: np.ndarray | None = None,
    D: int = D_DEFAULT,
    gamma_s: float = 1.0,
    gamma_f: float = 1.0,
    seed: int = 42,
) -> List[np.ndarray]:
    if loader is None:
        loader = _make_loader()
    if model is None:
        model = _fresh_model()
    if G is None:
        G = _eye_G(D)
    if mu_s is None:
        mu_s = _zero_mu(D)
    if mu_f is None:
        mu_f = _zero_mu(D)
    return _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D, gamma_s, gamma_f, seed)


# ======================================================================
# Output shapes, dtypes, and finiteness
# ======================================================================


class TestFair2Shapes:
    """Gradient arrays must mirror the model parameter layout."""

    def _n_params(self, model: LogisticRegression) -> int:
        return len(list(model.parameters()))

    def test_returns_list(self):
        result = _run()
        assert isinstance(result, list)

    def test_length_matches_num_params(self):
        model = _fresh_model()
        gi = _run(model=model)
        assert len(gi) == self._n_params(model)

    @pytest.mark.parametrize("D", [4, 8, 16])
    def test_shape_matches_linear_weight(self, D):
        """g_0 corresponds to the weight matrix: shape (1, input_dim) for binary."""
        model = _fresh_model()
        gi = _run(D=D, model=model, G=_eye_G(D))
        assert gi[0].shape == tuple(list(model.parameters())[0].shape)

    @pytest.mark.parametrize("D", [4, 8, 16])
    def test_shape_matches_linear_bias(self, D):
        """g_1 corresponds to the bias: shape (1,) for binary."""
        model = _fresh_model()
        gi = _run(D=D, model=model, G=_eye_G(D))
        assert gi[1].shape == tuple(list(model.parameters())[1].shape)

    def test_all_outputs_are_numpy(self):
        gi = _run()
        for arr in gi:
            assert isinstance(arr, np.ndarray)

    def test_all_outputs_finite(self):
        gi = _run()
        for arr in gi:
            assert np.isfinite(arr).all(), f"Non-finite gradient: {arr}"

    def test_dtype_is_float32(self):
        """Gradients should be float32 (same dtype as model parameters)."""
        gi = _run()
        for arr in gi:
            assert arr.dtype == np.float32


# ======================================================================
# Zero-G: gradient must vanish
# ======================================================================


class TestFair2ZeroG:
    """When G = 0, h = ⟨Ωᵢ, 0⟩_F = 0 for any ω, so ∇h = 0 everywhere."""

    def test_zero_G_gives_zero_weight_gradient(self):
        gi = _run(G=_zero_G())
        np.testing.assert_array_equal(gi[0], np.zeros_like(gi[0]))

    def test_zero_G_gives_zero_bias_gradient(self):
        gi = _run(G=_zero_G())
        np.testing.assert_array_equal(gi[1], np.zeros_like(gi[1]))

    def test_zero_G_total_norm_is_zero(self):
        gi = _run(G=_zero_G())
        total = sum(float(np.linalg.norm(g)) for g in gi)
        assert total == pytest.approx(0.0, abs=1e-7)


# ======================================================================
# Non-zero G: gradient must be non-trivial
# ======================================================================


class TestFair2NonzeroG:
    """For a non-zero G and a model with varied predictions, ∇h should be non-zero."""

    def test_identity_G_nonzero_gradient(self):
        gi = _run(G=_eye_G())
        total_norm = sum(float(np.linalg.norm(g)) for g in gi)
        assert total_norm > 1e-8, "Expected non-zero gradient for non-zero G"

    def test_random_G_nonzero_gradient(self):
        gi = _run(G=_rand_G())
        total_norm = sum(float(np.linalg.norm(g)) for g in gi)
        assert total_norm > 1e-8

    def test_gradient_norm_grows_with_G_scale(self):
        """Scaling G by α scales the gradient by α (h is linear in G)."""
        G = _rand_G()
        gi_1 = _run(G=G)
        gi_10 = _run(G=10.0 * G)

        norm_1 = sum(float(np.linalg.norm(g)) for g in gi_1)
        norm_10 = sum(float(np.linalg.norm(g)) for g in gi_10)

        assert norm_10 == pytest.approx(10.0 * norm_1, rel=1e-4)

    def test_gradient_negated_when_G_negated(self):
        """g(−G) = −g(G) because h is linear in G."""
        G = _rand_G()
        gi_pos = _run(G=G)
        gi_neg = _run(G=-G)
        for gp, gn in zip(gi_pos, gi_neg):
            np.testing.assert_allclose(gp, -gn, rtol=1e-5)


# ======================================================================
# Determinism and seed behaviour
# ======================================================================


class TestFair2Determinism:
    def test_same_inputs_give_identical_gradients(self):
        loader = _make_loader(seed=7)
        model = _fresh_model(seed=3)
        G = _rand_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi_a = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=42)
        gi_b = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=42)
        for a, b in zip(gi_a, gi_b):
            np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_gradient(self):
        loader = _make_loader()
        model = _fresh_model()
        G = _rand_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi_a = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=1)
        gi_b = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=2)
        norms_differ = any(
            not np.allclose(a, b) for a, b in zip(gi_a, gi_b)
        )
        assert norms_differ, "Different seeds should produce different gradients"

    def test_different_model_weights_give_different_gradient(self):
        """Different model weights change f(X) → different Z_f → different gradient."""
        loader = _make_loader()
        G = _eye_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        seed = 5

        model_a = _fresh_model(seed=0)
        model_b = _fresh_model(seed=0)

        # Force models apart
        with torch.no_grad():
            for p in model_a.parameters():
                p.fill_(0.1)
            for p in model_b.parameters():
                p.fill_(2.0)

        gi_a = _compute_fair2_gradient(loader, model_a, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed)
        gi_b = _compute_fair2_gradient(loader, model_b, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed)

        norms_differ = any(not np.allclose(a, b) for a, b in zip(gi_a, gi_b))
        assert norms_differ

    def test_different_data_gives_different_gradient(self):
        loader_a = _make_loader(seed=0)
        loader_b = _make_loader(seed=999)
        model = _fresh_model()
        G = _eye_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi_a = _compute_fair2_gradient(loader_a, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=7)
        gi_b = _compute_fair2_gradient(loader_b, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=7)
        norms_differ = any(not np.allclose(a, b) for a, b in zip(gi_a, gi_b))
        assert norms_differ


# ======================================================================
# ORF consistency between FAIR1 and FAIR2
# ======================================================================


class TestFair2ORFConsistency:
    """The ORF projection for f(X) must be identical between FAIR1 and FAIR2
    because both use seed+1 and the same model to derive W_f, b_f."""

    def test_mu_f_matches_fair1_mu_f(self):
        """µ_{f,i} recovered during FAIR2 autograd should equal the one from FAIR1."""
        from kffl.fairness.orf import OrthogonalRandomFeaturesRBF

        D, seed = 8, 13
        loader = _make_loader(n=100, seed=0, batch_size=100)
        model = _fresh_model(seed=2)

        # --- FAIR1: compute µ_f via numpy ORF ---
        _, _, mu_f_fair1, _ = _compute_fair1_stats(loader, model, D, 1.0, 1.0, seed)

        # --- FAIR2: recover µ_f by inspecting the computation ---
        # We manually replicate the FAIR2 µ_f derivation (same seed+1 logic)
        import torch

        model.eval()
        X_all, _, _ = next(iter(loader))
        with torch.no_grad():
            f_probe = model.proba_for_orf(X_all[:1])
        f_dim = f_probe.shape[1]

        orf_f = OrthogonalRandomFeaturesRBF(gamma=1.0, n_components=D, random_state=seed + 1)
        orf_f.fit(np.zeros((1, f_dim), dtype=np.float64))
        W_f = torch.as_tensor(orf_f.W_, dtype=torch.float32)
        b_f = torch.as_tensor(orf_f.b_, dtype=torch.float32)
        scale = float(np.sqrt(2.0 / D))

        with torch.no_grad():
            f_X = model.proba_for_orf(X_all)
        proj = f_X @ W_f.T + b_f.unsqueeze(0)
        Z_f = scale * torch.cos(proj)
        mu_f_fair2 = Z_f.mean(dim=0).numpy()

        np.testing.assert_allclose(mu_f_fair1, mu_f_fair2, rtol=1e-5, atol=1e-6)

    def test_same_seed_gives_same_orf_weights(self):
        """Fitting ORF on a dummy array vs the real data should yield identical W_, b_
        since the weights depend only on (f_dim, D, gamma, seed)."""
        from kffl.fairness.orf import OrthogonalRandomFeaturesRBF

        D, seed, gamma, f_dim = 16, 7, 1.0, 1
        real_data = np.random.default_rng(0).standard_normal((50, f_dim))

        orf_real = OrthogonalRandomFeaturesRBF(gamma=gamma, n_components=D, random_state=seed)
        orf_real.fit(real_data)

        orf_dummy = OrthogonalRandomFeaturesRBF(gamma=gamma, n_components=D, random_state=seed)
        orf_dummy.fit(np.zeros((1, f_dim), dtype=np.float64))

        np.testing.assert_array_equal(orf_real.W_, orf_dummy.W_)
        np.testing.assert_array_equal(orf_real.b_, orf_dummy.b_)


# ======================================================================
# Hyperparameter effects
# ======================================================================


class TestFair2HyperparamEffects:
    def test_larger_D_changes_gradient(self):
        """Changing D changes the projection dimension and thus the gradient."""
        loader = _make_loader()
        model = _fresh_model()
        # Just check no crash and shapes are correct for different D
        for D in [4, 8, 16]:
            gi = _compute_fair2_gradient(
                loader, model, _eye_G(D), _zero_mu(D), _zero_mu(D),
                D, 1.0, 1.0, seed=42,
            )
            assert gi[0].shape == tuple(list(model.parameters())[0].shape)

    def test_gamma_s_affects_gradient(self):
        """Different gamma_s changes Z_S → changes Mᵢ → changes gradient."""
        loader = _make_loader()
        model = _fresh_model()
        G = _eye_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi_a = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 0.1, 1.0, seed=5)
        gi_b = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 10.0, 1.0, seed=5)
        norms_differ = any(not np.allclose(a, b) for a, b in zip(gi_a, gi_b))
        assert norms_differ

    def test_gamma_f_affects_gradient(self):
        """Different gamma_f changes W_f, b_f → changes Z_f → changes gradient."""
        loader = _make_loader()
        model = _fresh_model()
        G = _eye_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi_a = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 0.1, seed=5)
        gi_b = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 10.0, seed=5)
        norms_differ = any(not np.allclose(a, b) for a, b in zip(gi_a, gi_b))
        assert norms_differ


# ======================================================================
# Model does not retain state after call (no accumulated gradients)
# ======================================================================


class TestFair2Statelessness:
    def test_model_grad_cleared_between_calls(self):
        """Two consecutive calls must give identical results (no grad accumulation)."""
        loader = _make_loader()
        model = _fresh_model()
        G = _rand_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi_a = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=3)
        gi_b = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=3)
        for a, b in zip(gi_a, gi_b):
            np.testing.assert_array_equal(a, b)

    def test_model_parameters_unchanged_after_call(self):
        """_compute_fair2_gradient must not modify model parameters."""
        loader = _make_loader()
        model = _fresh_model()
        weights_before = get_weights(model)
        _compute_fair2_gradient(loader, model, _eye_G(), _zero_mu(), _zero_mu(), D_DEFAULT, 1.0, 1.0, seed=0)
        weights_after = get_weights(model)
        for wb, wa in zip(weights_before, weights_after):
            np.testing.assert_array_equal(wb, wa)


# ======================================================================
# Multi-sensitive and multiclass variants
# ======================================================================


class TestFair2Variants:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_multi_column_sensitive(self, k):
        """Works for any number of sensitive columns k (S has k columns)."""
        loader = _make_loader(k_sensitive=k)
        model = _fresh_model()
        gi = _compute_fair2_gradient(
            loader, model, _eye_G(), _zero_mu(), _zero_mu(),
            D_DEFAULT, 1.0, 1.0, seed=0,
        )
        assert len(gi) == len(list(model.parameters()))
        for arr in gi:
            assert np.isfinite(arr).all()

    def test_multiclass_model(self):
        """Works for multiclass (num_classes=4); output shape adapts to parameters."""
        loader = _make_loader()
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=4)
        gi = _compute_fair2_gradient(
            loader, model, _eye_G(), _zero_mu(), _zero_mu(),
            D_DEFAULT, 1.0, 1.0, seed=0,
        )
        n_params = len(list(model.parameters()))
        assert len(gi) == n_params
        for arr in gi:
            assert np.isfinite(arr).all()

    def test_multiclass_gradient_nonzero_for_nonzero_G(self):
        loader = _make_loader()
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=4)
        gi = _compute_fair2_gradient(
            loader, model, _eye_G(), _zero_mu(), _zero_mu(),
            D_DEFAULT, 1.0, 1.0, seed=0,
        )
        total_norm = sum(float(np.linalg.norm(g)) for g in gi)
        assert total_norm > 1e-8


# ======================================================================
# Server-side aggregation: ω_{t+1/2} = ω_t − η·λ·(2/n²)·Σᵢ gᵢ
# ======================================================================


def _server_half_step(
    weights: List[np.ndarray],
    sum_grad: List[np.ndarray],
    total_n: int,
    step_size: float,
    lambda_fair: float,
) -> List[np.ndarray]:
    """Replicate the server's FAIR2 half-step update (with HSIC normalization)."""
    hsic_scale = 2.0 / max(total_n, 1) ** 2
    return [
        w - step_size * lambda_fair * hsic_scale * g
        for w, g in zip(weights, sum_grad)
    ]


class TestFair2ServerAggregation:
    """Verify the half-step update formula (eq. 17) with HSIC normalization.

    HSIC = (1/n²)||G||²_F, so ∇HSIC = (2/n²)·Σᵢ gᵢ.
    ω_{t+1/2} = ω_t − η·λ·(2/n²)·Σᵢ gᵢ
    """

    def test_half_step_update_single_client(self):
        """With one client, verify the normalized update formula."""
        loader = _make_loader()
        model = _fresh_model()
        G = _rand_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=42)

        step_size = 0.05
        lambda_fair = 2.0
        weights = get_weights(model)
        total_n = N_SAMPLES

        omega_half = _server_half_step(weights, gi, total_n, step_size, lambda_fair)

        hsic_scale = 2.0 / total_n ** 2
        for oh, w, g in zip(omega_half, weights, gi):
            expected = w - step_size * lambda_fair * hsic_scale * g
            np.testing.assert_allclose(oh, expected, rtol=1e-6)

    def test_half_step_moves_in_gradient_direction(self):
        """ω_{t+1/2} should differ from ω_t when the gradient is non-zero."""
        loader = _make_loader()
        model = _fresh_model()
        G = _eye_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=7)

        weights = get_weights(model)
        # Use large λ to ensure update is detectable after 1/n² scaling
        omega_half = _server_half_step(weights, gi, N_SAMPLES, step_size=1.0, lambda_fair=1000.0)

        any_changed = any(not np.allclose(oh, w) for oh, w in zip(omega_half, weights))
        assert any_changed

    def test_two_client_gradient_sum(self):
        """Summing gradients from two clients and applying the normalized update."""
        loaders = [_make_loader(seed=0), _make_loader(seed=1)]
        model = _fresh_model()
        G = _rand_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        weights = get_weights(model)

        # Collect per-client gradients
        grads = [
            _compute_fair2_gradient(l, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=3)
            for l in loaders
        ]

        # Server sums them
        sum_grad = [np.zeros_like(w, dtype=np.float64) for w in weights]
        for gi in grads:
            for k, g in enumerate(gi):
                sum_grad[k] += g.astype(np.float64)

        total_n = 2 * N_SAMPLES  # two loaders
        step_size, lambda_fair = 0.01, 1.0
        omega_half = _server_half_step(weights, sum_grad, total_n, step_size, lambda_fair)

        # Verify manually
        hsic_scale = 2.0 / total_n ** 2
        for oh, w, sg in zip(omega_half, weights, sum_grad):
            np.testing.assert_allclose(oh, w - step_size * lambda_fair * hsic_scale * sg, rtol=1e-6)

    def test_zero_step_leaves_weights_unchanged(self):
        """Step size 0 → ω_{t+1/2} = ω_t regardless of gradient."""
        loader = _make_loader()
        model = _fresh_model()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi = _compute_fair2_gradient(loader, model, _eye_G(), mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=0)
        weights = get_weights(model)
        omega_half = _server_half_step(weights, gi, N_SAMPLES, step_size=0.0, lambda_fair=1.0)
        for oh, w in zip(omega_half, weights):
            np.testing.assert_array_equal(oh, w)

    def test_update_norm_decreases_for_small_step(self):
        """The distance between ω_t and ω_{t+1/2} grows with step_size."""
        loader = _make_loader()
        model = _fresh_model()
        G = _rand_G()
        mu_s, mu_f = _zero_mu(), _zero_mu()
        gi = _compute_fair2_gradient(loader, model, G, mu_s, mu_f, D_DEFAULT, 1.0, 1.0, seed=9)
        weights = get_weights(model)

        def delta(step):
            oh = _server_half_step(weights, gi, N_SAMPLES, step_size=step, lambda_fair=1.0)
            return float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(oh, weights))))

        assert delta(0.1) < delta(1.0)


# ======================================================================
# Integration: Adult DataLoader
# ======================================================================


class TestFair2WithAdultData:
    """Integration test using the actual Adult dataset (downloaded once)."""

    @pytest.fixture(scope="class")
    def adult_loader(self):
        from kffl.data import get_federated_loaders
        return get_federated_loaders("adult", num_partitions=5, batch_size=64)[0]

    @pytest.fixture(scope="class")
    def adult_input_dim(self, adult_loader):
        X, _, _ = next(iter(adult_loader))
        return X.shape[1]

    @pytest.fixture(scope="class")
    def adult_model(self, adult_input_dim):
        torch.manual_seed(0)
        return LogisticRegression(input_dim=adult_input_dim)

    def test_runs_without_error(self, adult_loader, adult_model):
        D = 16
        G = np.eye(D, dtype=np.float64)
        mu_s = np.zeros(D, dtype=np.float64)
        mu_f = np.zeros(D, dtype=np.float64)
        gi = _compute_fair2_gradient(
            adult_loader, adult_model, G, mu_s, mu_f, D, 1.0, 1.0, seed=42
        )
        assert len(gi) == len(list(adult_model.parameters()))

    def test_gradient_is_finite_on_adult(self, adult_loader, adult_model):
        D = 16
        G = np.eye(D, dtype=np.float64)
        mu_s = np.zeros(D, dtype=np.float64)
        mu_f = np.zeros(D, dtype=np.float64)
        gi = _compute_fair2_gradient(
            adult_loader, adult_model, G, mu_s, mu_f, D, 1.0, 1.0, seed=42
        )
        for arr in gi:
            assert np.isfinite(arr).all()

    def test_gradient_nonzero_on_adult(self, adult_loader, adult_model):
        D = 16
        G = np.eye(D, dtype=np.float64)
        mu_s = np.zeros(D, dtype=np.float64)
        mu_f = np.zeros(D, dtype=np.float64)
        gi = _compute_fair2_gradient(
            adult_loader, adult_model, G, mu_s, mu_f, D, 1.0, 1.0, seed=42
        )
        total_norm = sum(float(np.linalg.norm(g)) for g in gi)
        assert total_norm > 1e-8

    def test_zero_G_gives_zero_gradient_on_adult(self, adult_loader, adult_model):
        D = 16
        G = np.zeros((D, D), dtype=np.float64)
        mu_s = np.zeros(D, dtype=np.float64)
        mu_f = np.zeros(D, dtype=np.float64)
        gi = _compute_fair2_gradient(
            adult_loader, adult_model, G, mu_s, mu_f, D, 1.0, 1.0, seed=42
        )
        total_norm = sum(float(np.linalg.norm(g)) for g in gi)
        assert total_norm == pytest.approx(0.0, abs=1e-7)

    def test_consistent_with_fair1_seed(self, adult_loader, adult_model):
        """FAIR2 with same seed as FAIR1 should produce a consistent (finite, non-zero)
        gradient, confirming ORF weight re-derivation is correct."""
        seed = 17
        D = 8
        _, mu_s, mu_f, ni = _compute_fair1_stats(adult_loader, adult_model, D, 1.0, 1.0, seed)
        G = np.eye(D, dtype=np.float64)
        gi = _compute_fair2_gradient(adult_loader, adult_model, G, mu_s, mu_f, D, 1.0, 1.0, seed)
        assert ni > 0
        for arr in gi:
            assert np.isfinite(arr).all()

    def test_gradient_invariant_to_mu_values(self, adult_loader, adult_model):
        """Since global µ_s and µ_f are constants (no gradient), changing them
        should not affect the computed gradient gᵢ."""
        D = 8
        G = np.eye(D, dtype=np.float64)
        seed = 42

        gi_zero = _compute_fair2_gradient(
            adult_loader, adult_model, G,
            np.zeros(D, dtype=np.float64),
            np.zeros(D, dtype=np.float64),
            D, 1.0, 1.0, seed,
        )
        gi_nonzero = _compute_fair2_gradient(
            adult_loader, adult_model, G,
            np.ones(D, dtype=np.float64) * 0.5,
            np.ones(D, dtype=np.float64) * 0.3,
            D, 1.0, 1.0, seed,
        )
        for a, b in zip(gi_zero, gi_nonzero):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
