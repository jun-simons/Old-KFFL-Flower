"""KFFL ClientApp.

Handles the three message types sent by the KFFL server per iteration:

  query.fair1        — Compute and return local interaction terms
                       Φᵢ(ω_t) = { Mᵢ(ω_t), µ_{s,i}, µ_{f,i} }.
                       Uses real ORF kernel features from local data.

  query.fair2        — Given G(ω_t) from the server, compute and return
                       the local fairness gradient gᵢ(ω_t) via autograd.

  query.local_update — [Stub] Return ω_{t+1/2} weights unchanged.
"""

from __future__ import annotations

from logging import INFO
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import logger
from torch.utils.data import DataLoader

from kffl.data.dataset import get_federated_loaders
from kffl.fairness.orf import OrthogonalRandomFeaturesRBF
from kffl.ml import create_model, get_train_fn, get_weights, set_weights

app = ClientApp()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_partition(context: Context) -> DataLoader:
    """Return the DataLoader for this client's local partition."""
    dataset = str(context.run_config.get("dataset", "adult"))
    num_partitions = int(context.run_config.get("num-partitions", 5))
    partition_id = int(context.node_config.get("partition-id", 0))
    batch_size = int(context.run_config.get("batch-size", 64))

    # Optional comma-separated list of sensitive feature names, e.g. "sex" or "sex,race"
    sens_raw = context.run_config.get("sensitive-features", None)
    sensitive_features = (
        [s.strip() for s in str(sens_raw).split(",") if s.strip()]
        if sens_raw is not None
        else None   # use dataset-specific default
    )

    loaders = get_federated_loaders(
        dataset,
        num_partitions=num_partitions,
        batch_size=batch_size,
        sensitive_features=sensitive_features,
    )
    return loaders[partition_id]


def _extract_model(message: Message) -> nn.Module:
    """Reconstruct the model from weights and config in *message*."""
    crec: ConfigRecord = message.content["config"]  # type: ignore[index]
    input_dim = int(crec["input_dim"])
    num_classes = int(crec["num_classes"])
    model_name = str(crec["model_name"])

    model = create_model(model_name, input_dim=input_dim, num_classes=num_classes)
    arec: ArrayRecord = message.content["model"]  # type: ignore[index]
    n_params = len(list(model.parameters()))
    weights = [arec[f"w_{i}"].numpy() for i in range(n_params)]
    set_weights(model, weights)
    return model


def _collect_data(loader: DataLoader, device: torch.device):
    """Collect all batches from *loader* into full tensors on *device*."""
    X_list: List[torch.Tensor] = []
    s_list: List[torch.Tensor] = []
    for X_batch, _y_batch, s_batch in loader:
        X_list.append(X_batch)
        s_list.append(s_batch)
    X_all = torch.cat(X_list).to(device)
    S_all = torch.cat(s_list).numpy().astype(np.float64)   # (n, k)
    return X_all, S_all


# ---------------------------------------------------------------------------
# FAIR1 kernel computation
# ---------------------------------------------------------------------------


def _compute_fair1_stats(
    loader: DataLoader,
    model: nn.Module,
    D: int,
    gamma_s: float,
    gamma_f: float,
    seed: int,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute local FAIR1 statistics: Mᵢ, µ_{s,i}, µ_{f,i}, nᵢ.

    Equations (14), (19), (20) from the paper.
    Both ORFs share *seed* (orf_s) and *seed+1* (orf_f) so that all clients
    sample the same random projections.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    X_all, S_all = _collect_data(loader, dev)
    ni = S_all.shape[0]

    with torch.no_grad():
        f_X = model.predict_proba(X_all).cpu().numpy()  # type: ignore[attr-defined]
    if f_X.ndim == 1:
        f_X = f_X.reshape(-1, 1)

    orf_s = OrthogonalRandomFeaturesRBF(gamma=gamma_s, n_components=D, random_state=seed)
    Z_S = orf_s.fit_transform(S_all)           # (n, D)

    orf_f = OrthogonalRandomFeaturesRBF(gamma=gamma_f, n_components=D, random_state=seed + 1)
    Z_f = orf_f.fit_transform(f_X)             # (n, D)

    Mi = Z_S.T @ Z_f                           # (D, D), eq. 14
    mu_s = Z_S.mean(axis=0)                    # (D,),   eq. 19
    mu_f = Z_f.mean(axis=0)                    # (D,),   eq. 20

    return Mi, mu_s, mu_f, ni


# ---------------------------------------------------------------------------
# FAIR2 gradient computation 
# ---------------------------------------------------------------------------


def _compute_fair2_gradient(
    loader: DataLoader,
    model: nn.Module,
    G: np.ndarray,
    mu_s_global: np.ndarray,
    mu_f_global: np.ndarray,
    D: int,
    gamma_s: float,
    gamma_f: float,
    seed: int,
    device: str = "cpu",
) -> List[np.ndarray]:
    """Compute the local fairness gradient g_i(ω) = J_{Omega_i}(ω)^T G(ω).

    Uses autograd to differentiate

        h(ω) = tr(Omega_i(ω)^T G) = ⟨Omega_i(ω), G⟩_F

    with respect to model parameters ω, where

        Ωᵢ(ω) = M_i(ω) - n_i · mu_s · mu_f^T (TODO this may be incorrect)   (eq. 15)
        Mᵢ(ω)  = Z_{S,i}^T Z_{f,i}(ω)                          (eq. 14)

    mu_s and mu_f are the **global** aggregated means from the server
    (computed during FAIR1 and passed as constants)
    TODO: check on whether one or both of these should be the local mean for Omega_i
    
    The ORF weights (W_f, b_f) are re-derived from *seed+1* identically to
    FAIR1
    
    Parameters
    ----------
    loader:
        DataLoader yielding ``(X, y, s)`` batches.
    model:
        The global model ω_t (loaded with server weights).
    G:
        Global interaction matrix from the server, shape ``(D, D)``.
    mu_s_global:
        Global mean of Z_S from FAIR1, shape ``(D,)``.  Constant.
    mu_f_global:
        Global mean of Z_f from FAIR1, shape ``(D,)``.  Constant.
    D, gamma_s, gamma_f, seed:
        ORF parameters — must match those used in FAIR1.
    device:
        Torch device string.

    Returns
    -------
    List[np.ndarray]
        Per-parameter gradient arrays in the same order/shape as
        ``model.parameters()``.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.train()

    X_all, S_all = _collect_data(loader, dev)
    ni = S_all.shape[0]

    # ---- ORF for S (same seed as FAIR1) ----
    # TODO: see if we want to cache this on a client instead
    orf_s = OrthogonalRandomFeaturesRBF(gamma=gamma_s, n_components=D, random_state=seed)
    Z_S_np = orf_s.fit_transform(S_all)                        # (n, D)
    Z_S = torch.as_tensor(Z_S_np, dtype=torch.float32, device=dev)

    # ---- Global mu_s and mu_f from server----
    # note: these are constants, so no gradient from autograd
    # if we use these to compute Omega_i, the centering term has no gradient
    mu_s = torch.as_tensor(mu_s_global, dtype=torch.float32, device=dev)  # (D,)
    mu_f = torch.as_tensor(mu_f_global, dtype=torch.float32, device=dev)  # (D,)

    # ---- ORF weights for f(X) (seed+1, same as FAIR1) ----
    # We only need the projection matrices W_ and b_, which depend solely on
    # f_dim, D, gamma_f, and seed 
    with torch.no_grad():
        f_probe = model.proba_for_orf(X_all[:1])               # type: ignore[attr-defined]
    f_dim = f_probe.shape[1]

    orf_f = OrthogonalRandomFeaturesRBF(gamma=gamma_f, n_components=D, random_state=seed + 1)
    orf_f.fit(np.zeros((1, f_dim), dtype=np.float64))

    W_f = torch.as_tensor(orf_f.W_, dtype=torch.float32, device=dev)   # (D, f_dim)
    b_f = torch.as_tensor(orf_f.b_, dtype=torch.float32, device=dev)   # (D,)
    scale = float(np.sqrt(2.0 / D))

    # ---- Forward pass with gradient ----
    model.zero_grad()

    f_X = model.proba_for_orf(X_all)                          # (n, f_dim), grad enabled

    proj = f_X @ W_f.T + b_f.unsqueeze(0)                    # (n, D)
    Z_f = scale * torch.cos(proj)                             # (n, D)

    # Mi = Z_S.T @ Z_f                                          # (D, D)
    # Omega_i = Mi - ni * torch.outer(mu_s, mu_f)              # (D, D)
    #
    # G_t = torch.as_tensor(G, dtype=torch.float32, device=dev)
    # h = (Omega_i * G_t).sum()                                 # scalar: ⟨omega_i G⟩_F
    #
    # h.backward()

    Mi = Z_S.T @ Z_f                      # (D, D)
    mu_f_i = Z_f.mean(dim=0)              # (D,), gradient-enabled
    mu_s_global_t = torch.as_tensor(mu_s_global, dtype=torch.float32, device=dev)

    Omega_i = Mi - ni * torch.outer(mu_s_global_t, mu_f_i)

    G_t = torch.as_tensor(G, dtype=torch.float32, device=dev)
    h = (Omega_i * G_t).sum()
    h.backward()

    # ---- Collect per-parameter gradients ----
    gi: List[np.ndarray] = []
    for p in model.parameters():
        if p.grad is not None:
            gi.append(p.grad.detach().cpu().numpy().copy())
        else:
            gi.append(np.zeros(p.shape, dtype=np.float32))

    return gi


# ---------------------------------------------------------------------------
# FAIR1 handler
# ---------------------------------------------------------------------------


@app.query("fair1")
def handle_fair1(message: Message, context: Context) -> Message:
    """Compute Φᵢ(ω_t) = { Mᵢ, µ_{s,i}, µ_{f,i} } and return to server."""
    crec: ConfigRecord = message.content["config"]  # type: ignore[index]
    seed = int(crec["seed"])
    D = int(crec["num_rf"])
    gamma_s = float(crec["gamma_s"])
    gamma_f = float(crec["gamma_f"])

    model = _extract_model(message)
    loader = _load_partition(context)

    Mi, mu_s, mu_f, ni = _compute_fair1_stats(
        loader, model, D, gamma_s, gamma_f, seed
    )

    logger.log(
        INFO,
        "[Client %d] FAIR1: ni=%d  ||Mi||=%.4f  ||µ_s||=%.4f  ||µ_f||=%.4f",
        context.node_id, ni,
        float(np.linalg.norm(Mi)),
        float(np.linalg.norm(mu_s)),
        float(np.linalg.norm(mu_f)),
    )

    reply_content = RecordDict({
        "fair1_data": ArrayRecord({
            "Mi": Array(Mi.astype(np.float32)),
            "mu_s": Array(mu_s.astype(np.float32)),
            "mu_f": Array(mu_f.astype(np.float32)),
            "ni": Array(np.array([ni], dtype=np.int64)),
        }),
    })
    return Message(reply_content, reply_to=message)


# ---------------------------------------------------------------------------
# FAIR2 handler
# ---------------------------------------------------------------------------


@app.query("fair2")
def handle_fair2(message: Message, context: Context) -> Message:
    """Compute local fairness gradient g_i(ω_t) and return to server.

    Equations (15) and (16) from the paper:
        Ωᵢ(ω) = M_i(ω) - n_i · mu_{s,i} · µ_{f,i}^T
        gᵢ(ω)  = J_{Omega_i}(ω)^T G(ω)
    """
    crec: ConfigRecord = message.content["config"]
    seed = int(crec["seed"])
    D = int(crec["num_rf"])
    gamma_s = float(crec["gamma_s"])
    gamma_f = float(crec["gamma_f"])

    model = _extract_model(message)

    arec: ArrayRecord = message.content["fair2_data"]
    G = arec["G"].numpy().astype(np.float64)           # (D, D)
    mu_s_global = arec["mu_s"].numpy().astype(np.float64)  # (D,)
    mu_f_global = arec["mu_f"].numpy().astype(np.float64)  # (D,)

    loader = _load_partition(context)

    gi = _compute_fair2_gradient(
        loader, model, G, mu_s_global, mu_f_global, D, gamma_s, gamma_f, seed
    )

    logger.log(
        INFO,
        "[Client %d] FAIR2: ||gᵢ[0]||=%.4f  ||gᵢ[1]||=%.4f",
        context.node_id,
        float(np.linalg.norm(gi[0])) if gi else 0.0,
        float(np.linalg.norm(gi[1])) if len(gi) > 1 else 0.0,
    )

    reply_content = RecordDict({
        "gradient": ArrayRecord({f"g_{i}": Array(g.astype(np.float32)) for i, g in enumerate(gi)}),
    })
    return Message(reply_content, reply_to=message)


# ---------------------------------------------------------------------------
# Local Update handler
# ---------------------------------------------------------------------------


@app.query("local_update")
def handle_local_update(message: Message, context: Context) -> Message:
    """Solve the proximal local objective and return ωᵢ_{t+1}.

    Equation (21) from the paper:
        ωᵢ_{t+1} = argmin_ω [ fᵢ(ω) + (μ/2)||ω − ω_{t+1/2}||² ]

    The model is initialised at ω_{t+1/2} (received from the server) and
    trained for ``num_local_epochs`` epochs of SGD with the proximal penalty
    anchored at the same ω_{t+1/2}.
    """
    crec: ConfigRecord = message.content["config"]  # type: ignore[index]
    step_size = float(crec["step_size"])
    proximal_alpha = float(crec["proximal_alpha"])
    num_local_epochs = int(crec["num_local_epochs"])
    model_name = str(crec["model_name"])

    # Convert paper notation: objective is f_i(ω) + (1/2α)||ω − ω_{t+1/2}||²
    # train() uses (proximal_mu/2)||ω − center||², so proximal_mu = 1/α
    proximal_mu = 1.0 / proximal_alpha

    # Load model initialised at ω_{t+1/2}
    model = _extract_model(message)
    # ω_{t+1/2} is the proximal centre — weights already extracted into model
    center_weights = get_weights(model)

    loader = _load_partition(context)

    train_fn = get_train_fn(model_name)
    result = train_fn(
        model,
        loader,
        lr=step_size,
        num_epochs=num_local_epochs,
        proximal_mu=proximal_mu,
        proximal_center=center_weights,
    )

    updated_weights = get_weights(model)

    logger.log(
        INFO,
        "[Client %d] Local Update: epochs=%d  loss=%.4f  alpha=%.4f",
        context.node_id, num_local_epochs,
        result.final_loss, proximal_alpha,
    )

    reply_content = RecordDict({
        "model": ArrayRecord({
            f"w_{i}": Array(w.astype(np.float32)) for i, w in enumerate(updated_weights)
        }),
    })
    return Message(reply_content, reply_to=message)
