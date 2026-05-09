"""KFFL ServerApp.

Each KFFL iteration consists of three message rounds per global update:

  FAIR1        — Server sends (ω_t, seed ζ, ORF params) to each client.
                 Clients reply with Φᵢ = {Mᵢ, µ_{s,i}, µ_{f,i}}.
                 Server aggregates into G(ω_t) (eq. 13).

  FAIR2        — Server sends (ω_t, G(ω_t), ORF params) to each client.
                 Clients reply with local fairness gradients gᵢ(ω_t).
                 Server computes ω_{t+1/2} = ω_t − η·λ·Σᵢ gᵢ (eq. 17).

  Local Update — Server sends ω_{t+1/2} to clients.
                 Clients reply with locally optimised models ωᵢ_{t+1}.
                 Server averages to form ω_{t+1}.
                 [Client side is currently a stub.]
"""

from __future__ import annotations

from logging import INFO
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import ks_2samp

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.common import logger

from kffl.data.dataset import get_federated_loaders
from kffl.ml import create_model, get_weights, set_weights

app = ServerApp()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broadcast(
    grid: Grid,
    node_ids: List[int],
    message_type: str,
    content: RecordDict,
    group_id: str,
) -> List[Message]:
    """Send *content* to every node and return the ordered list of replies."""
    messages = [
        Message(content, dst_node_id=nid, message_type=message_type, group_id=group_id)
        for nid in node_ids
    ]
    return list(grid.send_and_receive(messages))


def _weights_to_array_record(weights: List[np.ndarray]) -> ArrayRecord:
    """Pack model weights as ``w_0, w_1, …`` keys in an ArrayRecord."""
    return ArrayRecord({f"w_{i}": Array(w.astype(np.float32)) for i, w in enumerate(weights)})


def _model_config_record(
    t: int, seed: int, D: int, gamma_s: float, gamma_f: float,
    input_dim: int, num_classes: int, model_name: str,
) -> ConfigRecord:
    """Build the shared ConfigRecord included in FAIR1 and FAIR2 messages."""
    return ConfigRecord({
        "round_num": t,
        "seed": seed,
        "num_rf": D,
        "gamma_s": gamma_s,
        "gamma_f": gamma_f,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "model_name": model_name,
    })


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    model,
    dataset_name: str,
    num_partitions: int,
    batch_size: int,
    sensitive_features: Optional[List[str]],
) -> Dict[str, float]:
    """Compute RMSE and KS distance on the full (reconstructed) dataset.

    Loads all client partitions and concatenates them so the server has
    a global view of predictions vs labels and sensitive groups.

    Metrics
    -------
    rmse:
        Root mean squared error of predicted probabilities vs true labels.
        Lower is better; a random model gives ~0.5 on balanced data.

    ks:
        KS distance — the maximum over all pairs (s, s') of sensitive groups
        of  sup_t |F^s(t; ω) − F^s'(t; ω)|,  where F^s is the empirical CDF
        of model predictions for group s (eq. 5–6 of the paper).
        Zero means identical prediction distributions across groups.

    acc:
        Classification accuracy (threshold 0.5).
    """
    loaders = get_federated_loaders(
        dataset_name,
        num_partitions=num_partitions,
        batch_size=batch_size,
        sensitive_features=sensitive_features,
        shuffle=False,
    )

    all_preds: List[np.ndarray] = []
    all_y:     List[np.ndarray] = []
    all_s:     List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for loader in loaders:
            for X_batch, y_batch, s_batch in loader:
                preds = model.predict_proba(X_batch)        # (batch,) for binary
                if preds.ndim > 1:
                    preds = preds[:, 1]                     # P(y=1) for multiclass
                all_preds.append(preds.cpu().numpy())
                all_y.append(y_batch.numpy())
                all_s.append(s_batch.numpy())               # (batch, k)

    preds_np = np.concatenate(all_preds).astype(np.float64)  # (n,)
    y_np     = np.concatenate(all_y).astype(np.float64)      # (n,)
    s_np     = np.concatenate(all_s)                          # (n, k)

    # RMSE of predicted probabilities vs binary labels
    rmse = float(np.sqrt(np.mean((preds_np - y_np) ** 2)))

    # Accuracy
    acc = float(np.mean((preds_np >= 0.5) == y_np))

    # KS distance: max over all sensitive attribute columns and all group pairs
    ks_max = 0.0
    k = s_np.shape[1]
    for col in range(k):
        groups = np.unique(s_np[:, col])
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                p1 = preds_np[s_np[:, col] == g1]
                p2 = preds_np[s_np[:, col] == g2]
                if len(p1) > 0 and len(p2) > 0:
                    ks_stat, _ = ks_2samp(p1, p2)
                    ks_max = max(ks_max, ks_stat)

    return {"rmse": rmse, "ks": ks_max, "acc": acc}


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Drive KFFL for *num-server-rounds* iterations."""
    # --- Read run config ---
    num_rounds: int = int(context.run_config.get("num-server-rounds", 3))
    D: int = int(context.run_config.get("num-random-features", 16))
    gamma_s: float = float(context.run_config.get("gamma-s", 1.0))
    gamma_f: float = float(context.run_config.get("gamma-f", 1.0))
    step_size: float = float(context.run_config.get("step-size", 0.01))
    lambda_fair: float = float(context.run_config.get("lambda", 1.0))
    dataset_name: str = str(context.run_config.get("dataset", "adult"))
    model_name: str = str(context.run_config.get("model", "logistic"))
    num_classes: int = int(context.run_config.get("num-classes", 2))
    num_partitions: int = int(context.run_config.get("num-partitions", 5))
    batch_size: int = int(context.run_config.get("batch-size", 64))
    proximal_alpha: float = float(context.run_config.get("proximal-alpha", 1.0))
    num_local_epochs: int = int(context.run_config.get("num-local-epochs", 1))

    # Optional comma-separated sensitive feature names (e.g. "sex" or "sex,race")
    sens_raw = context.run_config.get("sensitive-features", None)
    sensitive_features = (
        [s.strip() for s in str(sens_raw).split(",") if s.strip()]
        if sens_raw is not None
        else None   # use dataset-specific default
    )

    # --- Derive input_dim from one sample batch ---
    sample_loader = get_federated_loaders(
        dataset_name, num_partitions=num_partitions, batch_size=1,
        sensitive_features=sensitive_features,
    )[0]
    X_sample, _, _ = next(iter(sample_loader))
    input_dim: int = X_sample.shape[1]

    # --- Initialise model ---
    model = create_model(model_name, input_dim=input_dim, num_classes=num_classes)
    n_params = len(list(model.parameters()))
    logger.log(
        INFO,
        "[Server] Model: %s  input_dim=%d  num_classes=%d  params=%d",
        model_name, input_dim, num_classes, n_params,
    )

    node_ids: List[int] = list(grid.get_node_ids())
    logger.log(INFO, "[Server] %d client(s) available.", len(node_ids))

    for t in range(num_rounds):
        group = str(t)
        seed = int(np.random.randint(0, 2**31))
        logger.log(INFO, "[Server] === Round %d / %d ===", t + 1, num_rounds)

        model_cfg = _model_config_record(
            t, seed, D, gamma_s, gamma_f, input_dim, num_classes, model_name
        )

        # ------------------------------------------------------------------ #
        # FAIR1: collect local interaction terms                              #
        # ------------------------------------------------------------------ #
        fair1_content = RecordDict({
            "config": model_cfg,
            "model": _weights_to_array_record(get_weights(model)),
        })

        fair1_replies = _broadcast(grid, node_ids, "query.fair1", fair1_content, group)
        logger.log(
            INFO, "[Server] FAIR1: received %d / %d replies.",
            len(fair1_replies), len(node_ids),
        )

        sum_Mi = np.zeros((D, D), dtype=np.float64)
        sum_ni_mu_s = np.zeros(D, dtype=np.float64)
        sum_ni_mu_f = np.zeros(D, dtype=np.float64)
        total_n = 0

        for reply in fair1_replies:
            if reply.has_error():
                logger.log(INFO, "[Server] FAIR1 reply error: %s", reply.error)
                continue
            arec: ArrayRecord = reply.content["fair1_data"]  # type: ignore[index]
            ni = int(arec["ni"].numpy().item())
            sum_Mi += arec["Mi"].numpy().astype(np.float64)
            sum_ni_mu_s += ni * arec["mu_s"].numpy().astype(np.float64)
            sum_ni_mu_f += ni * arec["mu_f"].numpy().astype(np.float64)
            total_n += ni

        mu_s_global = sum_ni_mu_s / max(total_n, 1)
        mu_f_global = sum_ni_mu_f / max(total_n, 1)
        G = sum_Mi - total_n * np.outer(mu_s_global, mu_f_global)

        logger.log(
            INFO,
            "[Server] FAIR1: total_n=%d  ||G||=%.6f  ||µ_s||=%.4f  ||µ_f||=%.4f",
            total_n, float(np.linalg.norm(G)),
            float(np.linalg.norm(mu_s_global)),
            float(np.linalg.norm(mu_f_global)),
        )

        # ------------------------------------------------------------------ #
        # FAIR2: collect local fairness gradients                            #
        # ------------------------------------------------------------------ #
        fair2_content = RecordDict({
            "config": model_cfg,
            "model": _weights_to_array_record(get_weights(model)),
            "fair2_data": ArrayRecord({
                "G": Array(G.astype(np.float32)),
                "mu_s": Array(mu_s_global.astype(np.float32)),
                "mu_f": Array(mu_f_global.astype(np.float32)),
            }),
        })

        fair2_replies = _broadcast(grid, node_ids, "query.fair2", fair2_content, group)
        logger.log(
            INFO, "[Server] FAIR2: received %d / %d replies.",
            len(fair2_replies), len(node_ids),
        )

        # Sum per-parameter gradients: sumᵢ gᵢ
        sum_grad = [np.zeros_like(w, dtype=np.float64) for w in get_weights(model)]
        n_valid = 0
        for reply in fair2_replies:
            if reply.has_error():
                logger.log(INFO, "[Server] FAIR2 reply error: %s", reply.error)
                continue
            arec = reply.content["gradient"]  # type: ignore[index]
            for i in range(n_params):
                sum_grad[i] += arec[f"g_{i}"].numpy().astype(np.float64)
            n_valid += 1

        grad_norm = float(np.sqrt(sum(np.sum(g ** 2) for g in sum_grad)))

        hsic_scale = 2.0 / max(total_n - 1, 1) ** 2
        omega_half_weights = [
            w - step_size * lambda_fair * hsic_scale * g
            for w, g in zip(get_weights(model), sum_grad)
        ]

        scaled_grad_norm = float(np.sqrt(sum(
            np.sum((hsic_scale * g) ** 2) for g in sum_grad
        )))
        half_norm = float(np.sqrt(sum(np.sum(w ** 2) for w in omega_half_weights)))
        logger.log(
            INFO,
            "[Server] FAIR2: n_valid=%d  ||∇HSIC||=%.6f  hsic_scale=%.2e  ||ω_{t+1/2}||=%.4f",
            n_valid, scaled_grad_norm, hsic_scale, half_norm,
        )

        # ------------------------------------------------------------------ #
        # Local Update: clients perform proximal gradient step       #
        # ------------------------------------------------------------------ #
        local_cfg = ConfigRecord({
            "round_num": t,
            "step_size": step_size,
            "proximal_alpha": proximal_alpha,
            "num_local_epochs": num_local_epochs,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "model_name": model_name,
        })
        local_update_content = RecordDict({
            "config": local_cfg,
            "model": _weights_to_array_record(omega_half_weights),
        })

        local_replies = _broadcast(
            grid, node_ids, "query.local_update", local_update_content, group
        )
        logger.log(
            INFO, "[Server] Local Update: received %d replies.", len(local_replies)
        )

        # Average returned weights → ω_{t+1}
        valid_weights: List[List[np.ndarray]] = []
        for reply in local_replies:
            if reply.has_error():
                logger.log(INFO, "[Server] Local Update reply error: %s", reply.error)
                continue
            arec = reply.content["model"]  # type: ignore[index]
            valid_weights.append([arec[f"w_{i}"].numpy() for i in range(n_params)])

        if valid_weights:
            avg_weights = [
                np.mean([vw[i] for vw in valid_weights], axis=0)
                for i in range(n_params)
            ]
            set_weights(model, avg_weights)

        new_norm = float(np.sqrt(sum(np.sum(w ** 2) for w in get_weights(model))))

        metrics = _compute_metrics(
            model, dataset_name, num_partitions, batch_size, sensitive_features
        )
        logger.log(
            INFO,
            "[Server] Round %d complete.  ||ω_{t+1}||=%.4f"
            "  acc=%.4f  rmse=%.4f  ks=%.4f",
            t + 1, new_norm,
            metrics["acc"], metrics["rmse"], metrics["ks"],
        )

    logger.log(INFO, "[Server] KFFL finished after %d round(s).", num_rounds)
