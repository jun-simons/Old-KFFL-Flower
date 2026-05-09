"""Logistic Regression model for KFFL.

Architecture:
  A single linear layer mapping R^d → R^1 (binary) or R^C (multiclass).
  Binary case uses BCEWithLogitsLoss; multiclass uses CrossEntropyLoss.

Training:
  ``train()`` runs Adam for a configurable number of local epochs,
  optionally adding a proximal penalty  (μ/2)·‖ω − ω_center‖² that pulls the
  model toward the half-step ω_{t+1/2} received from the server (eq. 21).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import TrainResult, get_weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LogisticRegression(nn.Module):
    """Single-layer linear classifier.

    For binary classification (``num_classes=2``) the output is a single
    logit; ``predict_proba`` applies sigmoid to produce a probability in
    [0, 1].  For multiclass (``num_classes>2``) the output is a vector of
    ``num_classes`` logits; ``predict_proba`` applies softmax.

    Parameters
    ----------
    input_dim:
        Number of input features (d).
    num_classes:
        Number of target classes.  Defaults to 2 (binary).
    """

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self._binary = num_classes == 2

        out_features = 1 if self._binary else num_classes
        self.linear = nn.Linear(input_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits, shape ``(n,)`` for binary or ``(n, C)`` for multiclass."""
        out = self.linear(x)
        return out.squeeze(1) if self._binary else out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities (no gradient)

        Returns
        -------
        torch.Tensor
            Shape ``(n,)`` in [0, 1] for binary, or ``(n, C)`` for multiclass.
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits) if self._binary else torch.softmax(logits, dim=1)

    def proba_for_orf(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities shaped for ORF input, **keeping gradients**.

        Unlike ``predict_proba``, this method does not disable gradient
        computation so the result can be used as an intermediate in an
        autograd graph

        Returns
        -------
        torch.Tensor
            Shape ``(n, 1)`` for binary (scalar output per sample, ready for a
            1-D ORF), or ``(n, C)`` for multiclass.
        """
        logits = self.forward(x)
        if self._binary:
            return torch.sigmoid(logits).unsqueeze(1)   # (n, 1)
        return torch.softmax(logits, dim=1)              # (n, C)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices, shape ``(n,)``."""
        with torch.no_grad():
            if self._binary:
                return (torch.sigmoid(self.forward(x)) >= 0.5).long()
            return self.forward(x).argmax(dim=1)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train(
    model: LogisticRegression,
    loader: DataLoader,
    *,
    lr: float = 0.01,
    num_epochs: int = 1,
    weight_decay: float = 0.0,
    proximal_mu: float = 0.0,
    proximal_center: Optional[List[np.ndarray]] = None,
    device: str = "cpu",
) -> TrainResult:
    """Train *model* on the data provided by *loader* for one federated round.

    The optional proximal term implements the FedProx / KFFL local-update
    objective (eq. 21):

        L_local(ω) = f_i(ω) + (μ/2)·‖ω − ω_center‖²

    Parameters
    ----------
    model:
        The ``LogisticRegression`` to train (modified in-place).
    loader:
        DataLoader yielding ``(X, y, s)`` batches.  The sensitive tensor
        ``s`` is present but not used during standard local training.
    lr:
        Learning rate for the optimizer.
    num_epochs:
        Number of passes over the local dataset.
    weight_decay:
        L2 regularization coefficient passed to the optimizer.
    proximal_mu:
        Strength of the proximal penalty.  Set to 0 to disable (standard
        ERM training).
    proximal_center:
        Reference parameters (list of numpy arrays, same order as
        ``model.parameters()``) for the proximal term.  Required when
        ``proximal_mu > 0``; ignored otherwise.
    device:
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    TrainResult
        Contains per-epoch mean loss, total examples seen, and epoch count.

    Raises
    ------
    ValueError
        If ``proximal_mu > 0`` but ``proximal_center`` is not provided.
    """
    if proximal_mu > 0.0 and proximal_center is None:
        raise ValueError(
            "proximal_center must be provided when proximal_mu > 0."
        )

    dev = torch.device(device)
    model = model.to(dev)
    model.train()

    criterion: nn.Module = (
        nn.BCEWithLogitsLoss() if model._binary else nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Pre-convert proximal center to tensors on the target device
    center_tensors: Optional[List[torch.Tensor]] = None
    if proximal_mu > 0.0 and proximal_center is not None:
        center_tensors = [
            torch.as_tensor(w, dtype=p.dtype, device=dev)
            for p, w in zip(model.parameters(), proximal_center)
        ]

    loss_history: List[float] = []
    num_examples = 0

    for _ in range(num_epochs):
        epoch_loss = 0.0
        epoch_n = 0

        for batch in loader:
            X_batch, y_batch, _ = batch  # s unused during local training
            X_batch = X_batch.to(dev)
            y_batch = y_batch.to(dev)

            optimizer.zero_grad()
            logits = model(X_batch)

            # Task loss
            target = y_batch.float() if model._binary else y_batch
            loss = criterion(logits, target)

            # Proximal penalty: (mu/2) * sum_i ||param_i - center_i||^2
            if center_tensors is not None:
                prox = torch.tensor(0.0, device=dev)
                for param, center in zip(model.parameters(), center_tensors):
                    prox = prox + torch.sum((param - center) ** 2)
                loss = loss + (proximal_mu / 2.0) * prox

            loss.backward()
            optimizer.step()

            batch_n = X_batch.shape[0]
            epoch_loss += loss.item() * batch_n
            epoch_n += batch_n

        loss_history.append(epoch_loss / max(epoch_n, 1))
        num_examples = epoch_n  # same each epoch; record once

    return TrainResult(
        loss_history=loss_history,
        num_examples=num_examples,
        num_epochs=num_epochs,
    )
