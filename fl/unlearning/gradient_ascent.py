"""Gradient-ascent unlearning.

Idea: keep training the model, but on the revoked DS's data only and
with a **negative** learning rate. Each step moves the parameters in the
direction that *increases* loss on the revoked data — i.e. away from
remembering it. Simple, fast, and produces measurable MIA drops in
practice (Liu et al., 2020 use this as the FedEraser warm-up step;
Thudi et al., 2022 give a tighter analysis).

Hyperparameters (``unlearning.gradient_ascent`` block of the YAML):

  steps                 number of negative-gradient steps (default 10)
  learning_rate         step size (default 1e-4)
  loss_cap              optional cap on the per-step loss to prevent the
                        model from diverging into NaN territory; if the
                        per-batch loss exceeds this value, the step is
                        skipped (default: no cap)
  weight_decay          L2 regulariser applied during ascent (default 0)
"""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
from torch import nn

from fl.unlearning.base import BaseUnlearner, UnlearningRequest, UnlearningResult


class GradientAscentUnlearner(BaseUnlearner):
    name = "GRADIENT_ASCENT"

    def unlearn(self, request: UnlearningRequest) -> UnlearningResult:
        cfg = request.method_config
        steps = int(cfg.get("steps", 10))
        lr = float(cfg.get("learning_rate", 1e-4))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        loss_cap = cfg.get("loss_cap")  # None or float

        t0 = time.time()
        # Deep-copy so the original pre-unlearn model is untouched (its hash
        # must remain stable for the on-chain modelHashPre check).
        model = copy.deepcopy(request.pre_unlearn_model).to(request.device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        # Ascend by descending on the negated loss. (PyTorch's Adam rejects
        # negative learning rates, so we flip the sign of the loss instead.)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        step_losses: list[float] = []
        steps_taken = 0
        for step in range(steps):
            for X, y in request.revoked_train_loader:
                X = X.to(request.device, non_blocking=True)
                y = y.to(request.device, non_blocking=True)
                opt.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                if loss_cap is not None and loss.item() > float(loss_cap):
                    continue
                (-loss).backward()
                opt.step()
                step_losses.append(loss.item())
                steps_taken += 1
                # One inner mini-batch counts as one ascent step (keeps the
                # method snappy on small clients).
                break
            else:
                # No batches in revoked loader; abort.
                break

        details: dict[str, Any] = {
            "steps_requested": steps,
            "steps_taken": steps_taken,
            "lr": lr,
            "mean_step_loss": (sum(step_losses) / len(step_losses)) if step_losses else 0.0,
            "final_step_loss": step_losses[-1] if step_losses else 0.0,
        }
        return self._wrap(request, model, t0, details)
