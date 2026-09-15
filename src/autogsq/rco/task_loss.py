"""Task-loss RCO refinement via forward passes on a soft-mixed model.

Reference RCO does not allocate from static proxies: each step builds the
probability-weighted ("soft-mixed") model ``w_i = sum_k p_ik(alpha) * w_ik``,
runs a forward pass on calibration data, and backpropagates task
cross-entropy all the way to the allocation logits. Sensitivity, cross-layer
interaction, and downstream amplification are captured automatically.

This module implements exactly that, adapted to our candidate database:

* Mixed weights are built per call from the on-disk dequant candidates and
  swapped in with ``torch.func``-style stateless calls (the base model is
  never mutated), so gradients reach the mixed weights.
* ``dL/dalpha`` uses the exact softmax chain rule from a *single* backward
  pass per batch: ``dL/dalpha_ik = p_ik * (<g_i, w_ik> - <g_i, wbar_i>)``.
* Candidates are preloaded when they fit ``max_cache_bytes`` and streamed
  layer-by-layer otherwise (two passes per step: mix, then dot products).
* Minibatching: each call consumes the next ``batches_per_step`` calibration
  batches, cycling across steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskLossRefiner:
    """Callable ``(alpha) -> (loss, grad)`` closing over model + candidates."""

    def __init__(
        self,
        model: nn.Module,
        store: Any,
        group_names: List[str],
        option_labels: List[str],
        calib_batches: List[torch.Tensor],
        device: torch.device,
        batches_per_step: int = 2,
        model_dtype: Optional[torch.dtype] = None,
        max_cache_bytes: float = 4e9,
    ) -> None:
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.store = store
        self.group_names = list(group_names)
        self.option_labels = list(option_labels)
        self.batches = list(calib_batches)
        if not self.batches:
            raise ValueError("TaskLossRefiner needs at least one calibration batch")
        self.device = device
        self.batches_per_step = max(int(batches_per_step), 1)
        self.step_count = 0

        probe = model.get_submodule(self._mod_path(group_names[0])).weight
        self.model_dtype = model_dtype or probe.dtype

        # Preload-or-stream decision from on-disk footprint.
        total = 0
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        for name in self.group_names:
            for opt in self.option_labels:
                dp, _ = store.get_candidate_paths(name, opt)
                try:
                    total += dp.stat().st_size
                except OSError:
                    total += 4 * 1024 * 1024
        self.preload = total <= max_cache_bytes
        self._cache: Dict[Tuple[str, str], torch.Tensor] = {}
        if self.preload:
            for name in self.group_names:
                for opt in self.option_labels:
                    w = store.load_dequant_weight(name, opt, device=torch.device("cpu"))
                    self._cache[(name, opt)] = w.to(dtype=torch.float32)

    @staticmethod
    def _mod_path(weight_name: str) -> str:
        return weight_name[:-len(".weight")] if weight_name.endswith(".weight") else weight_name

    def _candidate(self, name: str, opt: str) -> torch.Tensor:
        """One dequant candidate in fp32 on CPU (cached or freshly loaded)."""
        key = (name, opt)
        if key in self._cache:
            return self._cache[key]
        w = self.store.load_dequant_weight(name, opt, device=torch.device("cpu"))
        return w.to(dtype=torch.float32)

    def loss_and_grad(self, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expected task CE loss and its exact gradient w.r.t. ``alpha``."""
        from autogsq.rco.loss import cross_entropy_eval

        dev = self.device
        alpha_f = alpha.detach().to(device=dev, dtype=torch.float32)
        with torch.no_grad():
            p = F.softmax(alpha_f, dim=-1)  # (N, K)
        N, K = p.shape

        # ---- Pass 1: build the soft-mixed weight dict (streamed per tensor).
        mixed: Dict[str, torch.Tensor] = {}
        mix_cache: Dict[str, List[torch.Tensor]] = {} if not self.preload else {}
        for i, name in enumerate(self.group_names):
            mod_path = self._mod_path(name)
            acc = None
            row = []
            for k, opt in enumerate(self.option_labels):
                w = self._candidate(name, opt).to(dev)
                if not self.preload:
                    row.append(w.cpu())
                else:
                    row = None  # type: ignore[assignment]
                wk = w.to(dtype=self.model_dtype)
                acc = wk * p[i, k] if acc is None else acc + wk * p[i, k]
            assert acc is not None
            mixed[mod_path + ".weight"] = acc.detach().requires_grad_(True)
            if not self.preload:
                mix_cache[name] = row  # type: ignore[assignment]

        # ---- Pass 2: minibatch forwards + backwards (model never mutated).
        n_batches = len(self.batches)
        start = (self.step_count * self.batches_per_step) % n_batches
        total_loss = 0.0
        grads: Dict[str, torch.Tensor] = {}
        for b in range(self.batches_per_step):
            batch = self.batches[(start + b) % n_batches].to(dev)
            for t in mixed.values():
                if t.grad is not None:
                    t.grad = None
            outputs = torch.func.functional_call(self.model, mixed, (batch,))
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss_b = cross_entropy_eval(logits.to(torch.float32), batch)
            loss_b.backward()
            total_loss += float(loss_b.detach().item())
            for mod_path, t in mixed.items():
                g = t.grad.detach().to(torch.float32).cpu()
                grads[mod_path] = g if mod_path not in grads else grads[mod_path] + g
        self.step_count += 1
        mean_loss = total_loss / self.batches_per_step

        # ---- Pass 3: exact softmax chain rule to alpha.
        dalpha = torch.zeros_like(alpha_f)
        p_cpu = p.detach().cpu()
        for i, name in enumerate(self.group_names):
            g = grads[name + ".weight"] if (name + ".weight") in grads else grads[name]
            dots = []
            mix_key = name + ".weight" if (name + ".weight") in mixed else name
            mix_w = mixed[mix_key].detach().to(torch.float32).cpu()
            wbar_dot = float((g * mix_w).sum().item())
            for k, opt in enumerate(self.option_labels):
                if self.preload:
                    w = self._cache[(name, opt)]
                else:
                    w = mix_cache[name][k]
                dots.append(float((g * w).sum().item()))
            for k in range(K):
                dalpha[i, k] = p_cpu[i, k] * (dots[k] - wbar_dot)

        loss_t = torch.tensor(mean_loss)
        return loss_t, dalpha.to(dtype=alpha.dtype, device=alpha.device)

    __call__ = loss_and_grad
