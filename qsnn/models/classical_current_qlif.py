"""Classical-current-source LIF layer -- the Scientific Reports revision ablation.

Structurally identical to QuantumLIFLayer (qsnn/models/qlif.py) in every
respect EXCEPT step 3 ("couple"): the quantum circuit (RY-prepare + trainable
entangling ansatz + measurement, qlif.py's self.circuit/self.qweights) is
replaced by a classical bottleneck network of matched parameter count:

    theta (n_qubits angles, same angle map phi(g*z+b), g=2, b=-2)
      -> Linear(n_qubits, bottleneck, bias=False)
      -> Linear(bottleneck, n_qubits, bias=False)
      -> sigmoid
      -> current I in [0,1]   (plays the same role as P(|1>) in QuantumLIFLayer)

Leak, threshold, reset, surrogate gradient, angle map and every other piece of
the membrane dynamics are copied verbatim from QuantumLIFLayer so that the
only experimental variable is quantum-circuit-vs-classical-current-source.

This file is a pure addition -- qsnn/models/qlif.py is not modified, so the
existing QSNN-NIDS results and checkpoints remain exactly reproducible from
the unmodified code.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .surrogate import spike_fn


class ClassicalCurrentLIFLayer(nn.Module):
    """A population of ``n_qubits`` LIF neurons whose coupling step is a
    classical Linear-Linear-sigmoid bottleneck instead of a quantum circuit."""

    def __init__(
        self,
        n_qubits: int,
        bottleneck: int = 3,
        beta: float = 0.9,
        learn_beta: bool = True,
        threshold: float = 0.55,
        learn_threshold: bool = False,
        reset: str = "subtract",
        angle_map: str = "sigmoid",
        angle_gain: float = 2.0,
        angle_bias: float = -2.0,
        surrogate: str = "fast_sigmoid",
        surrogate_alpha: float = 5.0,
        seed: int = 42,
    ):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.reset = reset
        if angle_map not in ("sigmoid", "tanh"):
            raise ValueError(f"unknown angle_map '{angle_map}'")
        self.angle_map = angle_map
        self.angle_gain = float(angle_gain)
        self.angle_bias = float(angle_bias)
        self.surrogate = surrogate
        self.surrogate_alpha = surrogate_alpha

        b0 = math.log(beta / (1.0 - beta))
        t0 = math.log(threshold / (1.0 - threshold))
        self.beta_logit = nn.Parameter(torch.tensor(b0), requires_grad=learn_beta)
        self.thr_logit = nn.Parameter(torch.tensor(t0), requires_grad=learn_threshold)

        # ---- classical current source (replaces the quantum circuit) -------
        # 48 params/layer at n_qubits=8, bottleneck=3 (8*3 + 3*8), matching
        # QuantumLIFLayer's StronglyEntanglingLayers (ansatz_layers=2,
        # n_qubits=8, 3) = 48 params/layer exactly.
        g = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(self.n_qubits, bottleneck, bias=False)
        self.fc2 = nn.Linear(bottleneck, self.n_qubits, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(0.1 * torch.randn(self.fc1.weight.shape, generator=g))
            self.fc2.weight.copy_(0.1 * torch.randn(self.fc2.weight.shape, generator=g))

        self.register_buffer("_last_spike_rate", torch.zeros(1), persistent=False)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_logit)

    @property
    def v_th(self) -> torch.Tensor:
        return torch.sigmoid(self.thr_logit)

    def membrane_to_angle(self, mem: torch.Tensor) -> torch.Tensor:
        """Identical to QuantumLIFLayer.membrane_to_angle (qlif.py)."""
        if self.angle_map == "tanh":
            return math.pi * torch.tanh(mem)
        return math.pi * torch.sigmoid(self.angle_gain * mem + self.angle_bias)

    def _current(self, theta: torch.Tensor) -> torch.Tensor:
        """theta [B, n_qubits] -> classical current I in [0,1], [B, n_qubits]."""
        return torch.sigmoid(self.fc2(self.fc1(theta)))

    def forward(
        self, current: torch.Tensor, return_states: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``current``: [B, T, n_qubits] synaptic input -> spikes [B, T, n_qubits].
        Identical control flow to QuantumLIFLayer.forward (qlif.py)."""
        if current.dim() != 3:
            raise ValueError(f"expected [B, T, N], got {tuple(current.shape)}")
        B, T, N = current.shape
        if N != self.n_qubits:
            raise ValueError(f"layer has {self.n_qubits} qubits, got {N} channels")

        beta, v_th = self.beta, self.v_th
        mem = torch.zeros(B, N, dtype=current.dtype, device=current.device)
        spk = torch.zeros_like(mem)

        spikes, mems, probs = [], [], []

        for t in range(T):
            if self.reset == "zero":
                mem = beta * mem * (1.0 - spk) + current[:, t]
            elif self.reset == "subtract":
                mem = beta * mem + current[:, t] - spk * v_th
            else:
                mem = beta * mem + current[:, t]

            theta = self.membrane_to_angle(mem)
            p1 = self._current(theta)

            spk = spike_fn(p1 - v_th, self.surrogate, self.surrogate_alpha)

            spikes.append(spk)
            if return_states:
                mems.append(mem)
                probs.append(p1)

        out = torch.stack(spikes, dim=1)
        self._last_spike_rate = out.detach().mean().reshape(1)
        if not return_states:
            return out
        return out, {
            "mem": torch.stack(mems, dim=1),
            "p_excited": torch.stack(probs, dim=1),
            "final_mem": mem,
        }
