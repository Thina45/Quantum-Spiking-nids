"""Classical SNN ablation: QSNN-NIDS with the quantum circuit in each QLIF
block replaced by a classical current source (classical_current_qlif.py).

Mirrors qsnn/models/qsnn.py's QSNN class (constructor + forward) verbatim,
substituting ClassicalCurrentLIFLayer for QuantumLIFLayer. This file is a
pure addition -- qsnn/models/qsnn.py is not modified, so the existing
QSNN-NIDS model/checkpoints are unaffected and exactly reproducible.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..config import Config, EncodingConfig, ModelConfig
from ..data.spike_encoding import SpikeEncoder
from .classical_current_qlif import ClassicalCurrentLIFLayer


class ClassicalSNN(nn.Module):
    """QSNN-NIDS architecture with classical-current-source LIF layers."""

    def __init__(
        self,
        n_features: int,
        model_cfg: ModelConfig,
        encoding_cfg: EncodingConfig,
        bottleneck: int = 3,
    ):
        super().__init__()
        self.n_features = n_features
        self.cfg = model_cfg
        self.enc_cfg = encoding_cfg
        self.encoder = SpikeEncoder(encoding_cfg)
        self.T = encoding_cfg.timesteps
        Q = model_cfg.n_qubits

        dims = [n_features] + [Q] * model_cfg.n_qlif_layers
        self.synapses = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(model_cfg.n_qlif_layers)]
        )
        for lin in self.synapses:
            nn.init.xavier_uniform_(lin.weight, gain=0.8)
            nn.init.zeros_(lin.bias)

        self.qlif_layers = nn.ModuleList([
            ClassicalCurrentLIFLayer(
                n_qubits=Q,
                bottleneck=bottleneck,
                beta=model_cfg.beta,
                learn_beta=model_cfg.learn_beta,
                threshold=model_cfg.threshold,
                learn_threshold=model_cfg.learn_threshold,
                reset=model_cfg.reset,
                angle_map=model_cfg.angle_map,
                angle_gain=model_cfg.angle_gain,
                angle_bias=model_cfg.angle_bias,
                surrogate=model_cfg.surrogate,
                surrogate_alpha=model_cfg.surrogate_alpha,
                seed=1234 + idx,
            )
            for idx in range(model_cfg.n_qlif_layers)
        ])

        readout_dim = {
            "rate": Q, "mem": Q, "prob": Q,
            "rate_mem": 2 * Q, "rate_prob": 2 * Q, "rate_mem_prob": 3 * Q,
        }[model_cfg.readout]
        self.norm = nn.BatchNorm1d(readout_dim)
        self.drop = nn.Dropout(model_cfg.dropout)
        self.head = nn.Linear(readout_dim, model_cfg.n_classes)
        nn.init.zeros_(self.head.bias)

        self._spike_rates: List[float] = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x if x.dim() == 3 else self.encoder(x)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        spikes_in = self.encode(x).to(next(self.parameters()).dtype)

        h = spikes_in
        aux: Dict[str, torch.Tensor] = {}
        self._spike_rates = []
        states: Dict[str, torch.Tensor] = {}

        for i, (syn, layer) in enumerate(zip(self.synapses, self.qlif_layers)):
            cur = syn(h)
            h, states = layer(cur, return_states=True)
            self._spike_rates.append(float(h.detach().mean()))
            if return_aux:
                aux[f"spikes_l{i}"] = h
                aux[f"mem_l{i}"] = states["mem"]

        rate = h.mean(dim=1)
        mem = torch.tanh(states["final_mem"])
        prob = states["p_excited"].mean(dim=1)
        parts = {"rate": rate, "mem": mem, "prob": prob}
        feats = torch.cat([parts[k] for k in self.cfg.readout.split("_")], dim=-1)

        logits = self.head(self.drop(self.norm(feats)))
        if return_aux:
            aux["input_spikes"] = spikes_in
            aux["readout"] = feats
            return logits, aux
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self(x), dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).argmax(dim=-1)

    @property
    def spike_rate(self) -> float:
        return float(sum(self._spike_rates) / max(len(self._spike_rates), 1))

    def param_counts(self) -> Dict[str, int]:
        current_source = sum(
            p.numel() for n, p in self.named_parameters() if "qlif_layers" in n and ("fc1" in n or "fc2" in n)
        )
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "current_source_params": int(current_source),
            "other_classical_params": int(total - current_source),
            "total_params": int(total),
        }


def build_classical_snn_model(cfg: Config, n_features: int, bottleneck: int = 3) -> ClassicalSNN:
    return ClassicalSNN(
        n_features=n_features,
        model_cfg=cfg.model,
        encoding_cfg=cfg.encoding,
        bottleneck=bottleneck,
    )
