from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch
from torch import nn



class ChebyshevHeatKernelBank(nn.Module):
    """Partition a graph signal into heat-kernel bands with one shared basis."""

    def __init__(
        self,
        network,
        heat_scales: tuple[float, ...],
        order: int,
        quadrature_points: int = 256,
    ) -> None:
        super().__init__()
        self.heat_scales = tuple(float(scale) for scale in heat_scales)
        self.order = int(order)
        self.band_count = len(self.heat_scales) + 1
        scaled_laplacian = self._scaled_normalized_laplacian(network)
        self.register_buffer("scaled_laplacian", scaled_laplacian)
        coefficients = self._chebyshev_coefficients(quadrature_points)
        self.register_buffer("coefficients", coefficients)
        self.sparse_multiply_count = 0

    def update_topology(
        self,
        network,
        feasible_destinations: Mapping[int, Sequence[int]] | None,
        edge_weights: Mapping[tuple[int, int], float] | None = None,
    ) -> None:
        """Replace only the observed Laplacian; spectral coefficients stay fixed."""

        scaled = self._scaled_normalized_laplacian(
            network,
            feasible_destinations=feasible_destinations,
            edge_weights=edge_weights,
        ).to(self.scaled_laplacian.device)
        self.scaled_laplacian = scaled

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("Graph signals must have shape [batch, nodes, features]")
        basis_0 = z
        output = torch.einsum("k,bnf->bnkf", self.coefficients[:, 0], basis_0)
        if self.order == 0:
            return output
        basis_1 = self._laplacian_multiply(basis_0)
        output = output + torch.einsum("k,bnf->bnkf", self.coefficients[:, 1], basis_1)
        previous, current = basis_0, basis_1
        for degree in range(2, self.order + 1):
            following = 2.0 * self._laplacian_multiply(current) - previous
            output = output + torch.einsum(
                "k,bnf->bnkf",
                self.coefficients[:, degree],
                following,
            )
            previous, current = current, following
        return output

    def exact_responses(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Return the nonnegative heat-band responses on eigenvalues in [0, 2]."""
        values = eigenvalues.to(dtype=torch.float64)
        responses = []
        previous = torch.exp(-self.heat_scales[0] * values)
        responses.append(previous)
        for scale in self.heat_scales[1:]:
            current = torch.exp(-scale * values)
            responses.append(current - previous)
            previous = current
        responses.append(1.0 - previous)
        return torch.stack(responses, dim=-1).to(dtype=eigenvalues.dtype)

    def _laplacian_multiply(self, values: torch.Tensor) -> torch.Tensor:
        batch, nodes, features = values.shape
        flattened = values.permute(1, 0, 2).reshape(nodes, batch * features)
        multiplied = torch.sparse.mm(self.scaled_laplacian, flattened)
        self.sparse_multiply_count += 1
        return multiplied.reshape(nodes, batch, features).permute(1, 0, 2)

    def _chebyshev_coefficients(self, quadrature_points: int) -> torch.Tensor:
        points = max(64, int(quadrature_points))
        indices = torch.arange(points, dtype=torch.float64)
        theta = (indices + 0.5) * math.pi / float(points)
        eigenvalues = 1.0 + torch.cos(theta)
        responses = self.exact_responses(eigenvalues).to(dtype=torch.float64)
        coefficients = []
        for degree in range(self.order + 1):
            factor = 1.0 if degree == 0 else 2.0
            coefficient = (
                factor
                / float(points)
                * torch.sum(responses * torch.cos(degree * theta)[:, None], dim=0)
            )
            coefficients.append(coefficient)
        return torch.stack(coefficients, dim=1).to(dtype=torch.float32)

    @staticmethod
    def _scaled_normalized_laplacian(
        network,
        feasible_destinations: Mapping[int, Sequence[int]] | None = None,
        edge_weights: Mapping[tuple[int, int], float] | None = None,
    ) -> torch.Tensor:
        node_count = network.zone_count
        adjacency = np.zeros((node_count, node_count), dtype=np.float64)
        for node in range(node_count):
            targets = (
                feasible_destinations[node]
                if feasible_destinations is not None
                else {
                    int(network.move(node, action))
                    for action in network.valid_actions(node)
                }
            )
            for target in targets:
                if target != node:
                    weight = (
                        1.0
                        if edge_weights is None
                        else float(edge_weights.get((node, int(target)), 1.0))
                    )
                    if not math.isfinite(weight) or weight <= 0.0:
                        raise ValueError("graph edge weights must be finite and positive")
                    adjacency[node, target] = max(adjacency[node, target], weight)
                    adjacency[target, node] = max(adjacency[target, node], weight)
        degree = adjacency.sum(axis=1)
        inv_sqrt = np.zeros_like(degree)
        nonzero = degree > 0
        inv_sqrt[nonzero] = degree[nonzero] ** -0.5
        normalized_adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
        # With lambda_max=2, the scaled Laplacian is L-I=-D^{-1/2}AD^{-1/2}.
        scaled = -normalized_adjacency
        rows, cols = np.nonzero(scaled)
        indices = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long)
        values = torch.as_tensor(scaled[rows, cols], dtype=torch.float32)
        with torch.sparse.check_sparse_tensor_invariants():
            return torch.sparse_coo_tensor(
                indices,
                values,
                (node_count, node_count),
                is_coalesced=True,
            )


class NodewiseDispatchGate(nn.Module):
    """Fuse graph-frequency bands according to pre-decision fleet pressure."""

    def __init__(self, hidden_dim: int, band_count: int, context_dim: int = 0) -> None:
        super().__init__()
        if context_dim < 0:
            raise ValueError("context_dim must be nonnegative")
        self.context_dim = int(context_dim)
        gate_hidden = max(8, hidden_dim // 2)
        self.gates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim + 1 + self.context_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
            )
            for _ in range(band_count)
        )

    def forward(
        self,
        scale_features: torch.Tensor,
        dispatch_pressure: torch.Tensor,
        gate_context: torch.Tensor | None = None,
        scale_mask: torch.Tensor | None = None,
        *,
        return_logits: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if dispatch_pressure.ndim == 2:
            dispatch_pressure = dispatch_pressure.unsqueeze(-1)
        if dispatch_pressure.shape[:2] != scale_features.shape[:2]:
            raise ValueError("Dispatch pressure must match the batch and node dimensions")
        if self.context_dim:
            if gate_context is None:
                raise ValueError("gate_context is required when context_dim is positive")
            if gate_context.shape != (*scale_features.shape[:2], self.context_dim):
                raise ValueError("gate_context must match batch, node, and context dimensions")
        else:
            gate_context = scale_features.new_zeros((*scale_features.shape[:2], 0))
        logits = torch.cat(
            [
                gate(
                    torch.cat(
                        [
                            scale_features[:, :, band, :],
                            dispatch_pressure,
                            gate_context,
                        ],
                        dim=-1,
                    )
                )
                for band, gate in enumerate(self.gates)
            ],
            dim=-1,
        )
        if scale_mask is not None:
            mask = torch.as_tensor(scale_mask, dtype=torch.bool, device=logits.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != (scale_features.shape[0], scale_features.shape[2]):
                raise ValueError("scale_mask must have shape [batch, bands]")
            if torch.any(mask.sum(dim=-1) == 0):
                raise ValueError("scale_mask must retain at least one band per sample")
            logits = logits.masked_fill(
                ~mask[:, None, :],
                torch.finfo(logits.dtype).min,
            )
        weights = torch.softmax(logits, dim=-1)
        fused = torch.sum(scale_features * weights.unsqueeze(-1), dim=2)
        if return_logits:
            return fused, weights, logits
        return fused, weights


class GraphWaveletNet(nn.Module):
    """Demand network with Chebyshev graph-frequency bands and dispatch gating."""

    def __init__(
        self,
        network,
        input_dim: int,
        hidden_dim: int,
        heat_scales: tuple[float, ...],
        chebyshev_order: int,
        *,
        gate_context_dim: int = 0,
        external_context_dim: int = 0,
        scale_dropout: float = 0.0,
        filter_mode: str = "graph_wavelet",
        prediction_head: bool = True,
    ) -> None:
        super().__init__()
        if gate_context_dim < 0 or external_context_dim < 0:
            raise ValueError("gate context dimensions must be nonnegative")
        if not 0.0 <= float(scale_dropout) < 1.0:
            raise ValueError("scale_dropout must be in [0, 1)")
        self.scale_dropout = float(scale_dropout)
        self.external_context_dim = int(external_context_dim)
        self.filter_mode = str(filter_mode).strip().lower().replace("-", "_")
        if self.filter_mode not in {"graph_wavelet", "gcn"}:
            raise ValueError("filter_mode must be graph_wavelet or gcn")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.filter_bank = ChebyshevHeatKernelBank(
            network,
            heat_scales,
            chebyshev_order,
        )
        self.scale_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(self.filter_bank.band_count)
        )
        self.gate = NodewiseDispatchGate(
            hidden_dim,
            self.filter_bank.band_count,
            context_dim=gate_context_dim + self.external_context_dim,
        )
        self.zone_embedding = (
            nn.Embedding(network.zone_count, gate_context_dim)
            if gate_context_dim > 0
            else None
        )
        self.residual = nn.Linear(hidden_dim, hidden_dim)
        self.out = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus(),
            )
            if prediction_head
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        dispatch_pressure: torch.Tensor | None = None,
        external_context: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor | None,
        dict[str, torch.Tensor | None],
    ]:
        # Equation (3) uses a linear latent projection before graph filtering;
        # the scale-specific encoders below apply the first nonlinearity.
        h = self.input_proj(x)
        if self.filter_mode == "graph_wavelet":
            raw_scales = self.filter_bank(h)
        else:
            one_hop = -self.filter_bank._laplacian_multiply(h)
            raw_scales = one_hop.unsqueeze(2).expand(
                -1,
                -1,
                self.filter_bank.band_count,
                -1,
            )
        scale_features = torch.stack(
            [
                encoder(raw_scales[:, :, band, :])
                for band, encoder in enumerate(self.scale_encoders)
            ],
            dim=2,
        )
        if dispatch_pressure is None:
            dispatch_pressure = torch.zeros(
                (*x.shape[:2], 1),
                dtype=x.dtype,
                device=x.device,
            )
        context_parts = []
        zone_context = None
        if self.zone_embedding is not None:
            zone_ids = torch.arange(x.shape[1], device=x.device)
            zone_context = self.zone_embedding(zone_ids).unsqueeze(0).expand(
                x.shape[0], -1, -1
            )
            context_parts.append(zone_context)
        if self.external_context_dim:
            if external_context is None:
                external_context = x.new_zeros(
                    (*x.shape[:2], self.external_context_dim)
                )
            if external_context.shape != (
                *x.shape[:2],
                self.external_context_dim,
            ):
                raise ValueError(
                    "external_context must match batch, nodes, and configured width"
                )
            context_parts.append(external_context)
        gate_context = (
            torch.cat(context_parts, dim=-1)
            if context_parts
            else None
        )
        scale_mask = torch.ones(
            (x.shape[0], self.filter_bank.band_count),
            dtype=torch.bool,
            device=x.device,
        )
        if self.training and self.scale_dropout > 0.0:
            scale_mask = (
                torch.rand(
                    (x.shape[0], self.filter_bank.band_count),
                    device=x.device,
                )
                >= self.scale_dropout
            )
            empty_rows = torch.nonzero(
                scale_mask.sum(dim=-1) == 0,
                as_tuple=False,
            ).flatten()
            if empty_rows.numel():
                scale_mask[empty_rows, 0] = True
        fused, gate_weights, gate_logits = self.gate(
            scale_features,
            dispatch_pressure,
            gate_context,
            scale_mask,
            return_logits=True,
        )
        residual = self.residual(h)
        representation = residual + fused
        prediction = (
            None
            if self.out is None
            else self.out(representation).squeeze(-1)
        )
        if not return_aux:
            if prediction is None:
                raise RuntimeError(
                    "GraphWaveletNet has no standalone prediction head"
                )
            return prediction
        return prediction, {
            "scale_features": scale_features,
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "scale_mask": scale_mask,
            "input_projection": h,
            "residual": residual,
            "fused_scales": fused,
            "dispatch_representation": representation,
            "zone_embedding": zone_context,
            "gate_context": gate_context,
        }

    def update_topology(
        self,
        network,
        feasible_destinations: Mapping[int, Sequence[int]] | None,
        edge_weights: Mapping[tuple[int, int], float] | None = None,
    ) -> None:
        self.filter_bank.update_topology(
            network,
            feasible_destinations,
            edge_weights,
        )



__all__ = ["ChebyshevHeatKernelBank", "GraphWaveletNet", "NodewiseDispatchGate"]
