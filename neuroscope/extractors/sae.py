"""
Sparse Autoencoder (SAE) feature extraction.

Decomposes dense activation vectors into sparse, interpretable features.
Each feature corresponds to a "voxel" in the NeuroScope brain scan --
a discrete, meaningful unit of model computation.

References:
  - Bricken et al., "Towards Monosemanticity" (Anthropic, 2023)
  - Templeton et al., "Scaling Monosemanticity" (Anthropic, 2024)
  - Gao et al., SAE Survey (arXiv:2503.05613, 2025)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class SAEFeature:
    """A single extracted feature from the SAE."""
    feature_idx: int
    activation_strength: float
    layer_idx: int
    description: Optional[str] = None  # from oracle narration
    category: Optional[str] = None


@dataclass
class SAEScanResult:
    """Result of running an SAE scan on model activations."""
    features: list[SAEFeature]
    sparsity: float  # fraction of features that are active
    reconstruction_loss: float
    layer_idx: int
    raw_latents: Optional[torch.Tensor] = None


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for extracting interpretable features from
    transformer activations.

    Architecture: x -> encoder -> ReLU -> latents -> decoder -> x_hat

    The encoder projects from d_model to d_sae (expansion factor typically 4-64x).
    Sparsity is enforced via L1 penalty on the latent activations.
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        l1_coefficient: float = 5e-3,
        tied_weights: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coefficient = l1_coefficient

        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        if tied_weights:
            self.decoder.weight = nn.Parameter(self.encoder.weight.T.clone())

        # Initialize with Kaiming
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse latent space."""
        return torch.relu(self.encoder(x))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct activations from sparse latents."""
        return self.decoder(latents)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            (reconstruction, latents, loss)
        """
        latents = self.encode(x)
        reconstruction = self.decode(latents)

        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - reconstruction) ** 2)

        # Sparsity loss (L1 on latents)
        l1_loss = self.l1_coefficient * torch.mean(torch.abs(latents))

        total_loss = recon_loss + l1_loss

        return reconstruction, latents, total_loss

    def extract_features(
        self,
        activations: torch.Tensor,
        layer_idx: int = -1,
        top_k: Optional[int] = None,
        threshold: float = 0.0,
    ) -> SAEScanResult:
        """
        Extract interpretable features from activations.

        Args:
            activations: Tensor of shape (batch, seq_len, d_model) or (seq_len, d_model).
            layer_idx: Which layer these activations came from.
            top_k: If set, only return top-k features by activation strength.
            threshold: Minimum activation strength to include.

        Returns:
            SAEScanResult with extracted features.
        """
        self.eval()
        with torch.no_grad():
            latents = self.encode(activations)
            reconstruction = self.decode(latents)
            recon_loss = torch.mean((activations - reconstruction) ** 2).item()

        # Aggregate across batch and sequence dimensions
        if latents.dim() == 3:
            feature_activations = latents.mean(dim=(0, 1))  # (d_sae,)
        elif latents.dim() == 2:
            feature_activations = latents.mean(dim=0)  # (d_sae,)
        else:
            feature_activations = latents

        # Find active features
        active_mask = feature_activations > threshold
        active_indices = torch.where(active_mask)[0]
        active_values = feature_activations[active_mask]

        # Sort by activation strength
        sorted_order = torch.argsort(active_values, descending=True)
        active_indices = active_indices[sorted_order]
        active_values = active_values[sorted_order]

        if top_k is not None:
            active_indices = active_indices[:top_k]
            active_values = active_values[:top_k]

        features = [
            SAEFeature(
                feature_idx=idx.item(),
                activation_strength=val.item(),
                layer_idx=layer_idx,
            )
            for idx, val in zip(active_indices, active_values)
        ]

        sparsity = 1.0 - (active_mask.sum().item() / self.d_sae)

        return SAEScanResult(
            features=features,
            sparsity=sparsity,
            reconstruction_loss=recon_loss,
            layer_idx=layer_idx,
            raw_latents=latents,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "SparseAutoencoder":
        """Load a pretrained SAE from disk."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            d_model=checkpoint["d_model"],
            d_sae=checkpoint["d_sae"],
            l1_coefficient=checkpoint.get("l1_coefficient", 5e-3),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def save(self, path: str | Path):
        """Save the SAE to disk."""
        torch.save(
            {
                "d_model": self.d_model,
                "d_sae": self.d_sae,
                "l1_coefficient": self.l1_coefficient,
                "state_dict": self.state_dict(),
            },
            path,
        )
