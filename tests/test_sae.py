"""Tests for the Sparse Autoencoder module."""

import torch
import pytest
from neuroscope.extractors.sae import SparseAutoencoder


class TestSparseAutoencoder:
    def test_init(self):
        sae = SparseAutoencoder(d_model=768, d_sae=768 * 8)
        assert sae.d_model == 768
        assert sae.d_sae == 768 * 8

    def test_forward_shape(self):
        sae = SparseAutoencoder(d_model=64, d_sae=256)
        x = torch.randn(2, 10, 64)  # (batch, seq_len, d_model)
        recon, latents, loss = sae(x)
        assert recon.shape == x.shape
        assert latents.shape == (2, 10, 256)
        assert loss.dim() == 0  # scalar

    def test_sparsity(self):
        sae = SparseAutoencoder(d_model=64, d_sae=512, l1_coefficient=0.1)
        x = torch.randn(4, 10, 64)
        _, latents, _ = sae(x)
        # With ReLU, at least some values should be zero
        zero_fraction = (latents == 0).float().mean().item()
        assert zero_fraction > 0.0, "Expected some sparsity from ReLU"

    def test_extract_features(self):
        sae = SparseAutoencoder(d_model=64, d_sae=256)
        x = torch.randn(1, 10, 64)
        result = sae.extract_features(x, layer_idx=6, top_k=5)
        assert len(result.features) <= 5
        assert result.layer_idx == 6
        assert 0.0 <= result.sparsity <= 1.0

    def test_save_load(self, tmp_path):
        sae = SparseAutoencoder(d_model=32, d_sae=128)
        path = tmp_path / "test_sae.pt"
        sae.save(path)
        loaded = SparseAutoencoder.from_pretrained(path)
        assert loaded.d_model == 32
        assert loaded.d_sae == 128
