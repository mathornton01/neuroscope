"""
NeuroScope Scanner -- the main entry point.

Orchestrates the full pipeline: hook activations -> extract features ->
map connectivity -> generate visualization. This is the "fMRI machine."
"""

from __future__ import annotations

import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from .extractors.hooks import ActivationHook, ActivationCapture
from .extractors.sae import SparseAutoencoder, SAEScanResult
from .connectivity.graph import ConnectivityGraph


@dataclass
class ScanResult:
    """Complete result of a NeuroScope scan."""
    input_text: str
    activation_captures: list[ActivationCapture]
    sae_results: list[SAEScanResult]
    connectivity: Optional[dict] = None
    summary: Optional[str] = None


class NeuroScanner:
    """
    The unified LLM brain scanner.

    Usage:
        scanner = NeuroScanner(model, tokenizer)
        scanner.load_sae("path/to/sae_layer_6.pt", layer_idx=6)

        # Single scan
        result = scanner.scan("The capital of France is")

        # Build connectivity map over many inputs
        for text in corpus:
            scanner.scan(text, record_connectivity=True)
        networks = scanner.get_functional_networks()

        # Launch live dashboard
        scanner.serve(port=8080)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        layers: Optional[list[int]] = None,
        components: Optional[list[str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hook = ActivationHook(model, tokenizer)
        self.hook.attach(layers=layers, components=components)

        self.saes: dict[int, SparseAutoencoder] = {}
        self.connectivity = ConnectivityGraph()
        self._scan_count = 0

    def load_sae(self, path: str, layer_idx: int):
        """Load a pretrained SAE for a specific layer."""
        sae = SparseAutoencoder.from_pretrained(path)
        self.saes[layer_idx] = sae
        return self

    def train_sae(
        self,
        layer_idx: int,
        d_sae: Optional[int] = None,
        expansion_factor: int = 8,
        n_epochs: int = 10,
        corpus: Optional[list[str]] = None,
    ):
        """
        Train an SAE on activations from the specified layer.
        Placeholder -- full training loop to be implemented.
        """
        raise NotImplementedError(
            "SAE training not yet implemented. Use load_sae() with a pretrained SAE, "
            "or see examples/train_sae.py for a standalone training script."
        )

    def scan(
        self,
        text: str,
        top_k_features: int = 20,
        record_connectivity: bool = False,
    ) -> ScanResult:
        """
        Run a full brain scan on the given text.

        Args:
            text: Input text to scan.
            top_k_features: Number of top features to extract per layer.
            record_connectivity: If True, record activations for
                                connectivity analysis.

        Returns:
            ScanResult with activations, features, and optional connectivity.
        """
        # Step 1: Capture raw activations
        captures = self.hook.run(text)

        # Step 2: Extract SAE features where we have trained SAEs
        sae_results = []
        for capture in captures:
            if capture.layer_idx in self.saes:
                sae = self.saes[capture.layer_idx]
                result = sae.extract_features(
                    capture.activations,
                    layer_idx=capture.layer_idx,
                    top_k=top_k_features,
                )
                sae_results.append(result)

        # Step 3: Record connectivity if requested
        if record_connectivity:
            activation_snapshot = {}
            for result in sae_results:
                for feature in result.features:
                    node_id = f"layer_{feature.layer_idx}_feature_{feature.feature_idx}"
                    activation_snapshot[node_id] = feature.activation_strength
            self.connectivity.record_activations(activation_snapshot)

        self._scan_count += 1

        return ScanResult(
            input_text=text,
            activation_captures=captures,
            sae_results=sae_results,
        )

    def get_functional_networks(self, min_size: int = 3):
        """
        Discover functional networks from accumulated scans.
        Requires at least 10 scans with record_connectivity=True.
        """
        self.connectivity.build_graph()
        return self.connectivity.find_networks(min_size=min_size)

    def get_hub_features(self, top_k: int = 10):
        """Find the most connected features across the model."""
        return self.connectivity.find_hubs(top_k=top_k)

    def serve(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Launch the real-time visualization dashboard.
        Placeholder -- will use FastAPI + WebSocket + Three.js.
        """
        raise NotImplementedError(
            "Dashboard not yet implemented. See frontend/ for the planned architecture."
        )

    def __repr__(self):
        return (
            f"NeuroScanner(backend={self.hook.backend!r}, "
            f"saes={list(self.saes.keys())}, "
            f"scans={self._scan_count})"
        )
