"""
Activation hooking infrastructure for transformer models.

Provides a clean interface to capture intermediate activations from any
HuggingFace or TransformerLens-compatible model. These raw activations
are the "signal" that downstream modules (SAE, connectivity, viz) consume.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class ActivationCapture:
    """Container for captured activations from a forward pass."""
    layer_name: str
    layer_idx: int
    activations: torch.Tensor
    token_positions: Optional[list[int]] = None
    metadata: dict = field(default_factory=dict)


class ActivationHook:
    """
    Attaches forward hooks to transformer layers and captures activations.

    Supports:
    - HuggingFace transformers (AutoModel, GPT2, LLaMA, etc.)
    - TransformerLens HookedTransformer
    - nnsight NNsight models

    Usage:
        hook = ActivationHook(model)
        hook.attach(layers=[0, 6, 11], components=["mlp", "attn"])
        captures = hook.run("The cat sat on the mat")
        hook.detach()
    """

    def __init__(self, model: nn.Module, tokenizer=None, backend: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = self._detect_backend(backend)
        self._handles: list = []
        self._captures: list[ActivationCapture] = []
        self._target_layers: set[str] = set()

    def _detect_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        cls_name = type(self.model).__name__
        if "HookedTransformer" in cls_name:
            return "transformer_lens"
        if "NNsight" in cls_name:
            return "nnsight"
        return "huggingface"

    def attach(
        self,
        layers: Optional[list[int]] = None,
        components: Optional[list[str]] = None,
        custom_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    ) -> "ActivationHook":
        """
        Attach hooks to specified layers and components.

        Args:
            layers: Layer indices to hook. None = all layers.
            components: Which sub-modules to hook ("mlp", "attn", "residual").
                       None = all components.
            custom_filter: Custom filter function(name, module) -> bool.
        """
        self.detach()  # clean slate
        self._captures = []

        for name, module in self.model.named_modules():
            if custom_filter and not custom_filter(name, module):
                continue
            if self._should_hook(name, layers, components):
                handle = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._handles.append(handle)
                self._target_layers.add(name)

        return self

    def _should_hook(
        self,
        name: str,
        layers: Optional[list[int]],
        components: Optional[list[str]],
    ) -> bool:
        """Determine if a named module should be hooked."""
        # Extract layer index from name (e.g., "transformer.h.6.mlp" -> 6)
        parts = name.split(".")
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break

        if layers is not None and layer_idx is not None:
            if layer_idx not in layers:
                return False

        if components is not None:
            component_match = any(c in name for c in components)
            if not component_match:
                return False

        # Only hook leaf-ish modules that produce activations
        has_weight = hasattr(self, "weight") if False else True  # placeholder
        return any(keyword in name for keyword in ["mlp", "attn", "ln", "residual", "hook_"])

    def _make_hook(self, layer_name: str):
        """Create a forward hook closure for the given layer."""
        def hook_fn(module, input, output):
            # Handle various output formats
            if isinstance(output, tuple):
                tensor = output[0]
            elif isinstance(output, torch.Tensor):
                tensor = output
            else:
                return  # skip non-tensor outputs

            # Extract layer index
            parts = layer_name.split(".")
            layer_idx = -1
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break

            self._captures.append(
                ActivationCapture(
                    layer_name=layer_name,
                    layer_idx=layer_idx,
                    activations=tensor.detach().cpu(),
                )
            )
        return hook_fn

    @torch.no_grad()
    def run(self, text: str, **kwargs) -> list[ActivationCapture]:
        """
        Run a forward pass and return captured activations.

        Args:
            text: Input text to process.
            **kwargs: Additional arguments passed to the model.

        Returns:
            List of ActivationCapture objects, one per hooked layer.
        """
        self._captures = []

        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text input. Pass tokenizer to constructor.")

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        self.model(**inputs, **kwargs)

        return list(self._captures)

    def detach(self):
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._target_layers = set()

    def __del__(self):
        self.detach()

    def __repr__(self):
        return (
            f"ActivationHook(backend={self.backend!r}, "
            f"hooked_layers={len(self._target_layers)})"
        )
