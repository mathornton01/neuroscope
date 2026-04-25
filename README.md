# NeuroScope

**Functional MRI for Large Language Models**

A unified toolkit for real-time visualization and analysis of LLM internal activations, functional connectivity, and feature extraction. Think of it as a brain scanner for transformer models.

## Core Capabilities

- **Feature Extraction** -- SAE-based decomposition of activations into interpretable features (the "voxels")
- **Functional Connectivity** -- Graph-based mapping of co-activation patterns (the "connectome")
- **Activation Narration** -- Oracle models that describe what features represent (the "radiologist")
- **Real-time Visualization** -- 3D brain-scan-style dashboard (the "scanner")

## Quick Start

```bash
pip install -e ".[all]"
```

```python
from neuroscope import NeuroScanner
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

scanner = NeuroScanner(model, tokenizer, layers=[0, 6, 11])
result = scanner.scan("The capital of France is")

for sae_result in result.sae_results:
    print(f"Layer {sae_result.layer_idx}: {len(sae_result.features)} active features")
    for f in sae_result.features[:5]:
        print(f"  Feature {f.feature_idx}: strength={f.activation_strength:.3f}")
```

## Architecture

```
neuroscope/
  extractors/      # Activation hooks + SAE feature extraction
    hooks.py       # Forward hook infrastructure
    sae.py         # Sparse autoencoder implementation
  connectivity/    # Functional connectivity analysis
    graph.py       # Co-activation graphs + network discovery
  visualization/   # 3D rendering + web dashboard
  scanner.py       # Main orchestrator (the "fMRI machine")
```

## Key References

- [Brain-Inspired Exploration of Functional Networks in LLMs](https://arxiv.org/abs/2502.20408) (COLING 2025)
- [Probing Neural Topology of LLMs](https://arxiv.org/abs/2506.01042) (2025)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (Anthropic, 2024)
- [Activation Oracles](https://arxiv.org/abs/2512.15674) (2025)
- [SAE Survey](https://arxiv.org/abs/2503.05613) (2025)

## License

MIT
