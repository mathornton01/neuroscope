"""
NeuroScope Real-Time Server

Loads a small transformer model (GPT-2), hooks every layer,
generates tokens one at a time, and streams activation data
plus per-layer "logit lens" predictions to the web frontend.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# ---------------------------------------------------------------------------
# Activation capture with logit lens
# ---------------------------------------------------------------------------

class RealtimeHook:
    """Hook system that captures residual stream at each layer exit
    and projects through the unembedding head to get per-layer predictions."""

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._handles = []
        self._layer_activations: dict[str, dict] = {}
        self._residual_streams: dict[int, torch.Tensor] = {}

    def attach(self):
        """Hook every transformer block's output (residual stream)
        plus attention and MLP sub-modules for activation stats."""
        self.detach()
        for name, module in self.model.named_modules():
            # Hook full transformer blocks to capture residual stream
            # GPT-2: transformer.h.0, transformer.h.1, etc.
            if hasattr(self.model, 'transformer'):
                # GPT-2 style
                for i, block in enumerate(self.model.transformer.h):
                    handle = block.register_forward_hook(self._make_residual_hook(i))
                    self._handles.append(handle)
                break
            else:
                # Generic: hook modules with attn/mlp
                if any(k in name for k in [".attn", ".mlp", ".ln_"]):
                    if len(list(module.children())) == 0 or "attn" in name:
                        handle = module.register_forward_hook(self._make_activation_hook(name))
                        self._handles.append(handle)

    def _make_residual_hook(self, layer_idx: int):
        """Capture the residual stream output of each transformer block."""
        def hook_fn(module, input, output):
            # GPT-2 block output is (hidden_states, present, ...)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._residual_streams[layer_idx] = hidden.detach()

            # Also compute activation stats
            act = hidden.detach().float()
            mean_mag = act.norm(dim=-1).mean().item()
            max_val = act.abs().max().item()
            sparsity = (act.abs() < 0.01).float().mean().item()

            self._layer_activations[f"layer_{layer_idx}"] = {
                "name": f"layer_{layer_idx}",
                "layer_idx": layer_idx,
                "mean_magnitude": round(mean_mag, 4),
                "max_activation": round(max_val, 4),
                "sparsity": round(sparsity, 4),
            }
        return hook_fn

    def _make_activation_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            elif isinstance(output, torch.Tensor):
                tensor = output
            else:
                return
            act = tensor.detach().float()
            mean_mag = act.norm(dim=-1).mean().item()
            max_val = act.abs().max().item()
            sparsity = (act.abs() < 0.01).float().mean().item()
            layer_idx = -1
            for part in name.split("."):
                if part.isdigit():
                    layer_idx = int(part)
                    break
            self._layer_activations[name] = {
                "name": name,
                "layer_idx": layer_idx,
                "mean_magnitude": round(mean_mag, 4),
                "max_activation": round(max_val, 4),
                "sparsity": round(sparsity, 4),
            }
        return hook_fn

    def clear(self):
        self._layer_activations = {}
        self._residual_streams = {}

    def get_activations(self) -> dict:
        return dict(self._layer_activations)

    @torch.no_grad()
    def get_layer_predictions(self, top_k: int = 5) -> list:
        """Apply logit lens: project each layer's residual stream through
        the final LayerNorm + unembedding to get per-layer token predictions."""
        if not self._residual_streams:
            return []

        predictions = []

        # Get the final LN and lm_head
        if hasattr(self.model, 'transformer'):
            final_ln = self.model.transformer.ln_f
            lm_head = self.model.lm_head
        else:
            return []

        for layer_idx in sorted(self._residual_streams.keys()):
            hidden = self._residual_streams[layer_idx]
            # Take only the last token position
            last_hidden = hidden[:, -1:, :]

            # Apply final LN then unembedding (logit lens)
            normed = final_ln(last_hidden)
            logits = lm_head(normed)  # [1, 1, vocab_size]
            logits = logits[0, 0]     # [vocab_size]

            # Get top-k predictions with probabilities
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(top_k)

            layer_preds = []
            for i in range(top_k):
                token_id = top_ids[i].item()
                token_text = self.tokenizer.decode([token_id])
                prob = top_probs[i].item()
                layer_preds.append({
                    "token": token_text,
                    "token_id": token_id,
                    "prob": round(prob, 4),
                })

            # Also get entropy of the distribution (how "decided" the layer is)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            max_entropy = float(np.log(probs.shape[0]))

            predictions.append({
                "layer": layer_idx,
                "top_k": layer_preds,
                "entropy": round(entropy, 3),
                "entropy_normalized": round(entropy / max_entropy, 4),
                "confidence": round(top_probs[0].item(), 4),
            })

        return predictions

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._residual_streams = {}


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages model loading and token-by-token generation."""

    def __init__(self, model_name: str = "gpt2"):
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.model.eval()
        self.hook = RealtimeHook(self.model, self.tokenizer)
        self.hook.attach()

        config = self.model.config
        self.model_info = {
            "name": model_name,
            "n_layers": config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers,
            "n_heads": config.n_head if hasattr(config, "n_head") else config.num_attention_heads,
            "d_model": config.n_embd if hasattr(config, "n_embd") else config.hidden_size,
            "vocab_size": config.vocab_size,
        }
        print(f"Model loaded: {self.model_info}")

    @torch.no_grad()
    def generate_step(self, input_ids: torch.Tensor, temperature: float = 0.8, past_key_values=None):
        """Run one forward pass and return next token + activations + layer predictions.

        Note: When using KV cache, the residual stream hooks only see the
        single-token slice, which is exactly what we want for logit lens
        on the current generation step.
        """
        self.hook.clear()

        # We need full forward pass (no KV cache) to get all layer residuals
        # for the logit lens to work correctly on the last token
        outputs = self.model(input_ids, use_cache=True)

        logits = outputs.logits[:, -1, :]
        new_past = outputs.past_key_values

        # Get top-k from the ACTUAL final output for comparison
        final_probs = torch.softmax(logits, dim=-1)
        final_top_probs, final_top_ids = final_probs.topk(5)
        final_predictions = []
        for i in range(5):
            token_id = final_top_ids[0, i].item()
            final_predictions.append({
                "token": self.tokenizer.decode([token_id]),
                "token_id": token_id,
                "prob": round(final_top_probs[0, i].item(), 4),
            })

        # Sample next token
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        activations = self.hook.get_activations()
        layer_predictions = self.hook.get_layer_predictions(top_k=5)

        return next_token, activations, layer_predictions, final_predictions, None  # no cache reuse

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        callback=None,
    ):
        """Generate tokens one at a time with logit lens data."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Process prompt
        self.hook.clear()
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        prompt_activations = self.hook.get_activations()
        prompt_predictions = self.hook.get_layer_predictions(top_k=5)

        # Get prompt's final predictions
        prompt_logits = outputs.logits[:, -1, :]
        prompt_probs = torch.softmax(prompt_logits, dim=-1)
        prompt_top_probs, prompt_top_ids = prompt_probs.topk(5)
        prompt_final_preds = []
        for i in range(5):
            token_id = prompt_top_ids[0, i].item()
            prompt_final_preds.append({
                "token": self.tokenizer.decode([token_id]),
                "token_id": token_id,
                "prob": round(prompt_top_probs[0, i].item(), 4),
            })

        if callback:
            await callback({
                "type": "prompt_processed",
                "prompt": prompt,
                "n_tokens": input_ids.shape[1],
                "activations": prompt_activations,
                "layer_predictions": prompt_predictions,
                "final_predictions": prompt_final_preds,
                "model_info": self.model_info,
            })

        # Generate tokens one by one
        for step in range(max_tokens):
            next_token, activations, layer_predictions, final_predictions, _ = self.generate_step(
                input_ids, temperature
            )
            input_ids = torch.cat([input_ids, next_token], dim=1)

            token_text = self.tokenizer.decode(next_token[0])
            token_id = next_token[0].item()

            if callback:
                await callback({
                    "type": "token_generated",
                    "step": step + 1,
                    "token": token_text,
                    "token_id": token_id,
                    "activations": activations,
                    "layer_predictions": layer_predictions,
                    "final_predictions": final_predictions,
                })

            if token_id == self.tokenizer.eos_token_id:
                break

            await asyncio.sleep(0.005)

        if callback:
            full_text = self.tokenizer.decode(input_ids[0])
            await callback({
                "type": "generation_complete",
                "full_text": full_text,
                "total_tokens": input_ids.shape[1],
            })


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

manager: Optional[ModelManager] = None

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    model_name = os.environ.get("NEUROSCOPE_MODEL", "gpt2")
    manager = ModelManager(model_name)
    yield


app = FastAPI(title="NeuroScope Real-Time", lifespan=lifespan)


@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>NeuroScope Real-Time</h1><p>Connect via WebSocket at /ws</p>")


@app.get("/api/model-info")
async def model_info():
    return manager.model_info


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()

            if data.get("type") == "generate":
                prompt = data.get("prompt", "Hello")
                max_tokens = min(data.get("max_tokens", 50), 200)
                temperature = data.get("temperature", 0.8)

                async def send_update(payload):
                    await ws.send_json(payload)

                await manager.stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    callback=send_update,
                )

            elif data.get("type") == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass


STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn
    port = int(os.environ.get("PORT", os.environ.get("NEUROSCOPE_PORT", "8080")))
    print(f"\n=== NeuroScope Real-Time ===")
    print(f"Starting on http://0.0.0.0:{port}")
    print(f"Open your browser to see the neural network light up!\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
