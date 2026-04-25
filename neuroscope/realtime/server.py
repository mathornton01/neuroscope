"""
NeuroScope Real-Time Server

Loads a small transformer model (GPT-2), hooks every layer,
generates tokens one at a time, and streams activation data
to a web frontend via WebSocket.
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
# Activation capture
# ---------------------------------------------------------------------------

class RealtimeHook:
    """Lightweight hook system for real-time activation streaming."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles = []
        self._layer_activations: dict[str, dict] = {}

    def attach(self):
        """Hook every transformer block's key sub-modules."""
        self.detach()
        for name, module in self.model.named_modules():
            # Hook attention outputs, MLP outputs, and layer norms
            if any(k in name for k in [".attn", ".mlp", ".ln_"]):
                # Only hook "leaf-ish" modules (not containers)
                if len(list(module.children())) == 0 or "attn" in name:
                    handle = module.register_forward_hook(self._make_hook(name))
                    self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            elif isinstance(output, torch.Tensor):
                tensor = output
            else:
                return

            # Compute per-layer statistics
            act = tensor.detach().float()

            # Mean activation magnitude (L2 norm per position, then mean)
            mean_magnitude = act.norm(dim=-1).mean().item()

            # Max activation
            max_val = act.abs().max().item()

            # Sparsity (fraction of near-zero activations)
            sparsity = (act.abs() < 0.01).float().mean().item()

            # Per-head attention pattern if it's attention
            head_data = None
            if "attn" in name and len(act.shape) >= 3:
                # Just send the overall magnitude, not full attention matrix
                head_data = act.norm(dim=-1).squeeze(0).tolist()
                if isinstance(head_data, float):
                    head_data = [head_data]
                # Limit to reasonable size
                if len(head_data) > 50:
                    head_data = head_data[:50]

            # Extract layer index
            layer_idx = -1
            for part in name.split("."):
                if part.isdigit():
                    layer_idx = int(part)
                    break

            self._layer_activations[name] = {
                "name": name,
                "layer_idx": layer_idx,
                "mean_magnitude": round(mean_magnitude, 4),
                "max_activation": round(max_val, 4),
                "sparsity": round(sparsity, 4),
                "shape": list(act.shape),
            }

        return hook_fn

    def clear(self):
        self._layer_activations = {}

    def get_activations(self) -> dict:
        return dict(self._layer_activations)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []


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
        self.hook = RealtimeHook(self.model)
        self.hook.attach()

        # Get model architecture info
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
        """Run one forward pass and return next token + activations.

        Uses KV cache when past_key_values is provided -- only the last
        token is processed through the model, making generation ~10x faster.
        """
        self.hook.clear()

        if past_key_values is not None:
            # Only feed the last token, reuse cached KV pairs
            outputs = self.model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
        else:
            outputs = self.model(input_ids, use_cache=True)

        logits = outputs.logits[:, -1, :]
        new_past = outputs.past_key_values

        # Sample next token
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        activations = self.hook.get_activations()
        return next_token, activations, new_past

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        callback=None,
    ):
        """Generate tokens one at a time, calling callback with each token's data.

        Uses KV caching so each step only runs the new token through the model
        instead of reprocessing the entire sequence. Much faster on CPU.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # First, send the prompt processing (full sequence, builds initial cache)
        self.hook.clear()
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        prompt_activations = self.hook.get_activations()

        if callback:
            await callback({
                "type": "prompt_processed",
                "prompt": prompt,
                "n_tokens": input_ids.shape[1],
                "activations": prompt_activations,
                "model_info": self.model_info,
            })

        # Generate tokens one by one with KV cache
        for step in range(max_tokens):
            next_token, activations, past_key_values = self.generate_step(
                input_ids, temperature, past_key_values=past_key_values
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
                })

            # Check for EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            # Small yield to allow WebSocket messages to flush
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
                    # Convert numpy/torch types for JSON serialization
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


# Ensure static directory exists and mount it
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn
    # Railway sets PORT; fall back to NEUROSCOPE_PORT or 8080
    port = int(os.environ.get("PORT", os.environ.get("NEUROSCOPE_PORT", "8080")))
    print(f"\n=== NeuroScope Real-Time ===")
    print(f"Starting on http://0.0.0.0:{port}")
    print(f"Open your browser to see the neural network light up!\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
