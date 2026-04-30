FROM python:3.11-slim

WORKDIR /app

# Install system deps + curl for Ollama install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install CPU-only PyTorch first (saves ~1.5GB vs full CUDA build)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the app dependencies + requests for Ollama HTTP calls
COPY pyproject.toml .
COPY neuroscope/ neuroscope/
COPY tests/ tests/
COPY README.md .
COPY LICENSE .

RUN pip install --no-cache-dir -e ".[viz]" requests

# Pre-download GPT-2 model so HuggingFace backend starts fast
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('gpt2'); \
    AutoModelForCausalLM.from_pretrained('gpt2')"

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Railway sets PORT env var
ENV NEUROSCOPE_MODEL=gpt2
ENV OLLAMA_MODELS=/app/ollama_models
ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 8080

CMD ["/app/start.sh"]
