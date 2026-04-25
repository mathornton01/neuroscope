FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves ~1.5GB vs full CUDA build)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the app dependencies
COPY pyproject.toml .
COPY neuroscope/ neuroscope/
COPY tests/ tests/
COPY README.md .
COPY LICENSE .

RUN pip install --no-cache-dir -e ".[viz]"

# Pre-download GPT-2 model so startup is fast
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('gpt2'); \
    AutoModelForCausalLM.from_pretrained('gpt2')"

# Railway sets PORT env var
ENV NEUROSCOPE_MODEL=gpt2
EXPOSE 8080

CMD ["python", "-m", "neuroscope.realtime.server"]
