#!/bin/bash
set -e

echo "=== NeuroScope Startup ==="

# Start Ollama in background
echo "Starting Ollama daemon..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to come online..."
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    sleep 1
done

# Pull a small model if not already present
echo "Pulling qwen2.5:1.5b model..."
ollama pull qwen2.5:1.5b || echo "Warning: Could not pull qwen2.5:1.5b"

echo "Pulling llama3.2:1b model..."
ollama pull llama3.2:1b || echo "Warning: Could not pull llama3.2:1b"

echo "Available Ollama models:"
ollama list 2>/dev/null || echo "(none yet)"

# Start NeuroScope server
echo "Starting NeuroScope server..."
exec python -m neuroscope.realtime.server
