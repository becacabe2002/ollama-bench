#!/bin/bash
set -euo pipefail

MODELS=(
  phi4-mini
  llama3.2:3b
  qwen3.5:4b
  gemma3:4b
  qwen3.5:9b
  llama3.1:8b
  ministral-3:8b
)

for m in "${MODELS[@]}"; do
  echo "Pulling $m ..."
  ollama pull "$m"
done

echo "---"
ollama list   # verify + record digests for the manifest