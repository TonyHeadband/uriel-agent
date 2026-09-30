#!/usr/bin/env bash
# (Re)create the decision models in the house Ollama. tev1-4b-cpu is tev1:4b pinned to the CPU: on the 8 GB GPU it
# would evict qwen3:8b (a ~26 s reload), and /v1/systemone ignores a per-request num_gpu.
# Usage: scripts/create-decision-models.sh   (needs `ssh homelab` with kubectl access to the ai namespace)
set -euo pipefail
ssh homelab 'export KUBECONFIG=~/.kube/config
kubectl -n ai exec deploy/ollama -- ollama pull tev1:4b-q4_K_M
kubectl -n ai exec deploy/ollama -- sh -c "printf \"FROM tev1:4b-q4_K_M\nPARAMETER num_gpu 0\n\" > /tmp/tev1-4b-cpu.Modelfile \
  && ollama create tev1-4b-cpu -f /tmp/tev1-4b-cpu.Modelfile"'
