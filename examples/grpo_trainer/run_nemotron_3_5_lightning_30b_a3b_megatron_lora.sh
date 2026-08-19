#!/usr/bin/env bash
# LoRA GRPO | NVIDIA Nemotron 3.5 Lightning 30B-A3B | Megatron actor
#
# This wrapper enables the Nemotron-H LoRA recipe on top of the validated GRPO
# launcher, retaining R3 router replay, BF16 raw logprobs, and one-token MTP
# rollout speculation. Megatron-Bridge merges adapters into full Hugging Face
# tensors for vLLM synchronization; LoRA reduces actor training memory, but not
# actor-to-rollout synchronization volume.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_LAUNCHER="${SCRIPT_DIR}/run_nemotron_3_5_lightning_30b_a3b_megatron.sh"

LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'["linear_qkv","linear_proj","linear_fc1","linear_fc2","in_proj","out_proj"]'}

if [[ ! "${LORA_RANK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LORA_RANK must be a positive integer, got: ${LORA_RANK}" >&2
    exit 1
fi
if [[ ! "${LORA_ALPHA}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LORA_ALPHA must be a positive integer, got: ${LORA_ALPHA}" >&2
    exit 1
fi

# The v0.7 adapter checkpoint format does not support virtual pipeline chunks.
ACTOR_PP=${ACTOR_PP:-1}
if [[ "${ACTOR_PP}" != 1 ]]; then
    echo "LoRA adapter checkpointing requires ACTOR_PP=1." >&2
    exit 1
fi

export ACTOR_PP
export ACTOR_LR=${ACTOR_LR:-3e-6}
# Leave enough headroom to restore vLLM's KV cache after adapter checkpoints.
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.60}
export PROJECT_NAME=${PROJECT_NAME:-verl_grpo_dapo_math}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-nemotron-3-5-lightning-30b-a3b-megatron-lora-r${LORA_RANK}}

# The pinned Megatron-Core stack cannot serialize the distributed optimizer
# state for this adapter layout. These checkpoints contain adapter weights and
# trainer metadata, but do not provide exact optimizer-state resume.
exec bash "${BASE_LAUNCHER}" \
    "actor_rollout_ref.model.lora.type=lora" \
    "actor_rollout_ref.model.lora.rank=${LORA_RANK}" \
    "actor_rollout_ref.model.lora.alpha=${LORA_ALPHA}" \
    "actor_rollout_ref.model.lora.dropout=${LORA_DROPOUT}" \
    "actor_rollout_ref.model.lora.dropout_position=pre" \
    "actor_rollout_ref.model.lora.lora_A_init_method=xavier" \
    "actor_rollout_ref.model.lora.lora_B_init_method=zero" \
    "actor_rollout_ref.model.lora.target_modules=${LORA_TARGET_MODULES}" \
    "+actor_rollout_ref.model.lora.share_expert_adapters=True" \
    "+actor_rollout_ref.model.lora.normalize_moe_lora=False" \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]" \
    "$@"
