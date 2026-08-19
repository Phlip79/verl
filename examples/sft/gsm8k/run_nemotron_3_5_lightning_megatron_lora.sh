#!/usr/bin/env bash
# LoRA SFT | NVIDIA Nemotron 3.5 Lightning 30B-A3B | Megatron
#
# This wrapper enables the Nemotron-H LoRA recipe on top of the full-SFT
# launcher. The defaults match Megatron-Bridge's Lightning PEFT recipe:
# rank 32, alpha 32, and adapters on attention, MoE/MLP, and Mamba
# projections. It uses the validated 2x8-GPU topology by default.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_LAUNCHER="${SCRIPT_DIR}/run_nemotron_3_5_lightning_megatron.sh"

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
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
if [[ "${PP_SIZE}" != 1 || "${VPP_SIZE}" != null ]]; then
    echo "LoRA adapter checkpointing requires PP_SIZE=1 and VPP_SIZE=null." >&2
    exit 1
fi

export NNODES=${NNODES:-2}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-${NUM_GPUS:-8}}
export PP_SIZE
export VPP_SIZE
export LR=${LR:-1e-4}
export SAVE_FREQ=${SAVE_FREQ:-1000}
export PROJECT_NAME=${PROJECT_NAME:-verl_sft_gsm8k}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-nemotron-3-5-lightning-megatron-lora-r${LORA_RANK}}

# The pinned Megatron-Core stack cannot serialize the distributed optimizer
# state for this adapter layout. These checkpoints contain adapter weights and
# trainer metadata, but do not provide exact optimizer-state resume.
exec bash "${BASE_LAUNCHER}" \
    "+model.lora.type=lora" \
    "+model.lora.rank=${LORA_RANK}" \
    "+model.lora.alpha=${LORA_ALPHA}" \
    "+model.lora.dropout=${LORA_DROPOUT}" \
    "+model.lora.dropout_position=pre" \
    "+model.lora.lora_A_init_method=xavier" \
    "+model.lora.lora_B_init_method=zero" \
    "+model.lora.target_modules=${LORA_TARGET_MODULES}" \
    "+model.lora.share_expert_adapters=True" \
    "+model.lora.normalize_moe_lora=False" \
    "checkpoint.save_contents=[model,extra]" \
    "$@"
