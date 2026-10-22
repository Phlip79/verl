# Guide to Using MTP in SFT/RL Training and Inference

**Author**: `https://github.com/meituan-search`

Last updated: 10/22/2026

## 1. Scope of Support

Currently, RL training can be performed on mimo-7B-RL, Qwen-next, and Deepseek series models based on the MTP architecture. SFT examples cover NVIDIA Nemotron 3.5 Super VL and Nemotron 3.5 Lightning through Megatron-Core's native `HybridModel` MTP implementation. Super VL adds image/text training through its multimodal wrapper and requires newer source dependencies, described below; GPU validation of this verl recipe is pending. Nemotron 3.5 Lightning additionally has a GRPO recipe combining vLLM rollout, R3 router replay, and one-token MTP speculation. The support rules for training and inference engines are as follows:

- **Training Engine**: Only supports the `Megatron-Bridge + megatron` combination; other training engines are not compatible at this time;

- **Inference Engine**: Generic MTP is available when the model is supported by the selected engine. The Nemotron 3.5 Lightning GRPO recipe currently supports vLLM only;

- **Dependency Versions**:

    - For the legacy mbridge backend, apply the patches and review suggestions from [mbridge#62](https://github.com/ISEEKYAN/mbridge/pull/62). The Nemotron recipes use NVIDIA Megatron-Bridge (`vanilla_mbridge=False`).

    - The Nemotron image pins Megatron-Bridge source revision **e97a67ac**, the matching Megatron-Core revision **89b28875**, vLLM **0.29.0**, and Transformers **5.20.1**. The newer Bridge/Core pair supplies Super VL's expanded-sequence model wrapper as well as native `HybridModel` MTP. Unlike upstream `main`, `release/v0.9.0` has no `uv.lock`; the Dockerfile installs these dependencies explicitly and overrides the release's Transformers upper bound for this isolated model environment. Full image and GPU training validation are still pending.

    - `docker/Dockerfile.nemotron_3_5_lightning` retains its historical filename but now builds the shared NGC `26.06-py3` x86-64 environment for Super VL SFT and the Lightning recipes. It includes the Super VL launcher and Pokemon data-preparation script; no manual dependency upgrade is needed inside the resulting image. From the repository root, build it with:

      ```bash
      docker buildx build --platform linux/amd64 --load \
        --build-arg VERL_COMMIT="$(git rev-parse HEAD)" \
        -f docker/Dockerfile.nemotron_3_5_lightning \
        -t gitlab-master.nvidia.com:5005/ppetrakian/containers/verl/super:base .
      ```

      `VERL_COMMIT` records the source revision in the image metadata; the Docker build uses the current build context, so build from a clean checkout of that revision. `--load` makes the image available locally for `docker run` and `docker push`. Build-time checks verify the source revisions, native MTP kernels, Super VL wrapper contract, and required example files; they do not substitute for GPU training validation.

      For an immutable Lightning run, download the model and GRPO datasets at the validated revisions before launching:

      ```bash
      hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
        --revision d468880b6ad3c6e0d21377ce7242adaea4cc884d \
        --local-dir "$HOME/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
      hf download BytedTsinghua-SIA/DAPO-Math-17k data/dapo-math-17k.parquet \
        --repo-type dataset --revision 65877096c24ffa7abc4e4fa5edb95cf3413a5674 \
        --local-dir "$HOME/verl/data/DAPO-Math-17k"
      hf download BytedTsinghua-SIA/AIME-2024 data/aime-2024.parquet \
        --repo-type dataset --revision aa49075e24ad594b79fdf0bdcefa735c2181be67 \
        --local-dir "$HOME/verl/data/AIME-2024"
      ```

      Set `MODEL_PATH` to the downloaded model directory. The GRPO launcher defaults match the two dataset paths above. The release's `HFModelConfig` does not expose a Hub revision field, so a local snapshot is required for exact model reproduction.

    - Megatron-Bridge v0.6.2 includes the `MiMoForCausalLM` provider and MTP weight mappings, including the input-projection layout conversion. The MiMo changes from [#2387](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2387) are merged; no separate MiMo bridge patch is required.

    - Megatron-Core 0.19.2 includes `padding_mask` in `MultiTokenPredictionLayer._checkpointed_forward`, required when using MTP with `recompute_granularity=full`. Older 0.18.0/0.18.2 environments need the compatibility fix tracked in [#4933](https://github.com/NVIDIA/Megatron-LM/issues/4933); the CUDA version pair above does not apply to the separate Ascend/MindSpeed environment.

    - SGLang rollout still requires the fix from [sglang#17870](https://github.com/sgl-project/sglang/pull/17870) for MTP update-weights-from-tensor OOM behavior.

## 2. MTP Training Configuration (Core Parameters)

The MTP training process can be flexibly controlled through the following configurations. All configurations are based on the `actor_rollout_ref.model.mtp` prefix:

| Configuration Scenario | Core Parameters                                                                                                                                                                                                                                                                                               | Description                                             |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| Load MTP Parameters Only | `enable=True`                                                                                                                                                                                                                                                                                              | VRAM usage will increase, but the exported parameters include the MTP module and can be directly used for online deployment. This mode is not supported for native Megatron-Core `HybridModel` MTP; set `enable_train=True` or disable MTP.              |
| Full-Parameter MTP Training | `enable=True`<br>`enable_train=True`<br>`mtp_loss_scaling_factor=0.1`                                                                                                                                                                                                                              | MTP Loss will apply to all model parameters                            |
| Detached MTP Auxiliary Training | `enable=True`<br>`enable_train=True`<br>`detach_encoder=True`                                                                                                                                                                                                                                      | Detach the hidden state consumed by MTP so the auxiliary MTP loss updates only the MTP module. The normal SFT loss still updates the base model. |
| MTP Accelerated Rollout | 1. vLLM configuration:<br>`enable=True`<br>`enable_rollout=True`<br>`method="mtp"`<br>`num_speculative_tokens=1`<br>2. SGLang configuration:<br>`enable=True`<br>`enable_rollout=True`<br>`speculative_algorithm="EAGLE"`<br>`speculative_num_steps=2`<br>`speculative_eagle_topk=2`<br>`speculative_num_draft_tokens=4` | Achieve inference acceleration during the Rollout phase based on MTP                      |

For native Megatron-Core `HybridModel` models, `enable=True` requires `enable_train=True`. The load-without-training combination is rejected because this path uses Megatron-Core's native MTP implementation instead of verl's legacy GPT MTP patches.

`detach_encoder=True` applies only to the auxiliary MTP branch. It prevents MTP-loss gradients from flowing back through the base-model hidden states, but it does not freeze the base model: the ordinary SFT language-model loss continues to train it.

When vLLM rollout uses R3 router replay together with MTP speculation, vLLM **0.26.0 or newer** is required so replay excludes draft-model router decisions. The Nemotron image uses vLLM 0.29.0. R3 also requires `actor_rollout_ref.rollout.enable_rollout_routing_replay=True` and context parallelism 1 for the validated Lightning topology.

## 3. Experimental Results

The experiment was conducted as follows:

* model = mimo-7B-math
* max_response_length = 8k

Experiment chart:

![fully_async_policy_revenue](
https://github.com/ArronHZG/verl-community/blob/main/docs/mimo-7b-mtp.png?raw=true)

The wandb link for the graph: [wandb](https://wandb.ai/hou-zg-meituan/mimo-7b-sft-mtp?nw=nwuserhouzg)

**Scenarios with No Significant Effect**

The following configurations will not have a noticeable impact on training results:

1. The base model does not carry MTP parameters;

2. The base model carries MTP parameters, but the MTP module is not trained;

3. The base model carries MTP parameters and trains MTP, with `mtp_loss_scaling_factor=0`;

4. The base model carries MTP parameters, trains MTP and detaches the encoder, with `mtp_loss_scaling_factor=0.1`.

**Scenarios with Significant Effect**

Only the following configuration will have a noticeable impact on training results:

- The base model carries MTP parameters, MTP Loss applies to all model parameters, and `mtp_loss_scaling_factor=0.1`.

**Recommended Training Method**

It is recommended to adopt the `detach_encoder=True` approach for MTP training. This isolates auxiliary MTP gradients; it does not freeze base-model training from the main loss.

## 4. Performance Notes for MTP in Rollout Inference

Enabling MTP improves the rollout acceptance rate by around 14%. However, on H20 GPUs, overall throughput does not increase and even decreases slightly.

![spec_log](
https://github.com/ArronHZG/verl-community/blob/main/docs/spec_log.png?raw=true)

The effectiveness of MTP-accelerated Rollout is significantly affected by **model size** and **inference hardware**. Key reference information is as follows:

**Hardware Tensor Core Performance**

| Hardware Model | FP16 Performance (TFLOPS) |
|----------------|---------------------------|
| H20  | 148            |
| H800 | 1,671          |
| H200 | 1,979          |

**Measured Performance and Recommendations**

Taking the mimo-7B model deployed separately on H20 hardware using SGLang as an example: After enabling MTP speculative decoding, the Rollout throughput decreases by approximately 50%.

- Current priority recommendation: Do not enable MTP acceleration during the inference phase for now;

- Future planning: Further optimization of the speculative logic in the Rollout phase will be conducted to improve throughput performance.

## 5. SFT training

SFT training with MTP is supported, using the same MTP training configuration as RL training. The relevant prefix is `model.mtp` for SFT and `actor_rollout_ref.model.mtp` for RL.

Example launchers:

- `examples/sft/gsm8k/run_mimo_7b_mtp_megatron.sh` for MiMo-7B;

- `examples/sft/vlm/run_nemotron_3_5_super_vl_megatron.sh` for NVIDIA Nemotron 3.5 Super VL 120B-A12B image/text SFT with native Megatron-Core `HybridModel` MTP. This replaces the Nemotron 3 Super GSM8K example; see section 7 for dependencies and supported inputs.

- `examples/sft/gsm8k/run_nemotron_3_5_lightning_megatron.sh` for full-parameter NVIDIA Nemotron 3.5 Lightning SFT with native MTP.

**SFT result**

The experiment was conducted using following data:
- model = mimo-7B-math
- dataset = gsm8k

The result: [wandb link](https://wandb.ai/hou-zg-meituan/mimo-7b-sft-mtp?nw=nwuserhouzg)

The presence of mtp layer has limited effect on main loss. However, when MTP layer is detached, the mtp_loss converges to a higher value.

## 6. Nemotron 3.5 Lightning GRPO

Use `examples/grpo_trainer/run_nemotron_3_5_lightning_30b_a3b_megatron.sh` for the 2-node x 8-H100 topology. Its defaults enable R3 router replay and one-token MTP speculation, use raw rollout log probabilities, and disable dynamic batching and router/permutation fusion to keep Megatron and vLLM routing reproducible. These settings completed a four-hour validation on the v0.7 backport; run a GPU smoke test before treating a new `release/v0.9.0` backport as hardware-validated. The recipe supports vLLM only.

verl also supports generic low-precision rollout and Megatron QAT as documented in `docs/low_precision/qat.md`. Nemotron 3.5 Lightning with native HybridModel MTP, R3, and quantized weight refits has not been validated as a combined configuration; MTP layers remain in BF16. Treat rollout quantization as experimental for this recipe and do not enable QAT without a dedicated end-to-end validation.

## 7. Nemotron 3.5 Super VL SFT

Use `examples/sft/vlm/run_nemotron_3_5_super_vl_megatron.sh` with `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`. It defaults to 4 nodes × 8 GPUs, TP=2, PP=1, EP=16, CP=1, and two prediction depths sharing one physical MTP block. The vision encoder is frozen by default; the projector and language model train. Set `FREEZE_VISION_MODEL=False` to train the vision encoder as well. The tokenizer and processor must come from this checkpoint, rather than the older text-only Nano checkpoint.

Super VL needs Bridge's expanded-sequence `NemotronOmniModel`: the processor expands image placeholders, verl packs the complete sequence, and Bridge replaces image positions with projected features without changing token lengths. verl forwards image sizes and one-frame metadata, preserves Mamba sequence boundaries, and uses the nested decoder's native MTP loss path. Images with different resolutions are converted to RADIO patches before batching.

The Dockerfile in section 1 installs [Bridge revision `e97a67ac`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/e97a67acbf1b7a8ae0a0e0f26c60fa9548e19bfe) and its matching [Megatron-Core revision `89b28875`](https://github.com/NVIDIA/Megatron-LM/commit/89b28875a8d786ab41333bc60e7e8022e83e8f65), so Super VL no longer requires a manual upgrade after building the image. Released Bridge 0.6.2 does not register this model, and installing the newer Bridge alone over Core 0.19.2 is insufficient because it imports newer Core APIs. For a non-container environment, use the same source pair and runtime dependencies as the Dockerfile rather than mixing the older release packages with the newer Bridge.

The image includes Transformers 5.20.1; the launcher sets `model.trust_remote_code=True` for the checkpoint's processor. This deliberately overrides the release's `transformers<5.11` requirement for Super VL; reinstalling verl with dependency resolution can restore an incompatible Transformers version. The source pair and build-time checks establish the required interfaces, but full image and GPU training validation are still pending.

Prepare an image-caption dataset and run once on each host with a distinct `NODE_RANK`:

```bash
python examples/data_preprocess/pokemon.py \
  --local_save_dir "$HOME/data/pokemon-gpt4o-captions"

hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
  --revision 1f8953f7f0935358e3605150d22bc15df062061c \
  --local-dir "$HOME/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"

MODEL_PATH="$HOME/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16" \
MASTER_ADDR=node-0.example.internal NODE_RANK=0 \
  bash examples/sft/vlm/run_nemotron_3_5_super_vl_megatron.sh
```

For custom data, set `TRAIN_FILES` and `VAL_FILES` to parquet files containing `messages` and `images`; include `<image>` placeholders in the user messages. Text-only conversations are also accepted. The recipe currently supports image/text SFT with CP=1 and fused verl kernels disabled. Video/audio batches and CP>1 are rejected. Super VL GRPO, R3, speculative rollout, quantization, and checkpoint restore have not been validated by this recipe. Earlier Nemotron 3 Super and Lightning H100 results do not validate Super VL.
