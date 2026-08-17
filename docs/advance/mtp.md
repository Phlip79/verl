# NVIDIA Nemotron 3.5 Lightning MTP

This v0.7.0 backport supports NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16
through Megatron-Core's native `HybridModel` multi-token-prediction (MTP)
implementation. It is deliberately native-only: use the Megatron training
engine with Megatron-Bridge and set both `mtp.enable=True` and
`mtp.enable_train=True`. Loading Lightning MTP weights without MTP training,
or routing it through verl's legacy GPT MTP patch, is unsupported.

Use the exact environment in
`docker/Dockerfile.nemotron_3_5_lightning`; build it with:

```bash
docker build \
  --build-arg VERL_COMMIT=$(git rev-parse HEAD) \
  -f docker/Dockerfile.nemotron_3_5_lightning \
  -t verl-nemotron-3-5-lightning .
```

The image pins the validation-critical stack: NGC PyTorch 26.06, Transformer
Engine `e7c550c5f80636cf841a8204b1d6f85a5f3f28b7`, Megatron-Bridge
`c93251151adeeadbae3ff2a2bf5ee7a1c34cff01` with Megatron-Core
`cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54`, vLLM
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, and Transformers 5.10.4.
Do not substitute a release with the same nominal version.

Launchers:

- `examples/sft/gsm8k/run_nemotron_3_5_lightning_megatron.sh` runs packed,
  native-MTP GSM8K SFT. The published topology is 2 nodes × 8 H100 GPUs,
  TP=2 and EP=8.
- `examples/grpo_trainer/run_nemotron_3_5_lightning_30b_a3b_megatron.sh`
  runs experimental GRPO using a Megatron actor and vLLM rollout. It forces
  `trainer.use_legacy_worker_impl=disable`, which is required on v0.7.0 for
  the native HybridModel path. It defaults to R3 router replay so Megatron
  reuses the expert routes captured by vLLM while recomputing actor log
  probabilities and updating the actor.

The public checkpoint revision used by the upstream hardware validation is
`d468880b6ad3c6e0d21377ce7242adaea4cc884d`; both launchers select it through
`MODEL_REVISION` when `MODEL_PATH` is a remote Hugging Face repository. A local
`MODEL_PATH` is used as-is.

Prepare the immutable data inputs with:

```bash
python examples/data_preprocess/gsm8k_multiturn_sft.py \
  --revision 740312add88f781978c0658806c59bc2815b9866 \
  --local_save_dir "$HOME/data/gsm8k_sft"

DATA_ROOT=/shared/data
hf download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset \
  --revision 65877096c24ffa7abc4e4fa5edb95cf3413a5674 \
  --local-dir "$DATA_ROOT/DAPO-Math-17k"
hf download BytedTsinghua-SIA/AIME-2024 --repo-type dataset \
  --revision aa49075e24ad594b79fdf0bdcefa735c2181be67 \
  --local-dir "$DATA_ROOT/AIME-2024"
```

For GRPO, pass
`TRAIN_FILES=$DATA_ROOT/DAPO-Math-17k/data/dapo-math-17k.parquet` and
`VAL_FILES=$DATA_ROOT/AIME-2024/data/aime-2024.parquet` to the launcher.

The GRPO launcher defaults to ordinary BF16 rollout (`MTP_ROLLOUT_SPEC=0`) and
R3 router replay (`ROUTER_REPLAY_MODE=R3`). Because this launcher uses the V1
engine, the R3 setting is applied at
`actor_rollout_ref.actor.megatron.router_replay.mode`; the similarly named
legacy actor setting does not enable replay on this path. Rollout route capture
is enabled at the same time.

The parity-oriented defaults use raw vLLM log probabilities and require
temperature 1.0, top-p 1.0, repetition penalty 1.0, fixed microbatches,
disabled trainer batch balancing, and Megatron router FP32. Both router and
permutation fusion are disabled; in particular, the pinned MCore fused router
bypasses replay. The launcher also enables vLLM's H100 batch-invariant kernels
and sets
`CUDA_DEVICE_MAX_CONNECTIONS=1`. These settings reduce
avoidable differences while comparing vLLM rollout log probabilities with
Megatron recomputation; overriding the sampling settings changes that parity
boundary.

R3 is currently restricted to `ACTOR_CP=1` and
`virtual_pipeline_model_parallel_size=null`. R3 and MTP speculative rollout
cannot be enabled together, so the launcher and engine exit early for
unsupported combinations. To experiment with one-token MTP rollout speculation, use
`ROUTER_REPLAY_MODE=disabled MTP_ROLLOUT_SPEC=1`; that path is outside the
validated baseline.

The reported hardware boundary is a two-hour, 37-step H100 soak for ordinary
BF16 rollout without R3. It does not validate the new R3 path or establish
convergence, resume, quantized weight synchronization, alternative topologies,
or speculative rollout correctness.
