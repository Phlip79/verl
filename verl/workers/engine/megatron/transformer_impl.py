# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from functools import partial
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.models.mcore import get_mcore_weight_converter
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    align_r3_router_replay_data,
    build_r3_replay_mask,
    iter_model_routers,
    set_model_router_replay_action,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import (
    vocab_parallel_entropy,
    vocab_parallel_log_probs_from_logits,
)
from verl.utils.megatron_utils import (
    check_mtp_config,
    get_megatron_mtp_loss,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    patch_engine_mtp,
    register_megatron_training_hooks,
    unwrap_model,
)
from verl.utils.model import (
    extract_multi_modal_inputs,
    load_mcore_dist_weights,
)
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import (
    postprocess_batch_func,
    prepare_micro_batches,
)
from .utils import set_random_seed

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronEngine(BaseEngine):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.use_mbridge, "use_mbridge must be True"
        self._init_device_mesh()

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_grad = self.engine_config.grad_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.mode = None

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None

        self.enable_routing_replay = self.engine_config.router_replay.mode != "disabled"
        if self.enable_routing_replay and self.engine_config.router_replay.mode != "R3":
            raise NotImplementedError(
                "The v0.7.0 V1 Megatron backport supports router_replay.mode=R3 only; use the legacy worker for R2."
            )
        if self.enable_routing_replay and self.engine_config.virtual_pipeline_model_parallel_size is not None:
            raise NotImplementedError(
                "R3 router replay for sparse-MoE models on the v0.7.0 V1 engine requires "
                "virtual_pipeline_model_parallel_size=null"
            )
        logger.info("enable_routing_replay in MegatronEngine: %s", self.enable_routing_replay)
        if self.enable_routing_replay:
            apply_router_replay_patch()

    def _init_device_mesh(self):
        # TODO: set different parallelism for actor, critic, ref
        if mpu.is_initialized():
            return

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.engine_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )

    def _build_tf_config(self):
        from verl.utils.megatron_utils import mapping_string_to_attn_backend
        from verl.utils.torch_dtypes import PrecisionType

        check_mtp_config(self.model_config, self.engine_config)

        self.param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        override_transformer_config = mapping_string_to_attn_backend({**self.engine_config.override_transformer_config})
        if self.enable_routing_replay:
            if override_transformer_config.get("moe_router_fusion", False):
                raise ValueError(
                    "R3 router replay requires moe_router_fusion=False because the fused router bypasses replay"
                )
            # Keep replay safe for callers that do not use the example launcher's
            # explicit parity override. This is also the pinned MCore default.
            override_transformer_config["moe_router_fusion"] = False

        self.provider = None
        self.vanilla_bridge = self.engine_config.vanilla_mbridge
        if self.vanilla_bridge:
            if self.model_config.mtp.enable:
                raise ValueError(
                    "Native HybridModel MTP requires engine.vanilla_mbridge=False and the pinned Megatron-Bridge."
                )
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            tf_config.fp16 = self.param_dtype == torch.float16
            tf_config.bf16 = self.param_dtype == torch.bfloat16
        else:
            from verl.models.mcore.bridge import AutoBridge

            # Use Megatron-Bridge to convert HF config to Megatron config
            bridge = AutoBridge.from_hf_pretrained(
                self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
            )
            # Get Megatron provider and configure it
            provider = bridge.to_megatron_provider(load_weights=False)

            # Match verl implementation (need variable_seq_lengths)
            from megatron.core.transformer.enums import AttnBackend

            virtual_pipeline_size = self.engine_config.virtual_pipeline_model_parallel_size
            provider_overrides = {
                "tensor_model_parallel_size": self.engine_config.tensor_model_parallel_size,
                "pipeline_model_parallel_size": self.engine_config.pipeline_model_parallel_size,
                "expert_model_parallel_size": self.engine_config.expert_model_parallel_size,
                "expert_tensor_parallel_size": self.engine_config.expert_tensor_parallel_size,
                "virtual_pipeline_model_parallel_size": virtual_pipeline_size,
                "context_parallel_size": self.engine_config.context_parallel_size,
                "sequence_parallel": self.engine_config.sequence_parallel,
                "overlap_p2p_comm": virtual_pipeline_size is not None and virtual_pipeline_size > 1,
                "batch_p2p_comm": False,
                "attention_backend": AttnBackend.flash,
                "variable_seq_lengths": True,
                "moe_token_dispatcher_type": "alltoall",
                "moe_router_load_balancing_type": "none",
                **override_transformer_config,
            }

            if self.enable_routing_replay:
                if hasattr(provider, "moe_enable_routing_replay"):
                    provider_overrides["moe_enable_routing_replay"] = True
                else:
                    provider_overrides["enable_routing_replay"] = True

            from verl.models.mcore.mtp_support import configure_native_hybrid_mtp

            configure_native_hybrid_mtp(provider, self.model_config.mtp, provider_overrides)

            if hasattr(provider, "apply_overrides_and_finalize"):
                provider.apply_overrides_and_finalize(dtype=self.param_dtype, overrides=provider_overrides)
            else:
                provider.params_dtype = self.param_dtype
                provider.fp16 = self.param_dtype == torch.float16
                provider.bf16 = self.param_dtype == torch.bfloat16
                for name, value in provider_overrides.items():
                    setattr(provider, name, value)
                provider.finalize()
            self.provider = provider
            tf_config = None  # Will be set after model creation
        self.bridge = bridge

        if self.enable_routing_replay and tf_config is not None:
            if hasattr(tf_config, "moe_enable_routing_replay"):
                tf_config.moe_enable_routing_replay = True
            else:
                tf_config.enable_routing_replay = True

        if not self.bridge:
            self.weight_converter = get_mcore_weight_converter(self.model_config.hf_config, self.dtype)

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.tf_config = tf_config

        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.model_config, bridge=self.bridge, provider=self.provider, dtype=self.param_dtype
        )

    def _build_megatron_module(self):
        from verl.utils.megatron_utils import (
            McoreModuleWrapperConfig,
            make_megatron_module,
        )
        from verl.utils.model import print_model_size

        # TODO: add more cases
        is_value_model = (
            "ForTokenClassification" in self.model_config.architectures[0]
            or "ForSequenceClassification" in self.model_config.architectures[0]
        )

        self.is_value_model = is_value_model

        if self.engine_config.forward_only:
            wrap_with_ddp = False
        else:
            wrap_with_ddp = True

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=is_value_model,  # actor is not value model
            share_embeddings_and_output_weights=self.model_config.share_embeddings_and_output_weights,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
        )
        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.model_config.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=self.engine_config.override_mcore_model_config,
            override_ddp_config=self.engine_config.override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.model_config.get("lora", None),
        )
        self.tf_config = updated_tf_config
        print(f"module: {len(module)}")

        if self.engine_config.use_dist_checkpointing:
            load_mcore_dist_weights(module, self.engine_config.dist_checkpointing_path, is_value_model=is_value_model)
        else:
            if self.vanilla_bridge:
                self.bridge.load_weights(module, self.model_config.local_path)
            else:
                allowed_mismatched_params = []
                if self.is_value_model:
                    allowed_mismatched_params = ["output_layer.weight"]
                self.bridge.load_hf_weights(
                    module, self.model_config.local_path, allowed_mismatched_params=allowed_mismatched_params
                )

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])
            if self.enable_routing_replay:
                live_router_count = sum(1 for _ in iter_model_routers(unwrap_model(module)))
                print(f"router replay live decoder layers: {live_router_count}")

        if self.enable_routing_replay and self.engine_config.pipeline_model_parallel_size == 1:
            live_router_count = sum(1 for _ in iter_model_routers(unwrap_model(module)))
            if live_router_count == 0:
                raise RuntimeError(
                    "R3 router replay is enabled, but Megatron built no replay-enabled live decoder routers"
                )

        return module

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            init_megatron_optim_config,
        )

        optim_config_megatron = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
        )
        optimizer = get_megatron_optimizer(model=self.module, config=optim_config_megatron)
        register_megatron_training_hooks(self.module, optimizer)
        return optimizer

    def _build_lr_scheduler(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer_param_scheduler

        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer, config=self.optimizer_config
        )
        return optimizer_scheduler

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )

    def initialize(self):
        self._build_tf_config()

        self.module = self._build_megatron_module()

        if self.model_config.mtp.enable:
            patch_engine_mtp(self.module, self.model_config)

        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
            return

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        tmp_config = OmegaConf.create({"model": {"path": self.model_config.local_path}})

        role = "actor" if not self.is_value_model else "critic"

        self.checkpoint_mananager = MegatronCheckpointManager(
            config=tmp_config,
            checkpoint_config=self.checkpoint_config,
            model_config=self.model_config.hf_config,
            transformer_config=self.tf_config,
            role=role,
            model=self.module,
            arch=self.model_config.architectures[0],
            hf_config=self.model_config.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.model_config.share_embeddings_and_output_weights,
            processing_class=self.model_config.get_processor(),
            optimizer=self.optimizer,
            optimizer_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.optimizer_config.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            provider=self.provider,
            use_dist_checkpointing=self.engine_config.use_dist_checkpointing,
            peft_cls=self.peft_cls,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def train_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        return EngineEvalModeCtx(self, **kwargs)

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        self.optimizer.zero_grad()
        # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        for chunk in self.module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError("Megatron optimizer step failed. This should not happen")

        return grad_norm

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        self.lr_scheduler.step(1)
        return get_megatron_last_lr(self.optimizer)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_megatron_model_to_gpu(self.module, load_grad=grad)
            if optimizer and self.optimizer is not None:
                load_megatron_optimizer(self.optimizer)
        elif device == "cpu":
            if model:
                offload_megatron_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_megatron_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def get_data_parallel_rank(self):
        return mpu.get_data_parallel_rank()

    def get_data_parallel_size(self):
        return mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self):
        return mpu.get_data_parallel_group()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module, load_grad=True)
        self.checkpoint_mananager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

    @staticmethod
    def _routed_num_tokens(data: TensorDict) -> torch.Tensor:
        """Count real full-sequence tokens fed through the MoE router."""

        attention_mask = data.get("attention_mask")
        if attention_mask is not None:
            return attention_mask.sum()
        input_ids = data["input_ids"]
        if input_ids.is_nested:
            return input_ids.offsets()[-1]
        return torch.tensor(input_ids.numel(), device=input_ids.device)

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)

        # compute num_tokens in global batch for loss normalization
        loss_mask = data["loss_mask"]
        batch_num_tokens = (loss_mask.values().sum() if loss_mask.is_nested else loss_mask.sum()).to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        if self.tf_config is not None and self.tf_config.calculate_per_token_loss:
            routed_num_tokens = self._routed_num_tokens(data).to(get_device_id())
            torch.distributed.all_reduce(
                routed_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
            )
            tu.assign_non_tensor(data, routed_num_tokens=routed_num_tokens.item())

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            num_batches_divided_by = self.tf_config.microbatch_group_size_per_vp_stage
        else:
            num_batches_divided_by = None

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=True,
            min_num_micro_batch=None,
        )

        if num_batches_divided_by is not None:
            assert len(micro_batches) % num_batches_divided_by == 0, (
                f"micro_batches {micro_batches} must be divisible by num_batches_divided_by "
                f"{num_batches_divided_by} for megatron backend"
            )

        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)

        forward_backward_func = get_forward_backward_func()

        postprocess_micro_batch_func = partial(
            self.postprocess_micro_batch_func,
            forward_only=forward_only,
            loss_function=loss_function,
        )

        tu.assign_non_tensor(data, num_micro_batch=n_micro_batch)

        forward_step = partial(self.forward_step, postprocess_micro_batch_func=postprocess_micro_batch_func)

        enable_routing_replay = tu.get_non_tensor_data(data, key="enable_routing_replay", default=False)
        if enable_routing_replay:
            if not self.enable_routing_replay:
                raise RuntimeError("The batch requested R3 router replay, but this Megatron engine has it disabled")
            if data.get("routed_experts") is None:
                raise RuntimeError(
                    "R3 router replay was requested, but the batch has no routed_experts. "
                    "Set rollout.enable_rollout_routing_replay=True and use a supported vLLM version."
                )
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        try:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.module,
                num_microbatches=n_micro_batch,
                seq_length=1,  # the communication shape is obtained via p2p comm
                micro_batch_size=1,  # the communication shape is obtained via p2p comm
                forward_only=forward_only,
            )
        finally:
            if enable_routing_replay:
                # Never leak targets/actions into the next batch, including when
                # the pipeline schedule raises partway through a forward.
                RouterReplay.clear_global_indices()
                RouterReplay.clear_global_router_replay_action()

        if self.model_config.mtp.enable and mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Every CP rank participates because MCore's tracker reduces over
            # the combined DP+CP averaging group. Only the output source rank
            # attaches the already-global metric to the returned batch.
            mtp_metrics = get_megatron_mtp_loss(n_micro_batch)
            if self.is_mp_src_rank_with_outputs() and losses_reduced:
                losses_reduced[0].setdefault("metrics", {}).update(mtp_metrics)
        # loss_reduces contains the stats returned from loss_func
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            return postprocess_batch_func(output_lst=losses_reduced, indices=indices, data=data)
        else:
            return {}

    def get_per_tensor_param(self, **kwargs):
        load_megatron_model_to_gpu(self.module, load_grad=False)
        if self.vanilla_bridge:
            per_tensor_param = self.bridge.export_weights(self.module)
        else:
            # Megatron-Bridge folds active adapters into the exported Hugging
            # Face tensors consumed by rollout weight synchronization.
            per_tensor_param = self.bridge.export_hf_weights(self.module)
        return per_tensor_param, None

    def forward_step(self, batch_iter, model, postprocess_micro_batch_func):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        raise NotImplementedError("postprocess_micro_batch_func must be implemented in subclass")


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    def prepare_model_inputs(self, batch: TensorDict):
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].to(bool)
        multi_modal_inputs = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []))

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "attention_mask": batch.get("attention_mask"),
            "multi_modal_inputs": multi_modal_inputs,
            "routed_experts": batch.get("routed_experts"),
        }

    def prepare_model_outputs(self, output: dict, data: TensorDict):
        calculate_entropy = tu.get_non_tensor_data(data, key="calculate_entropy", default=False)

        log_prob = output["log_probs"]
        model_output = {"log_probs": log_prob}
        if calculate_entropy:
            entropy = output["entropy"]
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, batch_iter: Iterator[TensorDict], model, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = batch["temperature"]

        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        loss_mask = model_inputs["loss_mask"]
        attention_mask = model_inputs["attention_mask"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        unwrapped_model = unwrap_model(model)
        vp_rank = unwrapped_model.vp_stage if hasattr(unwrapped_model, "vp_stage") else 0

        if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
            router_instances = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instances:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            set_model_router_replay_action(unwrapped_model, RouterReplayAction.REPLAY_FORWARD)

        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            layers_topk_idx = model_inputs["routed_experts"]
            if layers_topk_idx is None:
                raise RuntimeError("R3 router replay is active, but this micro-batch has no routed_experts")
            layers_topk_idx = align_r3_router_replay_data(layers_topk_idx, input_ids)
            replay_mask = build_r3_replay_mask(input_ids, batch["response_mask"])
            set_router_replay_data(
                layers_topk_idx,
                None,
                self.tf_config,
                vp_rank,
                replay_mask=replay_mask,
                model=unwrapped_model,
            )

        if pad_mode == DatasetPadMode.NO_PADDING:
            label = input_ids.clone()
        else:
            raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

        from verl.models.mcore import get_mcore_forward_no_padding_fn

        if use_fused_kernels:
            raise NotImplementedError("Fused kernels are not supported for megatron engine")

        forward_fn = get_mcore_forward_no_padding_fn(self.model_config.hf_config)

        def logits_processor(logits, label):
            assert logits.shape[:2] == label.shape[:2]
            logits.div_(temperature)
            ret = {}
            if calculate_entropy:
                logits_bak = logits.clone()
                # # disable the hint until the fused_kernel is optimized for triton>=3.3
                # if torch.distributed.get_rank() == 0:
                #     logger.warning_once(
                #         "For memory-efficient computation, enable fused kernels via "
                #         "`actor_rollout_ref.model.use_fused_kernels=True`. "
                #         "The current `clone()` operation ensures correctness but increases memory usage."
                #     )
                entropy = vocab_parallel_entropy(logits)
                ret["entropy"] = entropy
            else:
                logits_bak = logits

            log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
            ret["log_probs"] = log_probs
            return ret

        response_attention_mask = None
        if attention_mask is not None and not loss_mask.is_nested:
            response_attention_mask = attention_mask[:, -loss_mask.shape[-1] :]
        logits_processor_args = {
            "label": label,
            "loss_mask": loss_mask,
            "response_attention_mask": response_attention_mask,
        }

        output = forward_fn(
            model,
            input_ids,
            multi_modal_inputs,
            logits_processor=logits_processor,
            logits_processor_args=logits_processor_args,
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            data_format="thd" if self.engine_config.use_remove_padding else "bshd",
            mtp_enable_train=self.model_config.mtp.enable and self.model_config.mtp.enable_train,
        )

        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            router_instances = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instances:
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
            set_model_router_replay_action(unwrapped_model, RouterReplayAction.REPLAY_BACKWARD)

        return output, partial(postprocess_micro_batch_func, data=batch)

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        # For memory efficiency
        # We move calculation of entropy to compute_log_probs, forward_only == True
        device = data["input_ids"].device
        model_output = self.prepare_model_outputs(output, data)

        if loss_function is not None:
            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
            # scale loss by num_micro_batch because megatron will scale loss
            # by n_micro_batch inside pp schedule
            scaled_loss = loss * data["num_micro_batch"]
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device)
            scaled_loss = loss
            metrics = {}

        output = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }

        if self.tf_config is not None and self.tf_config.calculate_per_token_loss and loss_function is not None:
            if not self.engine_config.use_remove_padding:
                raise ValueError(
                    "calculate_per_token_loss=True requires engine.use_remove_padding=True in this backport"
                )
            cp_size = self.engine_config.context_parallel_size
            local_num_tokens = (self._routed_num_tokens(data) // cp_size).to(torch.int)
            routed_num_tokens = tu.get_non_tensor_data(data, key="routed_num_tokens", default=None)
            if routed_num_tokens is None:
                routed_num_tokens = self._routed_num_tokens(data)
            dp_size = tu.get_non_tensor_data(data, key="dp_size", default=1)
            local_loss_sum = loss * routed_num_tokens / dp_size
            return local_loss_sum, local_num_tokens, output

        # return loss and stats
        return scaled_loss, output


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngineWithLMHead):
    # for value head
    def forward_step(self, batch_iter, model, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        from verl.models.mcore import get_mcore_forward_no_padding_fn

        forward_fn = get_mcore_forward_no_padding_fn(self.model_config.hf_config)

        output = forward_fn(
            model,
            input_ids,
            multi_modal_inputs,
            value_model=True,
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
        )

        return output, partial(postprocess_micro_batch_func, data=batch)

    def prepare_model_outputs(self, output: dict | torch.Tensor, data: TensorDict):
        return {"values": output}
