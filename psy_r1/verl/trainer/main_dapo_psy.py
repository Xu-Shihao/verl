#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAPO训练器 - 心理诊断任务专用版本
基于原始main_dapo.py，集成心理诊断专用奖励函数
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

# 导入DAPO Ray训练器
from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

# 导入心理诊断奖励函数
from psy_r1.verl.utils.dapo_reward_score_psy import register_psy_reward_function


@hydra.main(config_path="../../../recipe/dapo/config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
            num_cpus=config.ray_init.num_cpus,
        )

    if (
        is_cuda_available
        and OmegaConf.select(config.trainer, "profile_steps") is not None
        and len(OmegaConf.select(config.trainer, "profile_steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print("=== 心理诊断DAPO任务配置 ===")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # reward model setup
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 创建专门的心理诊断奖励管理器
        print("=== 初始化心理诊断奖励函数 ===")
        reward_fn = create_psy_reward_manager(
            config,
            tokenizer,
            0,  # num_examine for training
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = create_psy_reward_manager(
            config,
            tokenizer,
            1,  # num_examine for validation  
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        
        print("=== 奖励函数初始化完成 ===")
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def create_psy_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """
    创建心理诊断专用的奖励管理器
    """
    from verl.workers.reward_manager import get_reward_manager_cls
    
    # 获取奖励管理器类
    reward_manager_cls = get_reward_manager_cls(config.reward_model.reward_manager)
    
    # 检查是否启用症状奖励
    use_symptom_reward = config.reward_model.get("use_symptom_reward", False)
    symptom_alpha = config.reward_model.get("symptom_alpha", 0.1)
    
    print(f"使用心理诊断专用奖励函数: register_psy_reward_function")
    print(f"症状奖励设置: use_symptom_reward={use_symptom_reward}, symptom_alpha={symptom_alpha}")
    
    # 创建包装函数，传递症状奖励参数
    def psy_compute_score_wrapper(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
        return register_psy_reward_function(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            use_symptom_reward=use_symptom_reward,
            symptom_alpha=symptom_alpha,
            **kwargs
        )
    
    # 创建奖励管理器实例，使用我们的包装函数
    reward_manager = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=psy_compute_score_wrapper,
        reward_fn_key=config.data.get("reward_fn_key", "data_source"),
        **reward_kwargs,
    )
    
    print(f"心理诊断奖励管理器初始化完成: {reward_manager_cls.__name__}")
    return reward_manager


if __name__ == "__main__":
    main()
