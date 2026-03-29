#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v8 GRPO trainer - 混合RL训练 (binary + multiclass + recommendation)
基于上游 main_ppo.py 的 TaskRunner，仅注入 v8 混合奖励函数。
"""

import os
import socket

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner, run_ppo, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

# 导入 v8 混合奖励函数
from psy_r1.verl.utils.v8_mixed_reward import create_v8_reward_fn


class PsyV8TaskRunner(BaseTaskRunner):
    """继承上游 TaskRunner，仅 override run() 注入 v8 混合奖励逻辑。"""

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print("=== v8 混合RL GRPO训练 (binary + multiclass + recommendation) ===")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # --- 使用上游 TaskRunner 的方法来设置 workers ---
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        from verl.trainer.ppo.utils import need_critic, need_reference_policy

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Download checkpoint
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # --- v8 混合奖励函数 ---
        print("=== 初始化 v8 混合RL奖励函数 ===")

        psy_cfg = config.get("psy", {})
        show_train_examples = psy_cfg.get("show_training_examples", True)
        show_val_examples = psy_cfg.get("show_val_examples", True)

        train_log_mode = "train" if show_train_examples else None
        val_log_mode = "val" if show_val_examples else None

        reward_fn = create_v8_reward_fn(is_validation=train_log_mode, tokenizer=tokenizer)
        val_reward_fn = create_v8_reward_fn(is_validation=val_log_mode, tokenizer=tokenizer)

        print("v8 混合奖励: binary(抑郁/焦虑) + multiclass(4分类) + recommendation(ICD-10)")
        print("奖励公式: format_score * 0.2 + format_score * exact_match * 0.8")
        print("=== v8 奖励函数初始化完成 ===")

        # --- 数据集 ---
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # --- 初始化 trainer (注入 v8 reward) ---
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="ppo_trainer_psy", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    # 使用自定义 TaskRunner 运行 PPO
    task_runner_class = ray.remote(num_cpus=1)(PsyV8TaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
