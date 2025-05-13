#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export DATA_DIR='/mnt/afs/tanka/shihao/project/verl/data/kg_extraction'

# export WANDB_KEY=d2754e1ba5c74b981214e7b00e7569a92627ec36
# wandb login --relogin $WANDB_KEY

WAND_PROJECT='kg_extraction_RL'

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-em
export BASE_MODEL='/mnt/afs/tanka/shihao/model/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME=RL_with_Qwen2.5-7B-Instruct

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = max_start_length + max_response_length * (max_turns - 1) + max_obs_length * max_turns

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_kg \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.fsdp_config.param_offload=Fasle \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=Fasle \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.prompt_length=6656 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_mini_batch_size=256 \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=65572 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console', "wandb"] \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=700 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    2>&1 | tee "kg_extraction_$(date +%Y%m%d_%H%M%S).log"
