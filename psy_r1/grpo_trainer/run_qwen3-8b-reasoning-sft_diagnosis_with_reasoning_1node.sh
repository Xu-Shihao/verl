# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

# 设置环境proxy
export https_proxy=http://10.119.16.227:7890 http_proxy=http://10.119.16.227:7890 all_proxy=socks5://10.119.16.227:7890

export WANDB_API_KEY=d8e131b9817bc59353326755d6db8b705a4d8d4d

wandb login --relogin d8e131b9817bc59353326755d6db8b705a4d8d4d

# 设置日志路径
set -x

# 创建日志路径
LOG_DIR="/tcci_mnt/shihao/project/verl/psy_r1/logs"
mkdir -p $LOG_DIR

# 读取现在时间
NOW=$(date +%Y%m%d_%H%M%S)

HOME="/tcci_mnt/shihao/project/verl"

HYDRA_FULL_ERROR=1 && python3 -m psy_r1.verl.trainer.main_ppo_psy \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/psy_r1/SMHC_data_v4/train.parquet \
    data.val_files=$HOME/psy_r1/SMHC_data_v4/val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/tcci_mnt/shihao/outputs/qwen3-8B_auxiliary_diagnosis_sft_reasoning_ds-r1_v1_lr1e-4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0.1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='SMHC_diagnosis_with_reasoning_RL' \
    trainer.experiment_name='qwen3_8B_reasoning_sft_8_gpu' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False \
    reward_model.use_psy_reward=True \
    reward_model.show_training_examples=True \
    reward_model.show_val_examples=True \
    $@ 2>&1 | tee $LOG_DIR/qwen3_8B_sft_8_gpu_$NOW.log