# 启用症状识别奖励功能的训练脚本
# 基于原有脚本，新增症状识别奖励参数

# 设置环境proxy
export https_proxy=http://10.119.16.227:7890 http_proxy=http://10.119.16.227:7890 all_proxy=socks5://10.119.16.227:7890

wandb login --relogin $WANDB_API_KEY

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置日志路径
set -x

# 创建日志路径
LOG_DIR="/tcci_mnt/shihao/project/verl/psy_r1/logs"
mkdir -p $LOG_DIR

# 读取现在时间
NOW=$(date +%Y%m%d_%H%M%S)

HOME="/tcci_mnt/shihao/project/verl"
MODEL_PATH="/tcci_mnt/shihao/outputs/dataset_v1/qwen3-8B_auxiliary_diagnosis_lora_sft_horizon-alpha_v3"
MODEL_BASE_NAME="qwen3-8B"

SYMPTOM_ALPHA=0.1
MAX_RESPONSE_LENGTH=4096
EXPERIMENT_NAME="${MODEL_BASE_NAME}_symptoms_reasoning_with_symptom_reward_lora_sft_ds-r1_8_gpu_${MAX_RESPONSE_LENGTH}_symptom_alpha-${SYMPTOM_ALPHA}_Rollout-5"


HYDRA_FULL_ERROR=1 && python3 -m psy_r1.verl.trainer.main_ppo_psy \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/psy_r1/SMHC_data_v4/train.parquet \
    data.val_files=$HOME/psy_r1/SMHC_data_v4/val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=6144 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=12240 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0.1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='SMHC_diagnosis_with_reasoning_RL' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=$LOG_DIR/rollout_data \
    trainer.validation_data_dir=$LOG_DIR/validation_data \
    reward_model.use_psy_reward=True \
    +reward_model.use_symptom_reward=True \
    +reward_model.symptom_alpha=$SYMPTOM_ALPHA \
    reward_model.show_training_examples=True \
    reward_model.show_val_examples=True \
    $@ 2>&1 | tee $LOG_DIR/$EXPERIMENT_NAME-$NOW.log