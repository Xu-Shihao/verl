python ./scripts/legacy_model_merger.py merge --backend fsdp \
    --hf_model_path '/tcci_mnt/shihao/outputs/dataset_v2/qwen3-8B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7_lr1e-6' \
    --local_dir '/tcci_mnt/shihao/project/verl/checkpoints/SMHC_v8_mixed_RL/grpo_qwen3-8B-sft_v7_v8_mixed/global_step_3580/actor' \
    --target_dir '/tcci_mnt/shihao/outputs/dataset_v2/grpo_qwen3-8B_sft_auxiliary_diagnosis_v8_mixed_tasks'