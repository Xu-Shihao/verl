python ./scripts/legacy_model_merger.py merge --backend fsdp \
    --hf_model_path '/mnt/tcci/shihao/outputs/dataset_v2/qwen3-8B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7_lr1e-6' \
    --local_dir '/mnt/tcci/shihao/project/verl/checkpoints/SMHC_ICD_recommendation_RL/grpo_qwen3-8B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7-1_lr1e-6_icd_recommendation_with_real_data/global_step_1750/actor' \
    --target_dir '/mnt/tcci/shihao/outputs/dataset_v2/grpo_qwen3-8B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7-1_lr1e-6_with_ICD_recommendation_with_real_data'