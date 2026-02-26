python ./scripts/legacy_model_merger.py merge --backend fsdp \
    --hf_model_path '/tcci_mnt/shihao/outputs/dataset_v2/qwen3-32B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7_lr1e-6_0113' \
    --local_dir '/tcci_mnt/shihao/project/verl/checkpoints/SMHC_ICD_recommendation_RL/grpo_qwen3-32B-sft_auxiliary_diagnosis_v7_lr1e-6_tp2_full_icd_recommendation/global_step_1550/actor' \
    --target_dir '/tcci_mnt/shihao/outputs/dataset_v2/grpo_qwen3-32B_sft_auxiliary_diagnosis_v7_lr1e-6_icd_recommendation'