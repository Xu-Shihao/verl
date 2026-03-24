python ./scripts/legacy_model_merger.py merge --backend fsdp \
    --hf_model_path '/tcci_mnt/shihao/outputs/dataset_v2/qwen3-8B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7_lr1e-6' \
    --local_dir '/tcci_mnt/shihao/project/verl/checkpoints/SMHC_ICD_recommendation_RL/grpo_qwen3-8B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7-1_lr1e-6_only_real_data_filtered_icd_recommendation_only_real_data/global_step_240/actor' \
    --target_dir '/tcci_mnt/shihao/outputs/dataset_v2/EverDiagnosis-8B_icd-code-prediction_kimi-k2-0905-cot_grpo_only-real-data_filtered'