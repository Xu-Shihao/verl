python ./scripts/legacy_model_merger.py merge --backend fsdp \
    --hf_model_path '/tcci_mnt/shihao/outputs/dataset_v1/qwen3-8B_auxiliary_diagnosis_full_sft_ds-r1_v2' \
    --local_dir '/tcci_mnt/shihao/project/verl/checkpoints/SMHC_diagnosis_with_reasoning_RL/dapo_qwen3-8B_reasoning_sft_ds-r1_v2_8_gpu/global_step_500/actor' \
    --target_dir '/tcci_mnt/shihao/outputs/dataset_v1/qwen3-8B_auxiliary_diagnosis_full_sft_ds-r1_with_DAPO'