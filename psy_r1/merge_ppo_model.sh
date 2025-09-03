python ./scripts/legacy_model_merger.py merge --backend fsdp \
    --hf_model_path '/mnt/tcci/shihao/models/Qwen3-8B' \
    --local_dir '/mnt/tcci/shihao/project/verl/outputs/actor' \
    --target_dir '/mnt/tcci/shihao/outputs/qwen3-8B-lora-sft-ds-r1_v2_auxiliary_diagnosis_grpo'