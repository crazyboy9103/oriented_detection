TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python /workspace/wandb_sweep.py \
    --model_type oriented \
    --dataset mvtec \
    --project_name orcnn-sweep \
    --experiment_name orcnn-sweep-256 \
    --image_size 256 \
    --data_path /datasets/split_ss_mvtec \
    --data_pth /workspace/datasets/mvtec.pth \
    --sweep_name orcnn-sweep