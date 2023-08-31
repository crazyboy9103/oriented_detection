python /workspace/wandb_sweep.py \
    --model_type rotated \
    --dataset mvtec \
    --project_name rfrcnn-sweep \
    --experiment_name rfrcnn-sweep-512 \
    --image_size 512 \
    --data_path /datasets/split_ss_mvtec \
    --data_pth /workspace/datasets/mvtec.pth \
    --sweep_name rfrcnn-sweep