python /workspace/wandb_sweep.py \
    --model_type rotated \
    --dataset dota \
    --project_name rfrcnn-sweep \
    --experiment_name rfrcnn-sweep-256 \
    --image_size 256 \
    --data_path /datasets/split_ss_dota_256 \
    --data_pth /workspace/datasets/dota_256.pth \
    --sweep_name rfrcnn-sweep