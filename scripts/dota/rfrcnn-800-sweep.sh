python /workspace/wandb_sweep.py \
    --model_type rotated \
    --dataset dota \
    --project_name rfrcnn-sweep \
    --experiment_name rfrcnn-sweep-800 \
    --image_size 800 \
    --data_path /datasets/split_ss_dota_800 \
    --data_pth /workspace/datasets/dota_800.pth \
    --sweep_name rfrcnn-sweep