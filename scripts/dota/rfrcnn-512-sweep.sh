python /workspace/wandb_sweep.py \
    --model_type rotated \
    --dataset dota \
    --project_name rfrcnn-sweep \
    --experiment_name rfrcnn-sweep-512 \
    --image_size 512 \
    --data_path /datasets/split_ss_dota_512 \
    --data_pth /workspace/datasets/dota_512.pth \
    --sweep_name rfrcnn-sweep