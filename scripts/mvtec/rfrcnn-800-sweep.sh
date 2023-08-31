python /workspace/wandb_sweep.py \
    --model_type rotated \
    --dataset mvtec \
    --project_name rfrcnn-sweep \
    --experiment_name rfrcnn-sweep-800 \
    --image_size 800 \
    --data_path /datasets/split_ss_mvtec \
    --data_pth /workspace/datasets/mvtec.pth \
    --sweep_name rfrcnn-sweep