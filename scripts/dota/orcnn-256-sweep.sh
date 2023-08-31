python /workspace/wandb_sweep.py \
    --model_type oriented \
    --dataset dota \
    --project_name orcnn-sweep \
    --experiment_name orcnn-sweep-256 \
    --image_size 256 \
    --data_path /datasets/split_ss_dota_256 \
    --data_pth /workspace/datasets/dota_256.pth \
    --sweep_name orcnn-sweep