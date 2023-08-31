python /workspace/wandb_sweep.py \
    --model_type oriented \
    --dataset dota \
    --project_name orcnn-sweep \
    --experiment_name orcnn-sweep-512 \
    --image_size 512 \
    --data_path /datasets/split_ss_dota_512 \
    --data_pth /workspace/datasets/dota_512.pth \
    --sweep_name orcnn-sweep