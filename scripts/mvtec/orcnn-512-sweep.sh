python /workspace/wandb_sweep.py \
    --model_type oriented \
    --dataset mvtec \
    --project_name orcnn-sweep \
    --experiment_name orcnn-sweep-512 \
    --image_size 512 \
    --data_path /datasets/split_ss_mvtec \
    --data_pth /workspace/datasets/mvtec.pth \
    --sweep_name orcnn-sweep