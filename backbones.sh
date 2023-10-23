declare -a backbones=("resnet50" "resnet18" "mobilenetv3large" "efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3")
declare -a image_sizes=(256 512 800)
declare -a batch_sizes=(8 4 2)
length=${#image_sizes[@]}
# Nested loop to iterate over all combinations
for ((i=0; i<$length; i++)) do
    image_size=${image_sizes[$i]}
    batch_size=${batch_sizes[$i]}
    for backbone in "${backbones[@]}"
    do
        python entrypoint.py \
            --model_type rotated \
            --image_size $image_size \
            --batch_size $batch_size \
            --backbone_type $backbone \
            --num_workers $batch_size
        python entrypoint.py \
            --model_type oriented \
            --image_size $image_size \
            --batch_size $batch_size \
            --backbone_type $backbone \
            --num_workers $batch_size
    done
done
