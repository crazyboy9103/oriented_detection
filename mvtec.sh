declare -a pretraineds=("True" "False")
declare -a pretrained_backbones=("True" "False")
declare -a freeze_bns=("True" "False")
declare -a skip_flips=("True" "False")
declare -a skip_image_transforms=("True" "False")
declare -a trainable_backbone_layers=(1 2 3 4 5)
declare -a learning_rates=("0.001" "0.0001")
declare -a image_sizes=(256 512 800)
declare -a batch_sizes=(16 4 2)
length=${#image_sizes[@]}
# Nested loop to iterate over all combinations
for ((i=0; i<$length; i++)) do
    image_size=${image_sizes[$i]}
    batch_size=${batch_sizes[$i]}
    for pretrained in "${pretraineds[@]}"
    do
        for pretrained_backbone in "${pretrained_backbones[@]}"
        do
            for freeze_bn in "${freeze_bns[@]}"
            do
                for skip_flip in "${skip_flips[@]}"
                do
                    for skip_image_transform in "${skip_image_transforms[@]}"
                    do
                        for trainable_backbone_layer in "${trainable_backbone_layers[@]}"
                        do
                            for learning_rate in "${learning_rates[@]}"
                            do
                                python entrypoint.py \
                                    --model_type rotated \
                                    --pretrained $pretrained \
                                    --pretrained_backbone $pretrained_backbone \
                                    --freeze_bn $freeze_bn \
                                    --skip_flip $skip_flip \
                                    --skip_image_transform $skip_image_transform \
                                    --trainable_backbone_layers $trainable_backbone_layer \
                                    --learning_rate $learning_rate \
                                    --image_size $image_size \
                                    --batch_size $batch_size
                                python entrypoint.py \
                                    --model_type oriented \
                                    --pretrained $pretrained \
                                    --pretrained_backbone $pretrained_backbone \
                                    --freeze_bn $freeze_bn \
                                    --skip_flip $skip_flip \
                                    --skip_image_transform $skip_image_transform \
                                    --trainable_backbone_layers $trainable_backbone_layer \
                                    --learning_rate $learning_rate \
                                    --image_size $image_size \
                                    --batch_size $batch_size
                            done
                        done
                    done
                done
            done
        done
    done
done
