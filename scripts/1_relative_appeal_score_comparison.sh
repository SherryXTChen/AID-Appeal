#!/bin/bash

name="pair_with_clip_$1"
root="datasets/$1"
copy_from="outputs/pair_with_clip_$1/scores_real_all.txt"

python train.py --name $name --loss_type pair --root $root --image_size 512 --batch_size 16 --gpus 2 --num_epochs 10 --lr 1e-3
python train.py --name $name --loss_type pair --root $root --image_size 512 --batch_size 16 --gpus 2 --num_epochs 10 --lr 1e-5 --unfreeze_pretrained --resume ckpts/$name/last.ckpt
python test.py --name $name --loss_type pair --root $root --image_size 512
cp outputs/$name/scores_real_all.txt $root/scores.txt
