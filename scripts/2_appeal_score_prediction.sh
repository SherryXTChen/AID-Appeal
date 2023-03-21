#!/bin/bash

name="singular_with_clip_$1"
root="ImageAppeal/$1"
echo "$name $root"

python train.py --name $name --loss singular --root $root --image_size 512 --batch_size 16 --gpus 2 --num_epochs 10 --lr 1e-3
python train.py --name $name --loss singular --root $root --image_size 512 --batch_size 16 --gpus 2 --num_epochs 10 --lr 1e-5 --unfreeze_pretrained --resume ckpts/$name/last.ckpt
python test.py --name $name --loss_type singular --root $root --image_size 512
