python train.py --name pair_with_clip_$1 --loss pair --root ImageAppeal/$1 --image_size 512 --batch_size 16 --gpus 2 --num_epochs 10 --lr 1e-3
python train.py --name pair_with_clip_$1 --loss pair --root ImageAppeal/$1 --image_size 512 --batch_size 16 --gpus 2 --num_epochs 10 --lr 1e-5 --unfreeze_pretrained --resume ckpts/pair_with_clip_$1/last.ckpt
python test.py --name pair_with_clip_$1 --loss_type pair --root ImageAppeal/$1 --image_size 512
cp outputs/pair_with_clip_$1/scores_real_all.txt ImageAppeal/$1/scores.txt