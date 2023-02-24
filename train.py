import os
import random
random.seed(0)

import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import options
import models
import datasets

if __name__ == '__main__':
    opt = options.BaseOptions().gather_options()
    opt.out_dir = os.path.join(opt.out_dir, opt.name)
    for k, v in vars(opt).items():
        print(f'{k}: {v}')
    os.makedirs(opt.out_dir, exist_ok=True)
    
    with open(os.path.join(opt.out_dir, 'command.txt'), 'w') as out:
        for k, v in vars(opt).items():
            out.write(f'{k}: {v}\n')

    checkpoint_callback = ModelCheckpoint(
        dirpath = opt.out_dir,
        filename = '{epoch:02d}-{step}',
        save_last = True,
        save_top_k = 1,
        save_on_train_epoch_end = True,
    )

    if opt.gpus > 1:
        strategy = 'ddp_find_unused_parameters_false'
    else:
        strategy = None

    log_dir = f'{opt.out_dir}/logs'
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name='', version=0)

    # binary labelled data
    if opt.loss_type == 'singular':
        train_set = datasets.ScoreDataset(opt, 'synthetic', 'train')
        val_set = datasets.ScoreDataset(opt, 'synthetic', 'val')
        model = models.CLIPScorer(opt)
    else:
        train_set = datasets.ComparisonDataset(opt, 'train')
        val_set = datasets.ComparisonDataset(opt, 'val')
        model = models.CLIPComparator(opt)
    print(f'train samples: {len(train_set)}, val samples: {len(val_set)}')

    train_loader = data.DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads,
        shuffle=True,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads, 
        shuffle=False
    )
    
    ckpt_path = os.path.join(opt.out_dir, 'last.ckpt') 
    if os.path.exists(ckpt_path):
        resume = ckpt_path
    else:
        resume = None
    print('resume from', resume)

    trainer = pl.Trainer(
        devices = opt.gpus,
        accelerator = 'gpu',
        strategy = strategy,
        max_epochs = opt.num_epochs,
        resume_from_checkpoint = resume,
        callbacks = [checkpoint_callback],
        logger = tb_logger,
        log_every_n_steps = 1,
        check_val_every_n_epoch = 1,
    )
    trainer.fit(model, train_loader, val_loader)