import os
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import options
import models
import datasets
import train

if __name__ == '__main__':
    opt, trainer, model = train.setup()

    # binary labelled data
    train_set = datasets.Food(opt, 'train')
    val_set = datasets.Food(opt, 'val')
    print(f'train samples: {len(train_set)}, val samples: {len(val_set)}')

    train_loader = data.DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads, 
        shuffle=False,
        pin_memory=False)
    
    # trainer.validate(model, val_loader)
    trainer.fit(model, train_loader, val_loader)
    # trainer.validate(model, val_loader)