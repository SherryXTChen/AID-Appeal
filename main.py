import os
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import options
import models
import datasets

if __name__ == '__main__':
    opt = options.BaseOptions().gather_options()
    opt.out_dir = os.path.join(opt.out_dir, opt.name)
    utils.mkdir(opt.out_dir)
    
    with open(os.path.join(opt.out_dir, 'command.txt'), 'w') as out:
        for k, v in vars(opt).items():
            out.write(f'{k}: {v}\n')

    checkpoint_callback = ModelCheckpoint(
        dirpath = opt.out_dir,
        filename = '{epoch:02d}-{step}',
        save_last = True,
        save_top_k = 1,
        mode='min',
        every_n_train_steps = 1e5 // opt.batch_size,
    )

    ckpt_path = f'{opt.out_dir}/last.ckpt'
    if os.path.exists(ckpt_path):
        resume = ckpt_path
    else:
        resume = None
    
    if opt.gpus > 1:
        strategy = 'ddp_find_unused_parameters_false'
    else:
        strategy = None

    log_dir = f'{opt.out_dir}/logs'
    utils.mkdir(log_dir)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name='')

    trainer = pl.Trainer(
        devices = opt.gpus,
        accelerator = 'gpu',
        strategy = strategy,
        max_epochs=opt.num_epochs,
        resume_from_checkpoint=resume,
        callbacks = [checkpoint_callback],
        logger = tb_logger,
        log_every_n_steps = 1,
        check_val_every_n_epoch=1,
    )

    train_loader, val_loader = datasets.create_datasets(opt)
    model = models.CLIPComparator(opt)
    trainer.validate(model, val_loader)
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)