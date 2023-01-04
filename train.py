import os
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import options
import models
import datasets

def setup():
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
        save_on_train_epoch_end = True,
    )

    model_type = getattr(models, f'CLIPComparator_{opt.model_type.capitalize()}')
    model = model_type(opt)

    ckpt_path = opt.resume
    if os.path.exists(ckpt_path):
        print('resume from', ckpt_path)
        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)

    resume = os.path.join(opt.out_dir, 'last.ckpt')
    if not os.path.exists(resume):
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
        check_val_every_n_epoch=opt.num_epochs * 10,
    )

    return opt, trainer, model