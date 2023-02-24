from collections import defaultdict
import random
random.seed(0)

import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip

from diffusers.models import AutoencoderKL

torch.backends.cudnn.benchmark = True

# part 1
class CLIPComparator(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.create_model()
            
    def create_model(self):
        clip_model, _ = clip.load('ViT-L/14', device='cpu')
        self.pretrained_model = clip_model.visual
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.eval()

        self.backbone = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        self.model_list = [self.backbone, self.head]

    def configure_optimizers(self):
        param_list = [list(m.parameters()) for m in self.model_list]
        param_list = sum(param_list, [])

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, param_list),
            lr=self.opt.lr,
        )
        return optimizer

    def forward(self, x):
        return NotImplementedError()

    def one_step(self, items, split):
        feature_list = [self.backbone(self.pretrained_model(image.to(self.device))) for image in items['image_list']]
        pred_label = self.head(torch.cat(feature_list, axis=-1))
        gt_score_list = [gt_score.to(self.device).float().unsqueeze(-1) 
            for gt_score in items['image_score_list']]
        gt_label = gt_score_list[0] - gt_score_list[1]

        ret = {}
        ret[f'{split}/pairwise_diff_loss'] = nn.L1Loss()(pred_label, gt_label).mean()
        return ret

    def training_step(self, batch, batch_idx):
        split = 'train'
        ret = self.one_step(batch, split)
        self.log_dict(ret, sync_dist=True)
        
        loss = sum([v for k, v in ret.items() if k.endswith('_loss')])
        return loss

    def validation_step(self, batch, batch_idx):
        split = 'val'
        ret = self.one_step(batch, split)
        self.log_dict(ret, sync_dist=True)
        return ret

    def validation_epoch_end(self, outputs):
        dict = defaultdict(list)
        for o in outputs:
            for k, v in o.items():
                dict[k].append(v)
        for k, v in dict.items():
            print('validation_epoch_end', k, sum(v)/len(v))

# part 2
class CLIPScorer(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.create_model()
            
    def create_model(self):
        self.pretrained_model = VQGAN()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.eval()

        clip_model, _ = clip.load('ViT-L/14', device='cpu')
        self.backbone = clip_model.visual
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        kernel_size = self.opt.image_size // 8 // (224 // 14) # patch size: 32 x 32 rgb pixels
        self.backbone.conv1 = nn.Conv2d(in_channels=4, out_channels=1024, kernel_size=kernel_size, stride=kernel_size, bias=False)
        for param in self.backbone.conv1.parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        self.model_list = [self.backbone.conv1, self.head]

    def configure_optimizers(self):
        param_list = [list(m.parameters()) for m in self.model_list]
        param_list = sum(param_list, [])

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, param_list),
            lr=self.opt.lr,
        )
        return optimizer

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.backbone(x)
        x = self.head(x)
        return x
      
    def one_step(self, items, split):
        pred_score = self(items['image'].to(self.device))
        gt_score = items['image_score'].to(self.device).float().unsqueeze(-1)
        ret = {}
        ret[f'{split}/score_loss'] = nn.L1Loss()(pred_score, gt_score) # TODO: add image log
        return ret

    def training_step(self, batch, batch_idx):
        split = 'train'
        ret = self.one_step(batch, split)
        self.log_dict(ret, sync_dist=True)
        
        loss = sum([v for k, v in ret.items() if k.endswith('_loss')])
        return loss

    def validation_step(self, batch, batch_idx):
        split = 'val'
        ret = self.one_step(batch, split)
        self.log_dict(ret, sync_dist=True)
        return ret

    def validation_epoch_end(self, outputs):
        dict = defaultdict(list)
        for o in outputs:
            for k, v in o.items():
                dict[k].append(v)
        for k, v in dict.items():
            print('validation_epoch_end', k, sum(v)/len(v))

class VQGAN(nn.Module):
    def __init__(self):
        super().__init__()    
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder='vae').cuda()

    def forward(self, image):
        with torch.no_grad():
            return self.vae.encode(image).latent_dist.mode()
