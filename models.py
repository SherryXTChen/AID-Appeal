import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip
import random
random.seed(0)

def generate_weights(counts):
    weights = torch.Tensor([sum(counts) / (len(counts) * c) for c in counts])
    return weights


class CLIPComparator(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        clip_model, _ = clip.load('ViT-L/14', device='cpu')
        input_size = 768
        self.clip_model = clip_model.visual

        self.clip_model.requires_grad_(False)
        self.clip_model.eval()

        # backbone
        self.scorer = nn.Sequential(
            nn.Linear(input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            #nn.ReLU(),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, list(self.scorer.parameters())),
            lr=self.opt.lr,
        )
        return optimizer

    def forward(self, x):
        score = self.scorer(self.clip_model(x))
        return score

    def one_step(self, items, split):
        score1 = self(items['image1'].to(self.device))
        score2 = self(items['image2'].to(self.device))

        pred_labels = torch.cat([score1, score2], axis=-1)
        gt_labels = items['label'].to(self.device)

        ret = {}
        ret[f'{split}/accu'] = (torch.max(pred_labels, 1)[1] == gt_labels).sum() / gt_labels.shape[0]
        ret[f'{split}/loss'] = self.criterion(pred_labels, gt_labels).mean()

        return ret

    def training_step(self, batch, batch_idx):
        split = 'train'
        ret = self.one_step(batch, split)
        self.log_dict(ret, sync_dist=True)
        return ret[f'{split}/loss']

    def validation_step(self, batch, batch_idx):
        split = 'val'
        ret = self.one_step(batch, split)
        self.log_dict(ret, sync_dist=True)
        return ret # ret[f'{split}/loss']

    def validation_epoch_end(self, outputs):
        loss_lst = [o['val/loss'] for o in outputs]
        accu_lst = [o['val/accu'] for o in outputs]
        print('average loss on validation set:', sum(loss_lst) / len(loss_lst))
        print('average accuracy on validation set:', sum(accu_lst) / len(accu_lst))