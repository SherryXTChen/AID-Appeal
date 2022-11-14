from collections import defaultdict
import random
random.seed(0)

import torch
import torch.nn as nn
import torch.utils.data as data

import pytorch_lightning as pl

import clip

import datasets


class CLIPPredictor(pl.LightningModule):
    def __init__(self, opt, train_set_stats):
        super().__init__()
        self.opt = opt
        clip_model, _ = clip.load('ViT-L/14', device='cpu')
        input_size = 768
        self.clip_model = clip_model.visual

        self.clip_model.requires_grad_(False)
        self.clip_model.eval()

        # backbone
        self.shared = nn.Sequential(
            nn.Linear(input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
        )

        # predict hashtag label
        self.hashtag_head = nn.Sequential(
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, len(datasets.HASHTAGS)),
        )

        # predict retweet count range
        self.retweet_head = nn.Sequential(
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, len(datasets.RETWEETS)-1),
        )

        # predict like count range
        self.like_head = nn.Sequential(
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, len(datasets.LIKES)-1),
        )

        # handle unbalanced data
        self.hashtag_criterion = nn.CrossEntropyLoss()

        retweet_weights = train_set_stats['retweet_count_lst']
        retweet_weights = torch.Tensor([sum(retweet_weights) / (len(retweet_weights) * w) for w in retweet_weights])
        self.retweet_criterion = nn.CrossEntropyLoss(weight=retweet_weights)

        like_weights = train_set_stats['like_count_lst']
        like_weights = torch.Tensor([sum(like_weights) / (len(like_weights) * w) for w in like_weights])
        self.like_criterion = nn.CrossEntropyLoss(weight=like_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad,
                list(self.shared.parameters()) + \
                list(self.hashtag_head.parameters()) + \
                list(self.retweet_head.parameters()) + \
                list(self.like_head.parameters())),
            lr=self.opt.lr,
        )
        return optimizer

    def forward(self, x):
        inputs = self.clip_model(x)
        shared_info = self.shared(inputs)
        pred_hashtags = self.hashtag_head(shared_info)
        pred_retweets = self.retweet_head(shared_info)
        pred_likes = self.like_head(shared_info)

        outputs = {
            'hashtag': pred_hashtags, #torch.max(pred_hashtags, 1)[1],
            'retweets': pred_retweets,
            'likes': pred_likes,
        }
        return outputs

    def one_step(self, items, split):
        images = items['image'].to(self.device)
        gt_hashtags = items['hashtag'].to(self.device)
        gt_retweets = items['retweets'].to(self.device)
        gt_likes = items['likes'].to(self.device)

        outputs = self(images)
        pred_hashtags = outputs['hashtag']
        pred_retweets = outputs['retweets']
        pred_likes = outputs['likes']

        losses = {}
        losses[f'{split}/hashtag_loss'] = self.hashtag_criterion(pred_hashtags, gt_hashtags).mean()
        losses[f'{split}/retweets_loss'] = self.retweet_criterion(pred_retweets, gt_retweets).mean()
        losses[f'{split}/likes_loss'] = self.like_criterion(pred_likes, gt_likes).mean()
        losses[f'{split}/total'] = losses[f'{split}/hashtag_loss'] + losses[f'{split}/retweets_loss'] + losses[f'{split}/likes_loss']

        return losses

    def training_step(self, batch, batch_idx):
        split = 'train'
        losses = self.one_step(batch, split)
        self.log_dict(losses, sync_dist=True)
        return losses[f'{split}/total']

    def validation_step(self, batch, batch_idx):
        split = 'val'
        losses = self.one_step(batch, split)
        self.log_dict(losses, sync_dist=True)
        return losses[f'{split}/total']

    def validation_epoch_end(self, outputs):
        print('average total loss on validation set:', sum(outputs) / len(outputs))