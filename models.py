import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip

def generate_weights(counts):
    weights = torch.Tensor([sum(counts) / (len(counts) * c) for c in counts])
    return weights

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

        # predict label
        self.label_head = nn.Sequential(
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, len(opt.label_list)),
        )

        # predict retweet count range
        self.retweet_head = nn.Sequential(
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, len(opt.retweets_range)),
        )

        # predict like count range
        self.like_head = nn.Sequential(
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, len(opt.likes_range)),
        )

        # handle unbalanced data
        self.label_criterion = nn.CrossEntropyLoss(weight=generate_weights(train_set_stats['label_count_list']))
        self.retweet_criterion = nn.CrossEntropyLoss(weight=generate_weights(train_set_stats['retweets_count_list']))
        self.like_criterion = nn.CrossEntropyLoss(weight=generate_weights(train_set_stats['likes_count_list']))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad,
                list(self.shared.parameters()) + \
                list(self.label_head.parameters()) + \
                list(self.retweet_head.parameters()) + \
                list(self.like_head.parameters())),
            lr=self.opt.lr,
        )
        return optimizer

    def forward(self, x):
        inputs = self.clip_model(x)
        shared_info = self.shared(inputs)
        pred_labels = self.label_head(shared_info)
        pred_retweets = self.retweet_head(shared_info)
        pred_likes = self.like_head(shared_info)

        outputs = {
            'label': pred_labels, #torch.max(pred_labels, 1)[1],
            'retweets': pred_retweets,
            'likes': pred_likes,
        }
        return outputs

    def one_step(self, items, split):
        images = items['image'].to(self.device)
        gt_labels = items['label'].to(self.device)
        gt_retweets = items['retweets'].to(self.device)
        gt_likes = items['likes'].to(self.device)

        outputs = self(images)
        pred_labels = outputs['label']
        pred_retweets = outputs['retweets']
        pred_likes = outputs['likes']

        ret = {}
        ret[f'{split}/label_accu'] = (torch.max(pred_labels, 1)[1] == gt_labels).sum() / gt_labels.shape[0]
        ret[f'{split}/retweets_accu'] = (torch.max(pred_retweets, 1)[1] == gt_retweets).sum() / gt_retweets.shape[0]
        ret[f'{split}/likes_accu'] = (torch.max(pred_likes, 1)[1] == gt_likes).sum() / gt_likes.shape[0]

        ret[f'{split}/label_loss'] = self.label_criterion(pred_labels, gt_labels).mean()
        ret[f'{split}/retweets_loss'] = self.retweet_criterion(pred_retweets, gt_retweets).mean()
        ret[f'{split}/likes_loss'] = self.like_criterion(pred_likes, gt_likes).mean()
        ret[f'{split}/total_loss'] = ret[f'{split}/label_loss'] + ret[f'{split}/retweets_loss'] + ret[f'{split}/likes_loss']

        return ret

    def training_step(self, batch, batch_idx):
        split = 'train'
        ret = self.one_step(batch, split)
        self.log_dict(ret, sync_dist=True)
        return ret[f'{split}/total_loss']

    def validation_step(self, batch, batch_idx):
        split = 'val'
        ret = self.one_step(batch, split)
        self.log_dict(ret, sync_dist=True)
        return ret[f'{split}/total_loss']

    def validation_epoch_end(self, outputs):
        print('average total loss on validation set:', sum(outputs) / len(outputs))
