import os
from tqdm import tqdm
import torch

import options
import models
import datasets

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def rank_by_compare(model, item, vote_feature_list):
    image = item['image'].unsqueeze(0).to(DEVICE)
    feature = model.backbone(model.pretrained_model(image))

    pred_label_total = 0
    for vote_feature in vote_feature_list:
        pred_label = model.head(torch.cat([feature, vote_feature], axis=-1)).mean().item()
        pred_label_total += pred_label

    in_f = item['image_path']
    score = pred_label_total / len(vote_feature_list)

    return in_f, score

@torch.no_grad()
def get_features(model, image_set):
    image_list = [item.unsqueeze(0).to(DEVICE) for item in image_set]
    feature_list = []
    for image in image_list:
        feature = model.backbone(model.pretrained_model(image))
        feature_list.append(feature)
    return feature_list

@torch.no_grad()
def rank_by_score(model, item):
    image = item['image'].unsqueeze(0).to(DEVICE)
    in_f = item['image_path']

    with torch.no_grad():
        score = model(image).item()
    return in_f, score

# appeal score inference using relative appeal score comparator
def rank_part_1(opt):
    opt = options.BaseOptions().gather_options()
    out_dir = os.path.join('outputs', opt.name)
    os.makedirs(out_dir, exist_ok=True)

    model = models.CLIPComparator(opt)
    ckpt_path = os.path.join(opt.out_dir, opt.name, 'last-v1.ckpt')
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval()
    model.to(DEVICE)

    vote_features = None
    out_total = open(os.path.join(out_dir, f'scores_all_all.txt'), 'w')

    for set_type, split in [('synthetic', 'all'), ('real', 'all')]:
        data_set = datasets.Relative2AbsoluteScoreDataset(opt, set_type, split)
        
        if vote_features == None:
            print('number of voting images', len(data_set.vote_dataset))
            vote_features = get_features(model, data_set.vote_dataset)

        out = open(os.path.join(out_dir, f'scores_{set_type}_{split}.txt'), 'w')
        for item in tqdm(data_set):
            in_f, pred_score = rank_by_compare(model, item, vote_features)
            gt_score = item['image_score']
            line = f'{in_f}\t{gt_score}\t{pred_score}\n'
            out.write(line)
            out_total.write(line)
        out.close()
    out_total.close()


# appeal score inference using appeal score predictor
def rank_part_2(opt):
    opt = options.BaseOptions().gather_options()
    out_dir = os.path.join('outputs', opt.name)
    os.makedirs(out_dir, exist_ok=True)

    model = models.CLIPScorer(opt)
    ckpt_path = os.path.join(opt.out_dir, opt.name, 'last-v1.ckpt')
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval()
    model.to(DEVICE)

    out_total = open(os.path.join(out_dir, f'scores_all_all.txt'), 'w')

    for set_type, split in [('real', 'all'), ('synthetic', 'all')]:
        data_set = datasets.ScoreDataset(opt, set_type, split)
        
        diff = 0
        out = open(os.path.join(out_dir, f'scores_{set_type}_{split}.txt'), 'w')
        for item in tqdm(data_set):
            in_f, pred_score = rank_by_score(model, item)
            gt_score = data_set.score_dict[item['image_path']]
            line = f'{in_f}\t{gt_score}\t{pred_score}\n'
            out.write(line)
            out_total.write(line)
            diff += abs(gt_score - pred_score)
        out.close()
        print('average score difference', diff / len(data_set))
    
    out_total.close()


if __name__ == '__main__':
    opt = options.BaseOptions().gather_options()
    if opt.loss_type == 'singular':
        rank_part_2(opt)
    else:
        rank_part_1(opt)