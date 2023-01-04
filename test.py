import os
import random
random.seed(0)

import torch

import utils
import options
import models
import datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def rank_by_compare(test_set, i):
    item = test_set[i]
    image = item['image1'].unsqueeze(0).to(device)
    with torch.no_grad():
        feat1 = model.clip_model(image)

    index_lst = list(range(len(test_set)))
    index_lst.remove(i)
    index_lst = random.sample(index_lst, opt.num_samples)

    feat2_lst = []
    for j in index_lst:
        image2 = test_set[j]['image1'].unsqueeze(0).to(device)
        with torch.no_grad():
            feat2 = model.clip_model(image2)
        feat2_lst.append(feat2)

    feat1_lst = torch.cat([feat1 for _ in range(opt.num_samples)], axis=0)
    feat2_lst = torch.cat(feat2_lst, axis=0)

    with torch.no_grad():
        pred_labels = model(torch.cat([feat2_lst, feat1_lst], axis=-1))
    vote = torch.sum(torch.max(pred_labels, 1)[1]).item()

    in_f = item['image1_path']
    score = vote / opt.num_samples

    return in_f, score

def rank_by_score(test_set, i):
    item = test_set[i]
    image = item['image1'].unsqueeze(0).to(device)
    in_f = item['image1_path']

    with torch.no_grad():
        score = model(image)[0][0]
    return in_f, score


if __name__ == '__main__':
    opt = options.BaseOptions().gather_options()
    out_path = os.path.join('outputs', opt.name + '_' + opt.split, 'lists.txt')
    out_dir = os.path.dirname(out_path)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out = open(out_path, 'w')

    test_set = datasets.Food(opt, opt.split)
    
    model_type = getattr(models, f'CLIPComparator_{opt.model_type.capitalize()}')
    model = model_type(opt)

    if opt.model_type == 'siamese':
        rank_func = rank_by_score
    else:
        assert opt.model_type == 'concate', opt.model_type
        rank_func = rank_by_compare

    state = torch.load( f'{opt.out_dir}/{opt.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval()
    model.to(device)

    for i in range(len(test_set)):
        print(f'{i+1}/{len(test_set)}')

        in_f, score = rank_func(test_set, i)
        dir_name, base_name = tuple(in_f.split('/')[-2:])
        out_f = f'score={score}_{dir_name}_{base_name}'
        out.write(f'{out_f}\t{in_f}\n')

    out.close()
