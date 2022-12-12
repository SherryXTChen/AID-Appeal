import os
import operator
import shutil

import torch

import utils
import options
import models
import datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    opt = options.BaseOptions().gather_options()
    opt.out_dir = os.path.join(opt.out_dir, opt.name)
    
    out_dir = opt.rank_dir
    utils.mkdir(out_dir)

    test_set = datasets.FoodRank(opt)
    model = models.CLIPComparator(opt)

    state = torch.load( f'{opt.out_dir}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval()
    model.to(device)

    image_score_lst = []
    for i in range(len(test_set)):
        print(f'{i+1}/{len(test_set)}')

        item = test_set[i]
        image = item['image'].unsqueeze(0).to(device)
        image_path = item['image_path']

        score = model(image)
        image_score_lst.append(
            (
                image_path,
                score,
            )
        )
        # break

    image_score_lst.sort(key = operator.itemgetter(1))
    for i in range(len(image_score_lst)-1, -1, -1):
        rank = i+1
        in_f, score = image_score_lst[i]
        dir_name, base_name = tuple(in_f.split('/')[-2:])
        out_f = os.path.join(out_dir, f'rank={str(rank).zfill(5)}_score={score}_{dir_name}_{base_name}')
        shutil.copy(in_f, out_f)
