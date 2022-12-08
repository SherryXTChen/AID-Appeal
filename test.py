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
    print('test samples:', len(test_set))
    model = models.CLIPComparator(opt)

    state = torch.load( f'{opt.out_dir}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval()
    model.to(device)
    
    last_layer = torch.nn.Softmax(dim=1).eval().to(device)

    image_rank_lst = []
    count = 0
    for item in test_set:
        count += 1
        print(f'{count}/{len(test_set)}')

        ref_image = item['ref_image'].to(device)
        image_lst = item['image_lst']
        ref_image_path = item['ref_image_path']

        score_lst = []
        for image in image_lst:
            image = image.to(device)
            inputs = torch.cat((ref_image, image), axis=0).unsqueeze(0)
            with torch.no_grad():
                outputs = last_layer(model.forward(inputs))
            score_lst.append(outputs[0][0])

        image_rank_lst.append(
            (
                ref_image_path,
                sum(score_lst)/len(score_lst),
                len([x for x in score_lst if x > 0.5]) /len(score_lst)
            )
        )
        # break

    image_rank_lst.sort(key = operator.itemgetter(2, 1))
    rank = 0
    for i in range(len(image_rank_lst)-1, -1, -1):
        rank += 1
        in_f, avg_score, order = image_rank_lst[i]
        dir_name, base_name = tuple(in_f.split('/')[-2:])
        out_f = os.path.join(out_dir, f'rank={str(rank).zfill(5)}_score={avg_score}_order={order}_{dir_name}_{base_name}')
        shutil.copy(in_f, out_f)