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
    
    last_layer = torch.nn.Softmax(dim=1).eval().to(device)

    image_compare_dict = {}
    image_rank_lst = []
    count = 0
    for item in test_set:
        count += 1
        print(f'{count}/{len(test_set)}')

        ref_image = item['ref_image'].to(device)
        image_lst = item['image_lst']
        ref_image_path = item['ref_image_path']

        preferred = 0
        for image_path, image in image_lst:
            k = tuple(sorted([ref_image_path, image_path]))
            if k in image_compare_dict:
                if image_compare_dict[k] == ref_image_path:
                    preferred += 1
            else:
                image = image.to(device)
                inputs = torch.cat((ref_image, image), axis=0).unsqueeze(0)
                with torch.no_grad():
                    outputs = last_layer(model.forward(inputs))
                
                if outputs[0][0] > 0.5:
                    image_compare_dict[k] = ref_image_path
                    preferred += 1
                else:
                    image_compare_dict[k] = image_path

        image_rank_lst.append(
            (
                ref_image_path,
                preferred / len(image_lst),
            )
        )
        # break

    # image_rank_lst.sort(key = operator.itemgetter(2, 1))
    image_rank_lst.sort(key = operator.itemgetter(1))
    rank = 0
    for i in range(len(image_rank_lst)-1, -1, -1):
        rank += 1
        in_f, order = image_rank_lst[i]
        dir_name, base_name = tuple(in_f.split('/')[-2:])
        out_f = os.path.join(out_dir, f'rank={str(rank).zfill(5)}_order={order}_{dir_name}_{base_name}')
        shutil.copy(in_f, out_f)
    '''
    image_rank_lst = [(test_set[0]['ref_image_path'], test_set[0]['ref_image'])] + test_set[0]['image_lst']
    for j in range(len(image_rank_lst)):
        print(f'{j+1}/{len(image_rank_lst)}')
        #initially swapped is false
        swapped = False
        i = 0
        while i<len(image_rank_lst)-1:
            #comparing the adjacent elements
            inputs = torch.cat((image_rank_lst[i][1], image_rank_lst[i+1][1]), axis=0).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = last_layer(model.forward(inputs))
            
            if outputs[0][0] < 0.5:
                image_rank_lst[i],image_rank_lst[i+1] = image_rank_lst[i+1],image_rank_lst[i]
                #Changing the value of swapped
                swapped = True
            i = i+1
        #if swapped is false then the list is sorted
        #we can stop the loop
        if swapped == False:
            break

    for i in range(len(image_rank_lst)):
        rank = i + 1
        in_f = image_rank_lst[i][0]
        dir_name, base_name = tuple(in_f.split('/')[-2:])
        out_f = os.path.join(out_dir, f'rank={str(rank).zfill(5)}_{dir_name}_{base_name}')
        shutil.copy(in_f, out_f)
    '''