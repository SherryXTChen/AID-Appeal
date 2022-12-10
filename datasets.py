import os
import glob
import json
import itertools
from PIL import Image
import random
random.seed(0)

import torch
import torch.utils.data as data
import clip


class Food(data.Dataset):
    def __init__(self, opt, split):
        super().__init__()
        self.opt = opt
        assert split in ['train', 'val'], split

        appeal_lst = self.setup(opt.appeal_root_list, split)
        unappeal_lst = self.setup(opt.unappeal_root_list, split)

        self.image_lst = [(f, random.choice(unappeal_lst)) for f in appeal_lst] + \
            [(random.choice(appeal_lst), f) for f in unappeal_lst]
        # self.image_lst = list(itertools.product(appeal_lst, unappeal_lst))

        _, self.clip_preprocess = clip.load('ViT-L/14', device='cpu')

    def setup(self, root_list, split):
        # 80 % train, 20 % val, last 10 samples test
        lst_split = []
        for root in root_list:
            dir_lst = glob.glob(f'{root}/*/', recursive=True)
            print(root, len(dir_lst))

            for dir in dir_lst:
                lst =  [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.jpg')]
                lst.sort()
                if split == 'train':
                    lst_split += lst[:int(0.8 * len(lst))]
                else:
                    lst_split +=  lst[int(0.8 * len(lst)):-10]
        return lst_split
        
    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, index):
        # get appeal image
        appeal_path, unappeal_path = self.image_lst[index]

        appeal_image = Image.open(appeal_path).convert('RGB')
        w, h = appeal_image.size
        appeal_image = appeal_image.resize((min(h, w), min(h, w)))
        appeal_image = self.clip_preprocess(appeal_image)

        unappeal_image = Image.open(unappeal_path).convert('RGB')
        w, h = unappeal_image.size
        unappeal_image = unappeal_image.resize((min(h, w), min(h, w)))
        unappeal_image = self.clip_preprocess(unappeal_image)

        flip = random.choice([True,False])
        if flip:
            images = torch.cat((unappeal_image, appeal_image), axis=0)
            label = 1
        else:
            images = torch.cat((appeal_image, unappeal_image), axis=0)
            label = 0   

        item = {
            'images': images,
            'label': label,
        }
        return item


class FoodRank(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        appeal_lst = self.setup(opt.appeal_root_list, 'val')
        unappeal_lst = self.setup(opt.unappeal_root_list, 'val')
        image_lst = appeal_lst + unappeal_lst

        _, self.clip_preprocess = clip.load('ViT-L/14', device='cpu')

        self.image_lst = []
        count = 0
        for image_path in image_lst:
            count += 1
            print(f'{count}/{len(image_lst)}')

            image = Image.open(image_path).convert('RGB')
            image = self.clip_preprocess(image)
            self.image_lst.append((image_path, image))

    def setup(self, root_list, split):
        # 80 % train, 20 % val, last 10 samples test
        lst_split = []
        for root in root_list:
            dir_lst = glob.glob(f'{root}/*/', recursive=True)
            if len(dir_lst) == 0:
                dir_lst = [root]
            
            print(root, len(dir_lst))
            root_lst = []

            for dir in dir_lst:
                lst =  [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.jpg')]
                lst.sort()
                root_lst +=  lst[int(0.8 * len(lst)):-10]
            
            random.shuffle(root_lst)
            lst_split += root_lst[:min(len(root_lst), self.opt.num_samples)]
        
        return lst_split

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, index):
        ref_image_path, ref_image = self.image_lst[index]

        image_lst = self.image_lst[:index] + self.image_lst[index+1:]        
        image_lst = [x for x in image_lst]

        item = {
            'ref_image': ref_image,
            'image_lst': image_lst,
            'ref_image_path': ref_image_path,
        }
        return item


def create_iterator(dataset, batch_size, shuffle):
    while True:
        sample_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True
        )

        for item in sample_loader:
            yield item

def create_datasets(opt):
    train_set = Food(opt, 'train')
    # train_set_stats = train_set.stats
    train_loader = data.DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    val_set = Food(opt, 'val')
    val_loader = data.DataLoader(
        val_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads, 
        shuffle=False,
        pin_memory=False)

    print(f'train samples: {len(train_set)}, val samples: {len(val_set)}')

    return train_loader, val_loader