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
        appeal_lst = glob.glob(f'{opt.data_dir}/appealing/*/*.jp*', recursive=True)
        unappeal_lst = glob.glob(f'{opt.data_dir}/unappealing/*/*.jp*', recursive=True)
        # appeal_lst = glob.glob(f'{opt.data_dir}/Cloudinary_Archive_*/*.jp*', recursive=True)
        # unappeal_lst = glob.glob(f'{opt.data_dir}/Cloudinary_Rejected_*/*.jp*', recursive=True)

        assert split in ['train', 'val'], split
        if split == 'train':
            appeal_lst = appeal_lst[:int(0.8 * len(appeal_lst))]
            unappeal_lst = unappeal_lst[:int(0.8 * len(unappeal_lst))]
        else:
            appeal_lst = appeal_lst[int(0.8 * len(appeal_lst)):]
            unappeal_lst = unappeal_lst[int(0.8 * len(unappeal_lst)):]

        self.image_lst = [(f, random.choice(unappeal_lst)) for f in appeal_lst] + \
            [(random.choice(appeal_lst), f) for f in unappeal_lst]
        # self.image_lst = list(itertools.product(appeal_lst, unappeal_lst))

        _, self.clip_preprocess = clip.load('ViT-L/14', device='cpu')
        
    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, index):
        # get appeal image
        appeal_path, unappeal_path = self.image_lst[index]
        appeal_image = Image.open(appeal_path).convert('RGB')
        appeal_image = self.clip_preprocess(appeal_image)

        unappeal_image = Image.open(unappeal_path).convert('RGB')
        unappeal_image = self.clip_preprocess(unappeal_image)

        if random.choice([True,False]):
            images = torch.cat((appeal_image, unappeal_image), axis=0)
            label = 0
        else:
            images = torch.cat((unappeal_image, appeal_image), axis=0)
            label = 1

        item = {
            'images': images,
            'label': label,
        }
        return item


class FoodAppeal(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        image_lst = glob.glob(f'{opt.data_dir}/appealing/adobe_stock-*/*.jp*', recursive=True)[:100] + \
            glob.glob(f'{opt.data_dir}/appealing/google-*/*.jp*', recursive=True)[:100]

        _, self.clip_preprocess = clip.load('ViT-L/14', device='cpu')

        self.image_lst = []
        count = 0
        for image_path in image_lst:
            count += 1
            print(f'{count}/{len(image_lst)}')

            image = Image.open(image_path).convert('RGB')
            image = self.clip_preprocess(image)
            self.image_lst.append((image_path, image))
        
    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, index):
        # get appeal image
        ref_image_path, ref_image = self.image_lst[index]

        image_lst = self.image_lst[:index] + self.image_lst[index+1:]        
        image_lst = [x[1] for x in image_lst]

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