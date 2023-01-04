import os
from PIL import Image
from collections import defaultdict
import random
random.seed(0)

import torch.utils.data as data
import clip

# appealing: 1, unappealing: 0
class Food(data.Dataset):
    def __init__(self, opt, split):
        super().__init__()
        self.opt = opt

        self.appeal_lst = self.split_data(opt.appeal_root_list, split)
        self.unappeal_lst = self.split_data(opt.unappeal_root_list, split)

        self.image_lst = [(f, 0) for f in self.appeal_lst] + [(f, 1) for f in self.unappeal_lst]

        _, self.clip_preprocess = clip.load('ViT-L/14', device='cpu')

    def split_data(self, root_list, split):
        # 80 % train, 20 % val, last 10 samples test
        lst_split = []
        for root in root_list:
            data_lst = open(os.path.join(root, 'data.txt')).readlines()
            data_lst = [(os.path.join(root, 'images', f.strip()), f.split('/')[0]) for f in data_lst]

            data_dict = defaultdict(list)
            for f, label in data_lst:
                data_dict[label].append(f)

            for _, lst in data_dict.items():
                if split == 'train':
                    lst_split += lst[:int(self.opt.split_radio * len(lst))]
                elif split == 'val':
                    lst_split +=  lst[int(self.opt.split_radio * len(lst)):]
                else:
                    assert split == None
                    lst_split += lst

        return lst_split
        
    def __len__(self):
        return len(self.image_lst)

    def preprocess(self, f):
        img = Image.open(f).convert('RGB')
        w, h = img.size
        img = img.resize((min(h, w), min(h, w)))
        img = self.clip_preprocess(img)
        return img

    def getitem_pair(self, index):
        image1_path, label = self.image_lst[index] # 0: appealing, 1: unappealing
        
        if label == 0:
            image2_path = random.choice(self.unappeal_lst)
        else:
            assert label == 1, label
            image2_path = random.choice(self.appeal_lst)

        image1 = self.preprocess(image1_path)
        image2 = self.preprocess(image2_path)

        item = {
            'image1': image1,
            'image2': image2,
            'image1_path': image1_path,
            'image2_path': image2_path,
            'label': label,
        }
        return item

    def getitem_triplet(self, index):
        anchor_path, label = self.image_lst[index]
        if label == 0: # anchor appealing
            pos_path = random.choice(self.appeal_lst)
            neg_path = random.choice(self.unappeal_lst)
        else:    
            pos_path = random.choice(self.unappeal_lst)
            neg_path = random.choice(self.appeal_lst)

        anchor_image = self.preprocess(anchor_path)
        pos_image = self.preprocess(pos_path)
        neg_image = self.preprocess(neg_path)

        item = {
            'anchor_image': anchor_image,
            'pos_image': pos_image,
            'neg_image': neg_image,
            'anchor_path': anchor_path,
            'pos_path': pos_path,
            'neg_path': neg_path,
            'label': label,
        }
        return item

    def __getitem__(self, index):
        if self.opt.loss_type == 'pair':
            return self.getitem_pair(index)
        else:
            return self.getitem_triplet(index)


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
