import os
from PIL import Image
from collections import defaultdict
import glob
import random
random.seed(0)

import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

import clip


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def get_augmentation(opt):
    augmentation = nn.Sequential(
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.RandomResizedCrop(size=(opt.image_size, opt.image_size), scale=(0.8, 1.0))
    )
    return augmentation

def preprocess_image(image_path, augmentation, transform):
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    image = image.resize((min(h, w), min(h, w)))

    # add augmentation
    image = augmentation(image)
    image = transform(image)
    return image


# part 1
class ComparisonDataset(data.Dataset):
    def __init__(self, opt, split):
        super().__init__()
        self.opt = opt
        self.image_dir_list = self.split_data_dir(split)
        if split == 'train':
            self.image_dir_list = [self.image_dir_list for _ in range(20)]
            self.image_dir_list = sum(self.image_dir_list, [])
            self.augmentation = get_augmentation(opt)
        else:
            self.augmentation = nn.Sequential()
        _, self.transform = clip.load('ViT-L/14', device='cpu')
        
    def __len__(self):
        return len(self.image_dir_list)
    
    def split_data_dir(self, split):
        image_dir_list = glob.glob(f'{self.opt.root}/5_data/*/', recursive=True)
        image_dir_list.sort()

        image_type_dict = defaultdict(list)
        for image_dir in image_dir_list:
            image_type = os.path.basename(image_dir[:-1]).split('_')[0]
            image_type_dict[image_type].append(image_dir)

        image_dir_list_split = []
        for k, v in image_type_dict.items():
            if split == 'train':
                image_dir_list_split += v[:int(self.opt.split_ratio * len(v))]
            else:
                image_dir_list_split +=  v[int(self.opt.split_ratio * len(v)):]
        return image_dir_list_split

    def get_score(self, image_path):
        score_change = int(os.path.basename(image_path).split('_')[0].replace('score=', ''))
        assert score_change in [0, 1, -1]
        score = score_change + 1 
        return score

    def __getitem__(self, index):
        image_dir = self.image_dir_list[index]
        image_list = glob.glob(f'{image_dir}/*_type=image.png', recursive=True)

        image_path_list = list(random.sample(image_list, 2))
        image_list = [
            preprocess_image(f, self.augmentation, self.transform) for f in image_path_list
        ]
        image_score_list = [self.get_score(f) for f in image_path_list]

        item = {
            'image_list': image_list,
            'image_path_list': image_path_list,
            'image_score_list': image_score_list,
        }
        return item


# part 2
class BaseScoreDataset(data.Dataset):
    def __init__(self, opt, set_type, split):
        super().__init__()
        self.opt = opt
        assert set_type in ['all', 'real', 'synthetic']
        assert split in ['train', 'val', 'all']
        assert (set_type, split) in [('all', 'all'), ('real', 'all'),
            ('synthetic', 'train'), ('synthetic', 'val'), ('synthetic', 'all')]

        # food, room, both have textual inversion
        image_dir_list_real = glob.glob(f'{opt.root}/4_filtered/images/', recursive=True) + \
            glob.glob(f'{opt.root}/4_preprocessed/*/', recursive=True) + \
            glob.glob(f'{opt.root}/6_textual_inversion/3_preprocessed/*/', recursive=True)
        image_dir_list_synthetic = glob.glob(f'{opt.root}/5_data/*/', recursive=True)

        if set_type == 'all':
            image_dir_list = image_dir_list_real + image_dir_list_synthetic
        elif set_type == 'real':
            image_dir_list = image_dir_list_real
        else:
            image_dir_list = image_dir_list_synthetic
        image_dir_list.sort()
        self.image_list = self.split_data_list(image_dir_list, split)
        self.image_list = [f.replace('//', '/') for f in self.image_list]

    def __len__(self):
        return len(self.image_list)
    
    def split_data_list(self, image_dir_list, split):
        image_list_total = []
        for image_dir in image_dir_list:
            image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
            if '_type=' in image_list[0]:
                image_list = [f for f in image_list if f.endswith('_type=image.png')]
            image_list.sort()
            if split == 'train':
                image_list = image_list[:int(self.opt.split_ratio * len(image_list))]
            elif split == 'val':
                image_list = image_list[int(self.opt.split_ratio * len(image_list)):]
            else:
                image_list = image_list
            image_list_total += image_list
        return image_list_total


class Relative2AbsoluteScoreDataset(BaseScoreDataset):
    def __init__(self, opt, set_type, split):
        super().__init__(opt, set_type, split)
        self.augmentation = nn.Sequential()
        _, self.transform = clip.load('ViT-L/14', device='cpu')

        self.vote_list = glob.glob(f'{opt.root}/6_textual_inversion/3_preprocessed/*/*.png', recursive=True)
        if '/food' in opt.root:
            vote_list_additional = glob.glob(f'{opt.root}/4_filtered/images/*.png', recursive=True)
            split_size = len(vote_list_additional) // 10
            vote_list_additional = [vote_list_additional[i*split_size+int(0.95*split_size):(i+1)*split_size] for i in range(10)]
            vote_list_additional = sum(vote_list_additional, [])
            self.vote_list += vote_list_additional

        self.vote_dataset = VoteDataset(self.vote_list)

    def get_score(self, image_path):
        if '/6_textual_inversion/' in image_path:
            if 'delicious' in image_path or 'clean' in image_path:
                score = 1
            else:
                score = -1
        elif '/4_filtered/' in image_path or '/4_preprocessed/' in image_path:
            score = 0
        else:
            image_name = os.path.basename(image_path)
            score = int(image_name.split('_')[0].replace('score=', ''))
        return score

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = preprocess_image(image_path, self.augmentation, self.transform)
        score = self.get_score(image_path)
        item = {
            'image': image,
            'image_path': image_path,
            'image_score': score,
        }
        return item            


class VoteDataset(data.Dataset):
    def __init__(self, image_list):
        super().__init__()
        self.image_list = image_list
        self.augmentation = nn.Sequential()
        _, self.transform = clip.load('ViT-L/14', device='cpu')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = preprocess_image(image_path, self.augmentation, self.transform)
        return image


class ScoreDataset(BaseScoreDataset):
    def __init__(self, opt, set_type, split):
        super().__init__(opt, set_type, split)
        self.score_dict = self.get_score_dict()
        assert set(self.image_list).issubset(set(list(self.score_dict.keys())))

        if split == 'train':
            self.augmentation = get_augmentation(opt)
        else:
            self.augmentation = nn.Sequential()

        self.transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_score_dict(self):
        score_list = open(os.path.join(self.opt.root, 'scores.txt')).readlines()
        score_list = [l.strip().split('\t') for l in score_list]
        score_list = [(l[0], float(l[-1])) for l in score_list]
        
        min_score = min([l[1] for l in score_list])
        max_score = max([l[1] for l in score_list])
        score_func = lambda x: (x - min_score) / (max_score - min_score) * 10.
        score_dict = {l[0]: score_func(l[1]) for l in score_list}
        return score_dict

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = preprocess_image(image_path, self.augmentation, self.transform)
        score = self.score_dict[image_path]
        item = {
            'image': image,
            'image_path': image_path,
            'image_score': score,
        }
        return item