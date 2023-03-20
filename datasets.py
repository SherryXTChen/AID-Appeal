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

def get_augmentation(opt):
    augmentation = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.RandomResizedCrop(size=(opt.image_size, opt.image_size), scale=(0.8, 1.0))
    ])
    return augmentation


def preprocess_image(image_path, augmentation, transform):
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    image = image.resize((min(h, w), min(h, w)))

    # add augmentation
    image = augmentation(image)
    image = transform(image)
    return image


# split images to train and val set
def split_data(data_path_list, split_start, split_end):
    assert 0 <= split_start < split_end <= 1
    data_path_list.sort()
    data_type_dict = defaultdict(list)
    for data_path in data_path_list:
        if data_path.endswith('/'):
            data_type = data_path[:-1]
        else:
            assert data_path.endswith('.jpg'), data_path
            data_type = data_path
        data_type = '_'.join(os.path.basename(data_type).split('_')[:2])
        data_type_dict[data_type].append(data_path)

    data_path_list_split = []
    for _, v in data_type_dict.items():
        data_path_list_split += v[int(split_start * len(v)) : int(split_end * len(v))]
    return data_path_list_split


# synthetic dataset for relative appeal score comparator
class ComparisonDataset(data.Dataset):
    def __init__(self, opt, split):
        super().__init__()
        self.opt = opt

        image_dir_list_synthetic = glob.glob(f'{opt.root}/5_synthetic/*/', recursive=True)
        if split == 'train':
            split_start = 0
            split_end = self.opt.split_ratio
        else:
            split_start = self.opt.split_ratio
            split_end = 1
        self.image_dir_list = split_data(image_dir_list_synthetic, split_start, split_end)

        if split == 'train':
            self.image_dir_list = [self.image_dir_list for _ in range(20)]
            self.image_dir_list = sum(self.image_dir_list, [])
            self.augmentation = get_augmentation(opt)
        else:
            self.augmentation = transforms.Compose([])
        _, self.transform = clip.load('ViT-L/14', device='cpu')
        
    def __len__(self):
        return len(self.image_dir_list)
    
    # raw score: [0, 1]
    # normalized score: [-5, 5]
    def get_score(self, image_path):
        score = float(os.path.basename(image_path).split('_')[0].replace('score=', '')) - 0.5
        return score * 10 

    def __getitem__(self, index):
        image_dir = self.image_dir_list[index]
        image_list = glob.glob(f'{image_dir}/*.jpg', recursive=True)

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


class BaseScoreDataset(data.Dataset):
    def __init__(self, opt, set_type, split):
        super().__init__()
        self.opt = opt
        assert set_type in ['all', 'real', 'synthetic']
        assert split in ['train', 'val', 'all']

        # food, room, both have textual inversion
        image_dir_list_real = glob.glob(f'{opt.root}/6_real/images/*.jpg', recursive=True)
        image_dir_list_synthetic = glob.glob(f'{opt.root}/5_synthetic/*/*.jpg', recursive=True)

        if set_type == 'all':
            image_dir_list = image_dir_list_real + image_dir_list_synthetic
        elif set_type == 'real':
            image_dir_list = image_dir_list_real
        else:
            image_dir_list = image_dir_list_synthetic
        image_dir_list.sort()
        
        if split == 'train':
            split_start = 0
            split_end = self.opt.split_ratio
        elif split == 'val':
            split_start = self.opt.split_ratio
            split_end = 1
        else:
            split_start = 0
            split_end = 1
        self.image_list = split_data(image_dir_list, split_start, split_end)
        self.image_list = [f.replace('//', '/') for f in self.image_list]

    def __len__(self):
        return len(self.image_list)


# for converting relative appeal score to absolute appeal score
class Relative2AbsoluteScoreDataset(BaseScoreDataset):
    def __init__(self, opt, set_type, split):
        super().__init__(opt, set_type, split)
        self.augmentation = transforms.Compose([])
        _, self.transform = clip.load('ViT-L/14', device='cpu')

        self.vote_list = glob.glob(f'{opt.root}/4_appeal_interpolation/images/*.jpg', recursive=True)
        self.vote_list.sort()
        self.vote_list = self.vote_list[::len(self.vote_list)//100][:100]
        self.vote_dataset = VoteDataset(self.vote_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = preprocess_image(image_path, self.augmentation, self.transform)
        score = -1 # placeholder
        item = {
            'image': image,
            'image_path': image_path,
            'image_score': score,
        }
        return item            


# images used for appeal score voting
class VoteDataset(data.Dataset):
    def __init__(self, image_list):
        super().__init__()
        self.image_list = image_list
        self.augmentation = transforms.Compose([])
        _, self.transform = clip.load('ViT-L/14', device='cpu')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = preprocess_image(image_path, self.augmentation, self.transform)
        return image


# real image dataset for appeal score prediction
class ScoreDataset(BaseScoreDataset):
    def __init__(self, opt, set_type, split):
        super().__init__(opt, set_type, split)
        self.score_dict = self.get_score_dict()
        assert set(self.image_list).issubset(set(list(self.score_dict.keys())))

        if split == 'train':
            self.augmentation = get_augmentation(opt)
        else:
            self.augmentation = transforms.Compose([])
        _, self.transform = clip.load('ViT-L/14', device='cpu')

    
    # take score from relative appeal score comparator, normalize it to [1, 10]
    def get_score_dict(self):
        score_list = open(os.path.join(self.opt.root, 'scores.txt')).readlines()
        score_list = [l.strip().split('\t') for l in score_list]
        score_list = [(l[0], float(l[-1])) for l in score_list]
        
        min_score = min([l[1] for l in score_list])
        max_score = max([l[1] for l in score_list])
        score_func = lambda x: (x - min_score) / (max_score - min_score) * 9 + 1
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
