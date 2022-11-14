import os
import glob
import json
import itertools
from PIL import Image

import torch.utils.data as data
import clip

HASHTAGS = ['#food', '#hotel', '#shoes']
RETWEETS = [0, 1, 3, 10, float('inf')]
LIKES = [0, 1, 3, 10, float('inf')]

class TwitterMedia(data.Dataset):
    def __init__(self, opt, split):
        super().__init__()
        self.opt = opt
        lsts = [glob.glob(f'{opt.data_dir}/{hashtag}/images/*.jpg', recursive=True) for hashtag in HASHTAGS]
        for l in lsts:
            l.sort()

        assert split in ['train', 'val'], split
        if split == 'train':
            lsts = [l[:int(0.8 * len(l))] for l in lsts]
        else:
            lsts = [l[int(0.8 * len(l)):] for l in lsts]

        self.image_lst = list(itertools.chain(*lsts))
        # self.image_lst = self.image_lst[:len(split) * 16] # for debugging

        # get stats
        if split == 'train':
            json_lst = []
            for image_path in self.image_lst:
                json_path, _ = self.get_json_path(image_path)
                json_lst.append(json_path)
                
            json_lst = list(set(json_lst))
            self.stats = {
                'retweet_count_lst': [176696, 44976, 15851, 3198], 
                'like_count_lst': [126049, 65959, 34022, 14691]
            } # get_stats(json_lst)
            # print('train set stats:', self.stats)

        _, self.clip_preprocess = clip.load('ViT-L/14', device='cpu')
        print(f'{split} set: {len(self)}')
        
    def __len__(self):
        return len(self.image_lst)

    def get_json_path(self, image_path, return_hashtag=False):
        hashtag = None

        json_path = image_path.split('/')
        assert json_path[-2] == 'images', json_path[-2]
        json_path[-2] = 'jsons'
        json_path[-1] = json_path[-1].split('_')[0] + '.json'

        if return_hashtag:
            hashtag = json_path[-3]
            assert hashtag in HASHTAGS, hashtag

        json_path = '/'.join(json_path)
        assert os.path.exists(json_path), json_path

        return json_path, hashtag


    def __getitem__(self, index):
        # get image
        image_path = self.image_lst[index]
        image = Image.open(image_path).convert('RGB')
        image = self.clip_preprocess(image)

        # get meta data
        # hashtag/images/id.json
        json_path, hashtag  = self.get_json_path(image_path, return_hashtag=True)
        hashtag_label = HASHTAGS.index(hashtag)

        dict = json.load(open(json_path))
        retweets = dict['retweets']
        likes = dict['likes']

        retweets_label = -1
        for r in RETWEETS:
            if retweets >= r:
                retweets_label += 1
            else:
                break
        assert retweets_label >= 0, retweets_label
        
        likes_label = -1
        for l in LIKES:
            if likes >= l:
                likes_label += 1
            else:
                break
        assert likes_label >= 0, likes_label

        item = {
            'image': image,
            'hashtag': hashtag_label,
            'retweets': retweets_label,
            'likes': likes_label
        }
        return item

def get_stats(json_lst):
    retweet_lst = []
    like_lst = []
    for j in range(len(json_lst)):
        print(f'{j+1}/{len(json_lst)}')
        json_file = json_lst[j]
        dict = json.load(open(json_file))
        retweet_count = dict['retweets']
        like_count = dict['likes']
        retweet_lst.append(retweet_count)
        like_lst.append(like_count)

    retweet_count_lst = []
    for i in range(len(RETWEETS)-1):
        min_bound = RETWEETS[i]
        max_bound = RETWEETS[i+1]
        count = len([x for x in retweet_lst if min_bound <= x < max_bound])
        retweet_count_lst.append(count)

    like_count_lst = []
    for i in range(len(LIKES)-1):
        min_bound = LIKES[i]
        max_bound = LIKES[i+1]
        count = len([x for x in like_lst if min_bound <= x < max_bound])
        like_count_lst.append(count)

    ret = {
        'retweet_count_lst': retweet_count_lst,
        'like_count_lst': like_count_lst
    }
    return ret


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
    train_set = TwitterMedia(opt, 'train')
    train_set_stats = train_set.stats
    train_loader = data.DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_set = TwitterMedia(opt, 'val')
    val_loader = data.DataLoader(
        val_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads, 
        shuffle=False,
        pin_memory=False)

    return train_loader, val_loader, train_set_stats