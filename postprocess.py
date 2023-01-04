import os
import sys
import glob
from PIL import Image, ImageOps
import itertools
import numpy as np
import cv2
import shutil
import datetime
import random
random.seed(0)

appeal_tags = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
unappeal_tags = ['disgusting', 'gross', 'moldy', 'rotten', 'spoiled', 'stale', 'unappetizing', 'undercooked', 'raw']

def generate_stats(loc, min_rank, max_rank):
    # if index_range:
    #     index_stats = f'{index_range[0]}-{index_range[1]} most popular:'
    stats = f'<div style="clear: {loc};">\n\t<p> ranking: top {min_rank}% - {max_rank}% </p>\n</div>\n'
    return stats


def generate_image_html(lst, loc, h, w):
    line = f'<div style="clear: {loc};">'
    for f in lst:
        line += f'\n\t<p style="float: {loc};"><img src="{f}" height="{h}" width="{w}" border="0px"></p>'
    line += '\n'
    return line


def symlink_images():
    dir = sys.argv[2]
    root = sys.argv[3]

    lines = open(os.path.join(dir, 'lists.txt')).readlines()
    out_dir = os.path.join(dir, 'symlinks')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    score_lst = [l.strip().split('\t')[0] for l in lines]
    score_lst = [float(f[f.index('score=')+len('score='):].split('_')[0]) for f in score_lst]
    score_lst = list(zip(score_lst, lines))
    score_lst = sorted(score_lst, key=lambda x: -x[0])

    rank = 0
    for l in score_lst:
        sym_f, in_f = tuple(l[1].strip().split('\t'))
        in_f = os.path.join(root, in_f)
        
        sym_f = os.path.basename(sym_f)
        rank += 1
        
        out_f = os.path.join(out_dir, f'rank={str(rank).zfill(10)}_' + sym_f)
        os.symlink(in_f, out_f)


def generate_html_adobe_food_101():
    in_dir = sys.argv[2]
    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]

    food_101_lst = [['adobe_stock-' + f] for f in appeal_tags]

    for filter_tags in food_101_lst:
        out_path = generate_html_by_food_type(in_dir, filter_tags, os.path.join('adobe_stock-food-101', filter_tags[0].replace('adobe_stock-', '')))


def generate_html_food_101():
    in_dir = sys.argv[2]
    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]

    food_101_lst = [['_' + f] for f in appeal_tags]

    for filter_tags in food_101_lst:
        out_path = generate_html_by_food_type(in_dir, filter_tags, os.path.join('food-101', filter_tags[0][1:]))


def generate_html_all_food_101():
    in_dir = sys.argv[2]
    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]

    food_101_lst = [['adobe_stock-' + f, '_' + f] for f in appeal_tags]

    for filter_tags in food_101_lst:
        out_path = generate_html_by_food_type(in_dir, filter_tags, os.path.join('appealing', filter_tags[-1][1:]))


def generate_html_all():
    in_dir = sys.argv[2]
    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]
    out_path = generate_html_by_food_type(in_dir, 'all', [])
    os.system(f'open {out_path}')


def generate_html_by_food_type(in_dir, filter_tags, name):
    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]
    out_dir = os.path.join(os.path.dirname(in_dir), 'webpages', name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_img_dir = os.path.join(out_dir, 'images')
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    out_path = os.path.join(out_dir, 'index.html')

    img_lst =  glob.glob(f'{in_dir}/*.jpg', recursive=True)
    img_lst = sorted(img_lst, key=lambda f: -float(f[f.index('score=')+len('score='):].split('_')[0]))
    if len(filter_tags):
        img_lst = [f for f in img_lst if any(t in f for t in filter_tags)]
        img_lst = [f for f in img_lst if not any(t in f for t in unappeal_tags)]
        skip = 1
    else:
        skip = 100
    
    h = 200
    w = 200
    bound_size = 5
    
    loc = 'left'

    appeal_count = 0
    unappeal_count = 0
    count = 0
    with open(out_path, "w") as outputfile:
        for i in range(0, 100, 10):
            lst = img_lst[int(len(img_lst) * i / 100):int(len(img_lst) * (i+10) / 100)]
            lst = lst[::skip]

            for img_path in lst:
                count += 1
                print(f'{count}/{len(img_lst)}')
                img = Image.open(img_path).convert('RGB').resize((w-2*bound_size, h-2*bound_size))
                img_name = os.path.basename(img_path)

                # decide image bound color
                color = None
                if any(t in img_name for t in unappeal_tags):
                    color = 'red'
                    unappeal_count += 1
                else:
                    color = 'green'
                    appeal_count += 1

                img = ImageOps.expand(img,border=bound_size,fill=color)
                img.save(os.path.join(out_img_dir, os.path.basename(img_path)))
            
            lst = [os.path.join('images', os.path.basename(f)) for f in lst]

            stats = generate_stats(loc, i, i+10)
            outputfile.write(stats)

            line = generate_image_html(lst, loc, h, w)
            outputfile.write(line)

    print('appeal_count', appeal_count)
    print('unappeal_count', unappeal_count)
    return out_path

def resize_image(f):
    img = Image.open(f).convert('RGB')
    max_side = 224
    w, h = img.size
    if w >= h:
        w_new = max_side
        h_new = max_side * h // w
        x = (max_side - h_new) //2
        y = 0
    else:
        h_new = max_side
        w_new = max_side * w // h
        x = 0
        y = (max_side - w_new) // 2
    img = img.resize((w_new, h_new))

    canvas = Image.new('RGB', (max_side, max_side))
    canvas.paste(img, (y, x))
    return canvas


def user_study():
    in_dir = sys.argv[2]
    num_pairs = int(sys.argv[3])

    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]
    
    now = datetime.datetime.now()
    filename = now.strftime("%m_%d_%Y_%H_%M_%S.txt")

    out_dir = os.path.join(os.path.dirname(in_dir), 'user_study', filename)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_img_dir = os.path.join(out_dir, 'images')
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    out_path = os.path.join(out_dir, 'answers.csv')
    out = open(out_path, 'w')

    img_lst =  glob.glob(f'{in_dir}/*.jpg', recursive=True)
    total = len(img_lst)
    food_101_lst = [['adobe_stock-' + f, '_' + f] for f in appeal_tags]
    food_101_lst = list(itertools.chain.from_iterable(food_101_lst))
    img_lst = [f for f in img_lst if any(t in f for t in food_101_lst)]
    img_lst = [f for f in img_lst if not any(t in f for t in unappeal_tags)]

    accu = 0

    out.write('\t'.join(['index', 'image1', 'image2', 'image1 score', 'image2 score', 'image1 rank by percentage', 'image2 rank by percentage', 'more appealing by model', 'more appealing by user']) + '\n')

    for i in range(num_pairs):
        f1 = random.choice(img_lst)
        rank1 = int(os.path.basename(f1).split('_')[0].replace('rank=', ''))
        score1 = float(os.path.basename(f1).split('_')[1].replace('score=', ''))

        tag = None
        for t in appeal_tags:
            if 'adobe_stock-' + t in f1 or '_' + t in f1:
                tag = t
                break
        assert tag, f1

        f_img_lst = [f for f in img_lst if tag in f]
        
        f2 = random.choice(f_img_lst)
        rank2 = int(os.path.basename(f2).split('_')[0].replace('rank=', ''))
        while abs(rank1-rank2) / total < 0.2:
            f2 = random.choice(f_img_lst)
            rank2 = float(os.path.basename(f2).split('_')[0].replace('rank=', ''))
        
        score2 = float(os.path.basename(f2).split('_')[1].replace('score=', ''))

        line = [i+1, os.path.basename(f1), os.path.basename(f2), score1, score2, rank1/total, rank2/total]
        if score1 > score2:
            line.append(1)
        else:
            line.append(2)

        img1 = resize_image(f1)
        img2 = resize_image(f2)

        img = np.concatenate([np.array(img1, np.uint8), np.array(img2, np.uint8)], axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(f'{i+1}/{num_pairs}: which do you prefer?', img)
        
        while True:
            k = cv2.waitKey(0)
            if k == 2: # img1 better
                line.append(1)
                break
            elif k == 3: # img2 better
                line.append(2)
                break
            else:
                print('Select left or right image')

        shutil.copy(f1, os.path.join(out_img_dir, f'{i+1}_1.jpg'))
        shutil.copy(f2, os.path.join(out_img_dir, f'{i+1}_2.jpg'))

        if line[-1] == line[-2]:
            accu += 1
        out.write('\t'.join([str(x) for x in line]) + '\n')

    out.close()
    print('model accuracy:', accu / num_pairs)


def get_textural_inversion_sample():
    in_dir = sys.argv[2]
    out_dir = sys.argv[3]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    img_lst =  glob.glob(f'{in_dir}/*.jpg', recursive=True)
    img_lst = sorted(img_lst, key=lambda f: float(f[f.index('score=')+len('score='):].split('_')[0]))

    appeal_lst = None
    unappeal_lst = None

    for i in range(len(img_lst)):
        img_path =  img_lst[i]
        img_name = os.path.basename(img_path)
        if any(t in img_name for t in unappeal_tags):
            appeal_lst = img_lst[:i]
            unappeal_lst = img_lst[i:]
            print('appeal rank threshold', i+1)
            break

    # appeal_score 10 - 5.1
    appeal_score = [float(os.path.basename(f).split('_')[1].replace('score=', ''))
        for f in appeal_lst]
    # unappeal_score 4.9 - 0
    unappeal_score = [float(os.path.basename(f).split('_')[1].replace('score=', ''))
        for f in unappeal_lst]

    print(appeal_score[0], appeal_score[-1], unappeal_score[0], unappeal_score[-1])

    appeal_score_norm_func = lambda x: (x - appeal_score[-1]) / (appeal_score[0] - appeal_score[-1]) * (10 - 5.1) + 5.1
    unappeal_score_norm_func =  lambda x: (x - unappeal_score[-1]) / (unappeal_score[0] - unappeal_score[-1]) * (4.9 - 0)
   
    appeal_score = [appeal_score_norm_func(x) for x in appeal_score]
    unappeal_score = [unappeal_score_norm_func(x) for x in unappeal_score]
    score_lst = appeal_score + unappeal_score

    score_dict = {
        (10, 8) : 'excellent',
        (8, 6) : 'good', 
        (6, 4) : 'neural',
        (4, 2) : 'bad',
        (2, -1) : 'horrible',
    }

    for k, v in score_dict.items():
        out_dir_v = os.path.join(out_dir, v)
        if not os.path.isdir(out_dir_v):
            os.mkdir(out_dir_v)

    for i in range(len(img_lst)):
        print(f'{i+1}/{len(img_lst)}')
        img_path = img_lst[i]
        img_path_lst = os.path.basename(img_path).split('_')
        score = score_lst[i]
        img_path_lst.insert(2, f'normscore={score}')
        img_path = '_'.join(img_path_lst)

        for k, v in score_dict.items():
            if k[0] >= score > k[1]:
                img_path = os.path.join(out_dir, v, img_path)
                shutil.move(img_lst[i], img_path)
                break
        print(img_path)


if __name__ == '__main__':
    eval(sys.argv[1] + '()')