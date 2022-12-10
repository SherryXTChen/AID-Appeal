import os
import sys
import glob
from PIL import Image, ImageOps
import math
import numpy as np
import cv2
import shutil
import random
random.seed(0)

appeal_tags = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
unappeal_tags = ['disgusting_food', 'food_with_bugs', 'gross_food', 'moldy_food', 'rotten_food', 'rotten_fruit', 'spoiled_food', 'stale_food', 'ugly_food', 'ugly_fruit', 'unappetizing_food']


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


def visualize():
    in_dir = sys.argv[2]
    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]
    base_name = os.path.basename(in_dir)
    
    out_dir = os.path.join('ranks_visualization', base_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_img_dir = os.path.join(out_dir, 'images')
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    out_path = os.path.join(out_dir, 'index.html')

    img_lst =  glob.glob(f'{in_dir}/*.jpg', recursive=True)
    img_lst.sort()

    
    h = 100
    w = 100
    bound_size = 5
    samples = 10
    loc = 'left'

    with open(out_path, "w") as outputfile:
        for i in range(0, 100, 10):
            lst = img_lst[int(len(img_lst) * i / 100):int(len(img_lst) * (i+10) / 100)]
            if len(lst) > samples:
                lst = lst[:samples]

            for img_path in lst:
                img = Image.open(img_path).convert('RGB').resize((w-2*bound_size, h-2*bound_size))
                img_name = os.path.basename(img_path)

                # decide image bound color
                color = None
                if any(t in img_name for t in appeal_tags):
                    color = 'green'
                else:
                    assert any(t in img_name for t in unappeal_tags), img_path
                    color = 'red'
                # print(img_path, color)

                img = ImageOps.expand(img,border=bound_size,fill=color)
                img.save(os.path.join(out_img_dir, os.path.basename(img_path)))
            
            lst = [os.path.join('images', os.path.basename(f)) for f in lst]

            stats = generate_stats(loc, i, i+10)
            outputfile.write(stats)

            line = generate_image_html(lst, loc, h, w)
            outputfile.write(line)

    os.system(f'open {out_path}')



def user_study():
    in_dir = sys.argv[2]
    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]
    base_name = os.path.basename(in_dir)
    
    out_dir = os.path.join('user_study', base_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_img_dir = os.path.join(out_dir, 'images')
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    out_path = os.path.join(out_dir, 'answer.txt')
    out = open(out_path, 'w')

    img_lst =  glob.glob(f'{in_dir}/*.jpg', recursive=True)
    random.shuffle(img_lst)

    num_pairs = 10
    accu = 0

    for i in range(num_pairs):
        f1 = random.choice(img_lst)
        rank1 = int(os.path.basename(f1).split('_')[0].replace('rank=', ''))
        
        f2 = random.choice(img_lst)
        rank2 = int(os.path.basename(f2).split('_')[0].replace('rank=', ''))
        
        while abs(rank1 - rank2) < 50:
            f2 = random.choice(img_lst)
            rank2 = int(os.path.basename(f2).split('_')[0].replace('rank=', ''))

        line = [i+1]
        if rank1 < rank2:
            line.append(1)
        else:
            line.append(2)

        img1 = Image.open(f1).convert('RGB').resize((224, 224))
        img2 = Image.open(f2).convert('RGB').resize((224, 224))

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


if __name__ == '__main__':
    eval(sys.argv[1] + '()')