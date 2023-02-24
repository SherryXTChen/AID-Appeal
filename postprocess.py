import os
import sys
import glob
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import cv2
import shutil
import random
random.seed(0)


def generate_stats(loc, min_rank, max_rank):
    # if index_range:
    #     index_stats = f'{index_range[0]}-{index_range[1]} most popular:'
    stats = f'<div style="clear: {loc};">\n\t<p> ranking: {min_rank}% - {max_rank}% </p>\n</div>\n'
    return stats


def generate_image_html(lst, loc, h, w):
    line = f'<div style="clear: {loc};">'
    for f in lst:
        line += f'\n\t<p style="float: {loc};"><img src="{f}" height="{h}" width="{w}" border="0px"></p>'
    line += '\n'
    return line


def rank_images(file_path, root='/Users/xiaotongchen/Documents/datasets'):
    lines = open(file_path).readlines()
    data_list = []
    for l in lines:
        in_f, gt_score, pred_score = tuple(l.strip().split('\t'))
        in_f = os.path.join(root, in_f)
        gt_score = float(gt_score)
        pred_score = float(pred_score)
        data_list.append((in_f, gt_score, pred_score))
    data_list = sorted(data_list, key=lambda x:-x[-1])

    return data_list


def generate_html_by_type():
    file_path = sys.argv[2]
    if 'food' in file_path:
        type_list = [
            'burger', 'cake', 'cookie', 'fried_rice', 'ice_cream', 'pizza', 'ramen', 'chicken', 'salad', 'steak'
        ]
        additional_type_list = ['moldy_food', 'burnt_food']
    elif 'room' in file_path:    
        type_list = [
            'bathroom', 'bedroom', 'kitchen', 'living_room'
        ]
        additional_type_list = ['dirty_room']

    for t in type_list:
        out_path = generate_html_helper(file_path, [t] + additional_type_list)
        if out_path:
            os.system(f'open {out_path}')

def generate_html_all():
    file_path = sys.argv[2]
    out_path = generate_html_helper(file_path, [])
    os.system(f'open {out_path}')


def generate_html_helper(file_path, filter_list):
    data_list = rank_images(file_path)
    name = 'all' 
    if filter_list:
        data_list = [x for x in data_list if (any(f in x[0] for f in filter_list))]
        name = filter_list[0]
    
    out_dir = os.path.splitext(file_path)[0]
    if not data_list:
        return

    image_out_dir = os.path.join(out_dir, 'images')
    os.makedirs(image_out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f'{name}_index.html')
    bound_size = 5
    loc = 'left'
    color_list = ['red', 'blue', 'green']

    with open(out_path, "w") as outputfile:
        for i in range(0, 100, 10):
            in_list = data_list[int(len(data_list) * i / 100):int(len(data_list) * (i+10) / 100)]
            out_list = []
            font = ImageFont.truetype("arial.ttf", 15)            
            for image_path, gt_score, pred_score in in_list:
                image_out_path = os.path.basename(os.path.dirname(image_path)) + '_' + os.path.splitext(os.path.basename(image_path))[0]
                image_out_path = os.path.join(image_out_dir, image_out_path[:min(len(image_out_path), 100)] + '.png')
                image_out_path_html = os.path.join(os.path.basename(image_out_dir), os.path.basename(image_out_path))

                if os.path.exists(image_out_path):
                    print('image exists')
                    out_list.append(image_out_path_html)
                    continue

                image = Image.open(image_path).convert('RGB')
                
                if int(gt_score) not in [-1, 0, 1]:
                    tag = f'gt:{gt_score}, pred:{round(pred_score,2)}'
                    if gt_score <= 4:
                        gt_score = -1
                    elif 4 < gt_score <= 7:
                        gt_score = 0
                    else:
                        gt_score = 1
                else:
                    tag = f'pred:{round(pred_score,2)}'
        
                color = color_list[int(gt_score) + 1]
                image = ImageOps.expand(image,border=bound_size,fill=color)
                draw = ImageDraw.Draw(image)
                position = (10,10)
                bbox = draw.textbbox(position, tag, font=font)
                draw.rectangle(bbox, fill='black')
                draw.text(position, tag, font=font, fill='white')

                image.save(image_out_path)
                out_list.append(image_out_path_html)

            stats = generate_stats(loc, i, i+10)
            outputfile.write(stats)

            line = generate_image_html(out_list, loc, h=512, w=512)
            outputfile.write(line)

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
    user_index = sys.argv[3]
    num_pairs = int(sys.argv[4])

    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]
    
    out_dir = os.path.join(os.path.dirname(in_dir), 'user_study', user_index)
    # out_img_dir = os.path.join(out_dir, 'images')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'answers.txt')
    out = open(out_path, 'w')

    dir_list = glob.glob(f'{in_dir}/*/', recursive=True)
    # accu = 0
    out.write('\t'.join(['index', 'image1 score', 'image2 score', 'more appealing by user']) + '\n')

    past_samples = []

    for i in range(num_pairs):
        dir = random.choice(dir_list)
        image_list = glob.glob(f'{dir}/*.jp*g', recursive=True) + glob.glob(f'{dir}/*.jp*g', recursive=True)
        
        f1, f2 = tuple(random.sample(image_list, 2))
        while (f1, f2) in past_samples or (f2, f1) in past_samples:
            f1, f2 = tuple(random.sample(image_list, 2))
        past_samples.append((f1, f2))

        score1 = float(f1[f1.index('pred=')+len('pred='):].split('_')[0])
        score2 = float(f2[f2.index('pred=')+len('pred='):].split('_')[0])

        line = [i+1, score1, score2]
        # if score1 > score2:
        #     line.append(1)
        # else:
        #     line.append(2)

        img1 = resize_image(f1)
        img2 = resize_image(f2)

        img = np.concatenate([np.array(img1, np.uint8), np.array(img2, np.uint8)], axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(f'{i+1}/{num_pairs}: which one do you like? (1: left, 2: right, 3: both, 4: neither)', img)
        
        while True:
            k = cv2.waitKey(0)
            if k in [49, 50, 51, 52]: # img1 better
                line.append(k-49)
                break
            else:
                print('Select 1-4')

        shutil.copy(f1, f'image_{str(i+1).zfill(4)}_1.jpg')
        shutil.copy(f2, f'image_{str(i+1).zfill(4)}_2.jpg')

        # if line[-1] == line[-2]:
        #     accu += 1
        out.write('\t'.join([str(x) for x in line]) + '\n')

    out.close()
    # print('model accuracy:', accu / num_pairs)


if __name__ == '__main__':
    eval(sys.argv[1] + '()')