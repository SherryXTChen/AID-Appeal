import os
import sys
import glob
import numpy as np
from collections import defaultdict
from PIL import Image, ImageOps, ImageDraw, ImageFont

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


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

    diff = 0
    for l in lines:
        in_f, gt_score, pred_score = tuple(l.strip().split('\t'))
        in_f = os.path.join(root, in_f)
        gt_score = float(gt_score)
        pred_score = float(pred_score)
        data_list.append((in_f, gt_score, pred_score))
        
        diff += abs(gt_score - pred_score) 

    data_list = sorted(data_list, key=lambda x:-x[-1])
    print(diff / len(lines))

    return data_list


def generate_html_all():
    file_path = sys.argv[2]
    out_path = generate_html_helper(file_path, [])
    os.system(f'open {out_path}')


def generate_html_helper(file_path, filter_list):
    data_list = rank_images(file_path)
    data_list = data_list[::100]
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

            line = generate_image_html(out_list, loc, h=100, w=100)
            outputfile.write(line)

    return out_path


def response_stats():
    meta = open('user_study/meta.txt').readlines()
    meta = [l.strip().split() for l in meta]
    meta = {l[0]: (float(l[1]), float(l[2])) for l in meta}

    response_file_list = glob.glob('user_study/responses/*.txt')[27:28]
    out_dir = 'user_study/plots'
    os.makedirs(out_dir, exist_ok=True)

    response_dict_label_all = {'food': defaultdict(list), 'room': defaultdict(list)}
    response_dict_mani_all = {'food': defaultdict(list), 'room': defaultdict(list)}

    for response_file in response_file_list:
        # if 'corrupted' in response_file:
        #     continue
        response_lines = open(response_file).readlines()[1:]
        response_lines = [l.strip().split('\t') for l in response_lines]
        response_lines = [(l[0], l[1], int(l[2])) for l in response_lines]

        response_dict_label = {'food': defaultdict(list), 'room': defaultdict(list)}
        response_dict_mani = {'food': defaultdict(list), 'room': defaultdict(list)}        
        for response in response_lines:
            f1, f2, response_value = response
            if 'food'  in f1:
                type = 'food'
            else:
                assert 'room' in f1, f1
                type = 'room'

            if f1 in meta:
                gt1 = meta[f1][0]
                gt2 = meta[f2][0]
                response_dict_label[type][response_value].append(gt1-gt2)
                response_dict_label_all[type][response_value].append(gt1-gt2)
            else:
                gt1 = 0 if os.path.basename(f1) == '00001.jpg' else 1
                gt2 = 1 - gt1
                response_dict_mani[type][gt1-gt2].append(response_value)
                response_dict_mani_all[type][gt1-gt2].append(response_value)

    for type, dict in response_dict_label_all.items():
        out_name = os.path.join(out_dir, f'Pescatarian_{type}_label.png')
        out_data = [dict[response_value] for response_value in range(1, 5+1)]
        plt.boxplot(out_data, labels=[1, 2, 3, 4, 5])
        plt.xlabel('Response Numbering')
        plt.ylabel('A(Image A) - A(Image B)')
        plt.title(f'{type.capitalize()} Dataset: Appeal Comparison Responses (Pescatarian)')
        plt.tight_layout()
        plt.savefig(out_name)
        plt.clf()

    weight_counts = defaultdict(list)
    labels = ['E pref strongly', 'E pref slightly',
        'N pref', 
        'O pref slightly', 'O pref strongly']
    type_list = []

    for type, dict in response_dict_mani_all.items():
        out_name = os.path.join(out_dir, f'Pescatarian_{type}_enhancement.png')
        out_data = [dict[diff] for diff in [1, -1]]
        out_data_stats_list = []

        for i in range(len(out_data)):
            out_data_i = out_data[i]
            out_data_stats = [len([x for x in out_data_i if x == response_value]) for response_value in [1, 2, 3, 4, 5]]
            if i == 1:
                out_data_stats.reverse()
            out_data_stats_list.append(out_data_stats)
        out_data = [out_data_stats_list[0][i] +  out_data_stats_list[1][i] for i in range(len(out_data_stats_list[0]))]
        out_data = [x / sum(out_data) for x in out_data]

        for i in range(len(out_data)):
            weight_counts[labels[i]].append(out_data[i])
        type_list.append(type)


    for k in weight_counts:
        weight_counts[k] = weight_counts[k][:1]
    labels = ['E pref strongly', 'E pref slightly',
        'N pref', 
        'O pref slightly', 'O pref strongly']
    type_list = ['food']

    x = np.arange(len(type_list))
    width = 0.16
    multiplier = 0
    fig, ax = plt.subplots(figsize=(12, 6))
    
    patterns = ["/", "\\", "-", "|", ""]

    for attribute, measurement in weight_counts.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, [round(x, 4) for x in measurement], width, label=attribute, edgecolor='black', hatch=patterns[multiplier])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Total Number of Responses (%)')
    ax.set_title('Appeal Enhancement Responses (Pescatarian)')
    ax.set_xticks(x + width, type_list)
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper left', ncol=len(labels), prop={'size': 14}) #bbox_to_anchor=(1.3, 0.5), loc="center right")
    out_name = out_name.replace(type, 'type')
    plt.tight_layout()
    plt.savefig(out_name)
    plt.clf()


def plot_participants_stats():
    labels = ['Pescatarian', 'Pescatarian', 'Pescatarian', 'Pescatarian', 'Pescatarian']
    sizes = [1, 1, 22, 1, 3]
    title = 'Dietary Preference'
    out_path = 'dietary.png'
    angle = 0
    plt.pie(sizes, labels=labels, startangle=angle, autopct='%1.1f%%')
    plt.title(f'User Study Participants {title} Distribution')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.clf()



if __name__ == '__main__':
    eval(sys.argv[1] + '()')