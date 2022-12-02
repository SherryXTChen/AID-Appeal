import os
import sys
import glob

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
    in_dir = sys.argv[1]
    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]
    dir_name = os.path.dirname(in_dir)
    base_name = os.path.basename(in_dir)
    out_path = os.path.join(dir_name, base_name + ' visualize_images.html')

    img_lst =  [os.path.join(base_name, f) for f in os.listdir(in_dir) if f.endswith('.jpg')]
    img_lst.sort()

    
    h = 100
    w = 100
    samples = 10
    loc = 'left'

    with open(out_path, "w") as outputfile:
        for i in range(0, 100, 10):
            lst = img_lst[int(len(img_lst) * i / 100):int(len(img_lst) * (i+10) / 100)]
            if len(lst) > samples:
                lst = lst[:samples]

            stats = generate_stats(loc, i, i+10)
            outputfile.write(stats)

            line = generate_image_html(lst, loc, h, w)
            outputfile.write(line)



if __name__ == '__main__':
    visualize()