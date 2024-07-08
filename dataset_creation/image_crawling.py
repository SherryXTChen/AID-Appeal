from selenium import webdriver 
from bs4 import BeautifulSoup
from collections import defaultdict
import os
import sys
import time
import glob
import shutil
import json
import csv
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm

FILTER_LIST = ['chili', 'chilis', 'dry', 'dried', 'pepper', 'peppers', 
    'candle', 'candles', 'birthday', 'cheese', 'fire', 'flame', 
    'flames', 'burning', 'cooking', 'raw', 'collage']

def get_adobe_links_helper(keyword, parent_dir, image_limit=None):
    keyword = keyword.replace(' ', '+')

    out_dir = os.path.join(parent_dir, keyword.replace('+', '_'))
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'links.csv')
    if os.path.exists(out_path):
        return
    out = open(out_path, "w")
    csv_writer = csv.writer(out)
    csv_writer.writerow(['url', 'caption'])

    url = f'https://stock.adobe.com/search?filters%5Bcontent_type%3Aphoto%5D=1&filters%5Bcontent_type%3Aimage%5D=1&filters%5Breleases%3Ais_exclude%5D=1&k={keyword}&order=relevance&safe_search=1&limit=100&search_page=1&search_type=pagination&acp=&aco={keyword}&get_facets=0'
    driver = webdriver.Firefox(log_path='NUL')

    driver.get(url)
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(1)

    try:
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser').prettify()
        soup = soup[soup.index('"grey dove-text text-sregular">') + len('"grey dove-text text-sregular">'):].strip()
        soup = soup[:soup.index(' page')].replace('of ', '')
        page_limit = int(soup)
    except:
        page_limit = 1

    links = {}
    for i in range(1, page_limit+1):
        url_i = url.replace('&search_page=1', f'&search_page={i}', 1)
        driver.get(url_i)

        height = driver.execute_script("return document.documentElement.scrollHeight")
        split = 20
        for j in range(split):
            h = height//split
            driver.execute_script(f"window.scrollTo({h*j}, {h*(j+1)});")
            time.sleep(1)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        elements = soup.find_all('img')
        for e in elements:
            link = e.get('src')
            if 'ftcdn.net' in link and '/360_F_' in link:
                caption = e.get('alt').lower()
                if any(w in caption.split() for w in FILTER_LIST):
                    continue

                if link not in links:
                    csv_writer.writerow([link, caption])
                    links[link] = caption

        print(f'after {i}/{page_limit} page, number of images: {len(links)}')
        if image_limit != None:
            if len(links) >= image_limit:
                break
    driver.quit()

def get_shutterstock_links(keyword, parent_dir, image_limit=None):
    out_dir = os.path.join(parent_dir, keyword.replace(' ', '_'))
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'links.csv')
    if os.path.exists(out_path):
        return
    out = open(out_path, "w")
    csv_writer = csv.writer(out)
    csv_writer.writerow(['url', 'caption'])
    links = {}

    url = f'https://www.shutterstock.com/search/{keyword.replace(" ", "-")}?c3apidt=71700000027388020&cr=c&gclid=Cj0KCQiAgOefBhDgARIsAMhqXA49wjUwQhMUEYCPiGqya9swPkTWtResx8qoi4WR86F-ZugFJdKYfkMaAjQOEALw_wcB&gclsrc=aw.ds&pl=PPC_GOO_US_DSA-644343210861&mreleased=false&image_type=photo&page='
    driver = webdriver.Firefox(log_path='NUL')
    driver.get(url + '1')
    for _ in range(10):
        try:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(1)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            span = soup.find_all('span', {'class' : 'MuiTypography-root MuiTypography-subtitle2 mui-1a6qx3c-totalPages'})[0]
            page_limit = int(span.text.split()[1].replace(',', ''))
            break
        except:
            page_limit = 1

    for i in range(1, page_limit+1):
        driver.get(url + str(i))
        height = driver.execute_script("return document.documentElement.scrollHeight")
        split = 20
        for j in range(split):
            h = height//split
            driver.execute_script(f"window.scrollTo({h*j}, {h*(j+1)});")
            time.sleep(1)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        elements = soup.find_all('img')
        for e in elements:
            link = e.get('src')
            if 'https://www.shutterstock.com/image-photo/' in link:
                caption = e.get('alt').lower()
                if any(w in caption.replace(',','').replace('.', '').split() for w in FILTER_LIST):
                    continue

                if link not in links:
                    csv_writer.writerow([link, caption])
                    links[link] = caption

        print(f'after {i}/{page_limit} page, number of images: {len(links)}')
        if image_limit != None:
            if len(links) >= image_limit:
                break
    driver.quit()

def download_pipeline():
    # get links
    parent_dir = sys.argv[2]

    # keyword_list = ['burger', 'cake', 'cookie', 'rice', 'yogurt', 'pizza', 'pasta', 'chicken', 'salad', 'steak']
    # keyword_list_unappealing = ['burnt ' + keyword for keyword in keyword_list] + \
    #     ['moldy ' + keyword for keyword in keyword_list] + \
    #     ['rotten ' + keyword for keyword in keyword_list] + \
    #     ['disgusting ' + keyword for keyword in keyword_list]
    # keyword_list_appealing = ['delicious ' + keyword for keyword in keyword_list]
    # keyword_list_additional = ['burnt', 'moldy', 'rotten', 'disgusting', 'delicious']
    # keyword_list_additional = [word + ' food' for word in keyword_list_additional]

    keyword_list = ['living room', 'kitchen', 'bedroom', 'bathroom', 'room']
    keyword_list_unappealing = ['dirty ' + k for k in keyword_list] + ['abandoned ' + k for k in keyword_list]
    keyword_list_appealing = [k + ' interior' for k in keyword_list]

    for keyword in keyword_list_unappealing :
        get_shutterstock_links(keyword, parent_dir, image_limit=1e5)
    for keyword in keyword_list_appealing:
        get_shutterstock_links(keyword, parent_dir, image_limit=2e4)

    # # remove duplicates
    d_lst = glob.glob(f'{parent_dir}/*/', recursive=True)
    d_lst.sort()

    # download images
    for d in d_lst:
        image_list = glob.glob(f'{d}/*.jpg', recursive=True)
        if len(image_list) > 0:
            continue
        command = f'img2dataset --url_list "{d}/links.csv" --output_folder {d} --input_format "csv" --url_col "url" --caption_col "caption" --processes_count 16 --thread_count 64 --encode_quality 100 --resize_mode no'
        os.system(command)

    # rename images
    f_lst = glob.glob(f'{parent_dir}/*/*/*.jpg', recursive=True)

    for in_f in f_lst:
        json_f = os.path.splitext(in_f)[0] + '.json'
        out_f = os.path.join(os.path.dirname(os.path.dirname(in_f)), json.load(open(json_f))['url'].split('/')[-1]) # [-1])
        with open(os.path.splitext(out_f)[0] + '.txt', 'w') as out:
            out.write(json.load(open(json_f))['caption'])
        shutil.move(in_f, out_f)

    remove_lst = glob.glob(f'{parent_dir}/*/*/', recursive=True)
    remove_lst += glob.glob(f'{parent_dir}/*/*.parquet', recursive=True)
    remove_lst += glob.glob(f'{parent_dir}/*/*_stats.json', recursive=True)
    for f in remove_lst:
        os.system(f'rm -rf {f}')
    remove_duplicates()

def remove_duplicates():
    root = sys.argv[2]
    image_list = glob.glob(f'{root}/*/*.jpg', recursive=True)
    image_list.sort()
    image_list_2 = [f for f in image_list if any(w in f for w in ['/kitchen_interior/', '/abandoned_room/', '/dirty_room/', '/room_interior/'])]
    image_list = [f for f in image_list if f not in image_list_2] + image_list_2

    image_dict = defaultdict(list)
    for image_path in image_list:
        image_name = os.path.basename(image_path)
        image_dict[image_name].append(image_path)

    count = 0
    for _, v in image_dict.items():
        if len(v) <= 1:
            continue
        for image_path in v[1:]:
            count += 1
            print('remove', count)
            os.remove(image_path)
            os.remove(os.path.splitext(image_path)[0] + '.txt')

# after download
def rename():
    dir_list = glob.glob('room/1_original/*/', recursive=True)
    dir_list = ['room/1_original/kitchen_interior', 'room/1_original/abandoned_room/', 'room/1_original/dirty_room/', 'room/1_original/room_interior/']

    count = 0
    index = 5 * 20000

    for dir in dir_list:
        count += 1
        print(count, len(dir_list))
        image_list = glob.glob(f'{dir}/*.jpg', recursive=True)
        image_list.sort()
        name = os.path.basename(dir[:-1])
        for i in range(len(image_list)):
            index += 1
            out_dir = f'room/2_preprocessed/images_original_{index//20000}/'
            os.makedirs(out_dir, exist_ok=True)

            image = Image.open(image_list[i]).convert('RGB')
            w, h = image.size
            image = image.crop((0, 0, w, h-20))
            w, h = image.size

            new_size = max(w, h)
            delta_w = new_size - w
            delta_h = new_size - h
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            image = ImageOps.expand(image, padding)

            out_name = os.path.join(out_dir, name + '_' + str(i+1).zfill(10) + '.jpg')
            image.save(out_name)

# after blip
def filter():
    image_list = glob.glob('./room/2_preprocessed/images/*.png', recursive=True)
    out_image_dir = './room/3_filtered/images'
    out_caption_dir = './room/3_filtered/captions'
    out_mask_dir = './room/3_filtered/masks'

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_caption_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for image_path in tqdm(image_list): 
        name = os.path.basename(os.path.splitext(image_path)[0]).split('-')[-1]
        out_image_path = os.path.join(out_image_dir, f'{name}.jpg')
        out_caption_path = os.path.join(out_caption_dir, f'{name}.txt')

        if os.path.exists(out_image_path):
            continue

        caption_path = os.path.splitext(image_path)[0] + '.txt'
        prompt = open(caption_path).read().strip()
        if 'room' not in prompt:
            continue

        image = Image.open(image_path).convert('RGB')
        out_final = Image.fromarray(np.ones(np.array(image).shape, np.uint8) * 255)
        shutil.copy(caption_path, out_caption_path)
        shutil.copy(image_path, out_image_path)
        out_final.save(os.path.join(out_mask_dir, f'{name}_count=1.png'))

def move_file():
    image_list = glob.glob(f'room/2_preprocessed/images_original_*/*.jpg')
    out_dir = 'room/2_preprocessed/images_original'
    os.makedirs(out_dir, exist_ok=True)
    for image_path in image_list:
        shutil.move(image_path, os.path.join(out_dir, os.path.basename(image_path)))

def get_appeal_interpolation():
    out_image_dir = './room/4_appeal_interpolation/images'
    out_caption_dir = './room/4_appeal_interpolation/captions'
    out_mask_dir = './room/4_appeal_interpolation/masks'

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_caption_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for room in ['bedroom', 'bathroom', 'kitchen', 'living_room']:
        image_list = []
        for adj in ['abandoned', 'dirty']:
            image_list += glob.glob(f'./room/3_filtered/images/{adj}_{room}_*.jpg')
        image_list.sort()

        length = 1000 // 4 // 2

        if len(image_list) > length:
            image_list = image_list[::len(image_list) // length]
            image_list = image_list[:length]

        image_list_2 = glob.glob(f'./room/3_filtered/images/{room}_interior_*.jpg')
        print(len(image_list), len(image_list_2))
        image_list_2 = image_list_2[::len(image_list_2) // (2 * length - len(image_list))]
        image_list += image_list_2
        if len(image_list) > 2 * length:
            image_list = image_list[:2*length]

        assert len(image_list) == 2*length, len(image_list)

        for image_path in image_list:
            image_path_out = image_path.replace('/3_filtered/', '/4_appeal_interpolation/')
            mask_path_in = glob.glob(f'{os.path.splitext(image_path)[0].replace("/images/", "/masks/")}*', recursive=True)[0]
            mask_path_out = mask_path_in.replace('/3_filtered/', '/4_appeal_interpolation/')
            caption_path_in = os.path.splitext(image_path)[0].replace('/images/', '/captions/') + '.txt'
            caption_path_out = caption_path_in.replace('/3_filtered/', '/4_appeal_interpolation/')
            shutil.copy(image_path, image_path_out)
            shutil.copy(mask_path_in, mask_path_out)
            shutil.copy(caption_path_in, caption_path_out)


if __name__ == '__main__':
    eval(sys.argv[1] + '()')
