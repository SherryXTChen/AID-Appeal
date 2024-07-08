import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import os
import glob
from PIL import Image, ImageOps
import shutil
import matplotlib.pyplot as plt
import spacy
import numpy as np
from collections import defaultdict
import cv2
from tqdm import tqdm

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
nlp = spacy.load('en_core_web_md')


def if_food(word):
    if word.endswith('cake') or word.endswith('cakes'): 
        return 1
    syns = wn.synsets(str(word), pos = wn.NOUN)
    type_list = list(set([syn.lexname() for syn in syns]))
    if 'noun.artifact' in type_list:
        return 0
    if 'noun.food' in type_list:
        return 1
    return 0


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp


def imshow(image):
    plt.axis('off')
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def filter_by_mask():
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    in_dir = './food/2_preprocessed/images'
    out_image_dir = './food/3_filtered/images'
    out_caption_dir = './food/3_filtered/captions'
    out_mask_dir = './food/3_filtered/masks'

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_caption_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    image_list = glob.glob(f'{in_dir}/*.png', recursive=True)
    image_list.sort()
    image_list = image_list[-len(image_list)//4:]

    prompt_set = set()
    for image_path in tqdm(image_list):
        name = os.path.basename(os.path.splitext(image_path)[0]).split('-')[-1]
        out_image_path = os.path.join(out_image_dir, f'{name}.jpg')
        out_caption_path = os.path.join(out_caption_dir, f'{name}.txt')

        if os.path.exists(out_image_path):
            continue

        caption_path = os.path.splitext(image_path)[0] + '.txt'
        prompt = open(caption_path).read().strip()
        image = Image.open(image_path).convert('RGB')
        food_type = os.path.basename(os.path.splitext(image_path)[0])
        food_type = ' '.join(food_type.split('-')[-1].split('_')[:2])

        doc = nlp(prompt)
        phrases = [str(noun_chunk) for noun_chunk in doc.noun_chunks]
        phrases = [p for p in phrases if 'orange background' not in p]
        phrases = [p for p in phrases if (any(if_food(w) for w in p.split()))]

        if len(phrases) == 0:
            phrases += [food_type]

        if len(phrases) == 1:
            phrases += phrases

        inputs = processor(text=phrases, images=[image] * len(phrases), padding="max_length", return_tensors="pt")
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        preds = outputs.logits.unsqueeze(1)
        preds = torch.nn.functional.interpolate(preds, size=np.array(image).shape[:2])

        out_final = np.zeros(np.array(image).shape)
        for i in range(len(phrases)):
            out = torch.sigmoid(preds[i][0]).detach().cpu().numpy()
            if np.min(out) == np.max(out):
                continue
            out -= np.min(out)
            out /= np.max(out)
            out = np.uint8(out * 255)
            out = np.tile(out[...,None], (1, 1, 3))
            out_final += out
        out_final = np.clip(out_final, 0, 255).astype(np.uint8)

        if np.mean(out_final) / 255 >= 0.3:
            out_mask_path = os.path.join(out_mask_dir, f'{name}_count={len(phrases)}.png')
            shutil.copy(caption_path, out_caption_path)
            shutil.copy(image_path, out_image_path)
            Image.fromarray(out_final).save(out_mask_path)

def sample_by_name():
    for food in ['burger', 'cake', 'cookie', 'rice', 'yogurt', 'pizza', 'pasta', 'chicken', 'salad', 'steak']:
        image_list = []
        for adj in ['burnt', 'moldy', 'rotten', 'disgusting']:
            image_list += glob.glob(f'./food/3_filtered/images/{adj}_{food}_*.jpg')
        image_list.sort()
        if len(image_list) > 50:
            image_list = image_list[::len(image_list) // 50]
            image_list = image_list[:50]

        image_list_2 = glob.glob(f'./food/3_filtered/images/delicious_{food}_*.jpg')
        print(len(image_list), len(image_list_2))
        image_list_2 = image_list_2[::len(image_list_2) // (100 - len(image_list))]
        image_list += image_list_2
        if len(image_list) > 100:
            image_list = image_list[:100]

        assert len(image_list) == 100, len(image_list)

        for image_path in image_list:
            image_path_out = image_path.replace('/3_filtered/', '/4_appeal_interpolation/')
            mask_path_in = glob.glob(f'{os.path.splitext(image_path)[0].replace("/images/", "/masks/")}*', recursive=True)[0]
            mask_path_out = mask_path_in.replace('/3_filtered/', '/4_appeal_interpolation/')
            caption_path_in = os.path.splitext(image_path)[0].replace('/images/', '/captions/') + '.txt'
            caption_path_out = caption_path_in.replace('/3_filtered/', '/4_appeal_interpolation/')
            shutil.move(image_path, image_path_out)
            shutil.move(mask_path_in, mask_path_out)
            shutil.move(caption_path_in, caption_path_out)

if __name__ == '__main__':
    filter_by_mask()
