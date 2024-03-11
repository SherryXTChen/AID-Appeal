import os
import argparse
from PIL import Image
import numpy as np

import torch
import clip

import models


# get appeal heatmap M_D^H
@torch.no_grad()
def get_appeal_score(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create output folder
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # load model
    model = models.CLIPScorer(args)
    state = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval().to(device)

    # get clip preprocessor
    _, clip_preprocess = clip.load("ViT-L/14", device="cpu")

    # get input image
    name = os.path.basename(os.path.splitext(args.input_path)[0])
    image = Image.open(args.input_path).convert("RGB")
    image.save(os.path.join(out_dir, f"{name}_input.png" ))

    ori_w, ori_h = image.size
    image_size = 512
    image = image.resize((image_size, image_size))

    # get appeal score
    input = clip_preprocess(image).unsqueeze(0).to(device)
    score = model(input)[0].item()
    with open(os.path.join(out_dir, f"{name}-_score.txt"), "w") as f:
        f.write(f"{score}\n")

    if not args.get_appeal_heatmap:
        return

    # define window size and step size parameter for appeal heatmap calculation
    step_size = 16
    window_size = 128
    score_change_sum = np.zeros((image_size, image_size))
    score_change_count = np.zeros((image_size, image_size))

    for i in range(-window_size, image_size, step_size):
        for j in range(-window_size, image_size, step_size):
            # get current window
            x1 = max(i, 0)
            x2 = min(i + window_size, image_size)
            y1 = max(j, 0)
            y2 = min(j + window_size, image_size)
            if x1 == x2 or y1 == y2:
                continue

            # get patch in the current window
            imageij = np.array(image)
            imageij = imageij[x1:x2,y1:y2]
            imageij = Image.fromarray(imageij)
            inputij = clip_preprocess(imageij).unsqueeze(0).to(device)

            # get appeal score of patch
            scoreij = model(inputij)[0].item()
            score_change_sum[x1:x2, y1:y2] += scoreij
            score_change_count[x1:x2, y1:y2] += 1

    # pixel appeal score = average of patch appeal score
    score_change = score_change_sum / score_change_count
    score_change -= np.min(score_change)
    if np.max(score_change) > 0:
        score_change /= np.max(score_change)

    # inverst heatmap: brighter color --> more unappealing
    score_change = 255 - np.uint8(score_change * 255)
    Image.fromarray(score_change).resize((ori_w, ori_h)).save(os.path.join(out_dir, name + "_mask.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="path to appeal score estimator checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="path to input image")
    parser.add_argument("--out_dir", type=str, default="outputs", help="path to save ouputs")
    parser.add_argument("--get_appeal_heatmap", action="store_true", help="whether to get appeal heatmap")
    args = parser.parse_args()

    get_appeal_score(args)