import os
import argparse
import glob
from PIL import Image
import numpy as np

import torch
import clip

import models

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

''''
class CLIPVisualizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        pretrained_model = VisionTransformer(
            input_resolution=224,
            patch_size=14,
            width=1024,
            layers=24,
            heads=16,
            output_dim=768,
        )

        pretrained_model.load_state_dict(model.pretrained_model.state_dict())
        model.pretrained_model = pretrained_model
        self.model = model
        self.heads = 16

        self.attentions = []
        for b in self.model.pretrained_model.transformer.resblocks:
            b.attn.register_forward_hook(self.get_feature())

    def get_feature(self):
        def func(module, input, output):
            self.attentions.append(output[1])
        return func

    def forward(self, x):
        self.attentions = []
        score = self.model(x), 
        return score, self.attentions
'''

@torch.no_grad()
def get_attention(opt):
    out_dir = os.path.join(opt.out_dir, opt.name)
    os.makedirs(out_dir, exist_ok=True)

    model = models.CLIPScorer(opt)
    ckpt_path = os.path.join(opt.ckpt_dir, opt.name, 'last-v1.ckpt')
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval()
    model.to(DEVICE)

    image_list = glob.glob(f'{opt.input_dir}/*.jpg', recursive=True)
    _, transform = clip.load('ViT-L/14', device='cpu')

    for image_path in image_list:
        name = os.path.basename(os.path.splitext(image_path)[0])

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = Image.fromarray(image[30:-30]).resize((opt.image_size, opt.image_size))

        input = transform(image).to(DEVICE).unsqueeze(0)
        score = model(input)[0].item()

        step_size = 16
        window_size = opt.image_size // 4
        score_change_sum = np.zeros((opt.image_size, opt.image_size))
        score_change_count = np.zeros((opt.image_size, opt.image_size))

        for i in range(-window_size, opt.image_size, step_size):
            for j in range(-window_size, opt.image_size, step_size):
                x1 = max(i, 0)
                x2 = min(i + window_size, opt.image_size)
                y1 = max(j, 0)
                y2 = min(j + window_size, opt.image_size)
                if x1 == x2 or y1 == y2:
                    continue

                # get patch in the slide window
                imageij = np.array(image)
                imageij = imageij[x1:x2,y1:y2]
                imageij = Image.fromarray(imageij)
                inputij = transform(imageij).to(DEVICE).unsqueeze(0)

                # get appeal score of patch
                scoreij = model(inputij)[0].item()
                score_change_sum[x1:x2, y1:y2] += scoreij
                score_change_count[x1:x2, y1:y2] += 1

        # pixel appeal score = average of patch appeal score
        score_change = score_change_sum / score_change_count
        score_change -= np.min(score_change)
        if np.max(score_change) > 0:
            score_change /= np.max(score_change)

        # inverse: brighter color --> more unappealing
        score_change = 255 - np.uint8(score_change *255)
        Image.fromarray(score_change).save(os.path.join(out_dir, f'{name}.png'))

        image.save(os.path.join(out_dir, f'{name}_{round(score, 2)}.jpg' ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='name of experiment')
    parser.add_argument('--input_dir', type=str, required=True, help='path to input images')
    parser.add_argument('--out_dir', type=str, default='attns', help='path to save ouputs')
    parser.add_argument('--ckpt_dir', type=str, default='ckpts', help='path to training results')
    parser.add_argument('--loss_type', type=str, required=True, choices=['singular', 'pair', 'triplet'], help='loss type')
    parser.add_argument('--image_size', type=int, default=512, help='image size')
    opt = parser.parse_args()
    get_attention(opt)