import os
import glob
import copy
import random
import numpy as np
from PIL import Image

import gradio as gr
import modules.scripts as scripts
from modules.processing import process_images
from modules.shared import state

"""
Synthetic Appeal Dataset Creation Script for Automatic1111 Stable Diffusion WebUI
"""
class Script(scripts.Script):
    def title(self):
        return "Synthetic appeal dataset creation"

    def ui(self, is_img2img):       
        image_dir = gr.Textbox(label="image directory") 
        mask_dir = gr.Textbox(label="mask directory")
        caption_dir = gr.Textbox(label="caption directory")
        out_dir = gr.Textbox(label="output directory")

        n_interpolations = gr.Slider(minimum=1, maximum=128, step=1, value=1, label="Number of interpolations")
        n_seeds = gr.Slider(minimum=1, maximum=106, step=1, value=1, label="Number of seed iteration per run")

        # add two interpolation prompts
        prompt1 = gr.TextArea(label="List of unappealing embedding, separated by space")
        prompt2 = gr.TextArea(label="List of appealing embedding, separated by space")

        return [image_dir, mask_dir, caption_dir, out_dir, n_interpolations, n_seeds, prompt1, prompt2]


    def run(self, p, image_dir: str, mask_dir: str, caption_dir: str, out_dir: str, \
        n_interpolations: int, n_seeds: int, prompt1: str, prompt2: str,):

        # get appealing/unapepealing text prompt
        unappealing_prompt_list = prompt1.split()
        appealing_prompt_list = prompt2.split()

        # get other parameters
        p.negative_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,"
        p.mask_mode = 0
        p.inpaint_area = 1

        # prepare input and output
        image_list = glob.glob(f"{image_dir}/*.jpg", recursive=True)
        os.makedirs(out_dir, exist_ok=True)

        # if seed is set, n_seed = 1
        if p.seed != -1:
            n_seeds = 1

        # set up jobs and process each image
        state.job_count = len(image_list) * len(unappealing_prompt_list) * len(appealing_prompt_list) * n_interpolations * n_seeds

        for image_path in image_list:
            # prepare input and output
            image = Image.open(image_path)
            caption = open(os.path.join(caption_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")).read().strip()
            dirname = os.path.splitext(os.path.basename(image_path))[0]
            image_out_dir = os.path.join(out_dir, dirname)
            os.makedirs(image_out_dir, exist_ok=True)

            # get seed, generate init_images with different background
            seed_list = random.sample(range(4294967294), n_seeds) if p.seed == -1 else [p.seed]
            init_image_list = [image]
            save_image_list = []

            # get mask and caption
            food_type = os.path.basename(image_path)
            food_type = food_type.split("_")[1:-1]
            food_type = food_type[0]

            # get domain relevancy map
            total_mask_path = glob.glob(f"{mask_dir}/{os.path.splitext(os.path.basename(image_path))[0]}_count=*.png", recursive=True)[0]
            total_mask = Image.open(total_mask_path).resize(image.size)
            total_mask = np.array(total_mask)
            
            # get backgroung map (domain irrelevant)
            background_mask = Image.fromarray(255 - np.array(total_mask))

            # augment image
            for seed in seed_list[1:]:
                copy_p = setup_processing(
                    p=p,
                    prompt="",
                    negative_prompt=p.negative_prompt,
                    init_images=[image],
                    image_mask=background_mask,
                    inpainting_fill=2,
                    denoising_strength=0.75,
                    seed= seed if p.seed == -1 else p.seed
                )

                processed = process_images(copy_p)
                for i, img in enumerate(processed.images):
                    init_image_list.append(img)

            # change image appeal
            for i in range(len(seed_list)):
                init_image = init_image_list[i]
                seed = seed_list[i]

                # change appeal for each unappealing embedding
                for appealing_prompt in appealing_prompt_list:
                    for unappealing_prompt in unappealing_prompt_list:
                        copy_p = setup_processing(
                            p=p,
                            prompt="",
                            negative_prompt=p.negative_prompt,
                            init_images=[init_image],
                            image_mask=total_mask,
                            inpainting_fill=1, # original
                            denoising_strength=0.5,
                            seed= seed
                        )
                        processed_images = process(copy_p, unappealing_prompt + " " +  caption, appealing_prompt + " " + caption, n_interpolations)

                        # prepare image output
                        for i in range(len(processed_images[0])):
                            img, interpolation = processed_images[0][i]
                            save_item = (
                                img, copy_p.image_mask,
                                {"score": interpolation, "seed": copy_p.seed, "prompt": unappealing_prompt + "-->" + appealing_prompt,
                                }
                            )
                            save_image_list.append(save_item)

            # save image
            for image_i, mask_i, dict_i in save_image_list:
                out_name = [f"{k}={v}" for k, v in dict_i.items()] + ["name=" + os.path.splitext(os.path.basename(image_path))[0]]
                out_path = "_".join(out_name)
                image_i.save(os.path.join(image_out_dir, out_path + ".jpg"))
                mask_i.save(os.path.join(image_out_dir, out_path + ".png"))


# setup Stable Diffusion parameters
def setup_processing(p, prompt: str, negative_prompt: str, init_images: list, image_mask: Image.Image,
    inpainting_fill: int, denoising_strength: float, seed: int, ):
    
    copy_p = copy.copy(p)
    copy_p.seed = seed

    copy_p.prompt = prompt
    copy_p.negative_prompt = negative_prompt
    copy_p.init_images = init_images
    copy_p.image_mask = image_mask
    copy_p.inpainting_fill = inpainting_fill
    copy_p.denoising_strength = denoising_strength

    return copy_p

# Stable Diffusion image generation with prompt interpolation
def process(p, prompt1, prompt2, n_images):
    assert n_images > 1, "Should generate more than one image with prompt interpolation"
    processed_images = []

    for k in range(n_images):
        delta = round(random.uniform(-0.2, 0.2), 2)
        alpha = max(0, min(k/(n_images-1) + delta, 1)) # appeal score

        p.prompt = f"{prompt1} :{1 - alpha} AND {prompt2} :{alpha}"
        processed = process_images(p)

        for i, img in enumerate(processed.images):
            processed_images.append((img, alpha))

    return processed_images
