# AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling (ECCV 2024)

## Dataset Creation

We crawlled two stock image websites for image thumbtails using `image_crawling.py` and filter them based on the criteria listed in paper Sec. 3.1 (Fig. 2) using `image_filtering.py`. For more details, please refer to the code and the paper section.

To generate our synthetic datasets, first install Automatic1111 [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to `<sd_webui_path>`. Then copy our unappealing/unappealing embeddings located at [here](https://www.dropbox.com/scl/fo/5xwcopp1pw8bhfkp2rhia/h?rlkey=kgnrs22otr4y81a0y2zxdwz4y&dl=0) under `embeddings` to `<sd_webui_path>/embeddings` and copy our local `synthetic_appeal_dataset_creation.py` to `<sd_webui_path>/scripts`.
