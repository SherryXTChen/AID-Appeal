# AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling (ECCV 2024)

AID-AppEAL is a system that automates dataset creation and implements algorithms to estimate and boost content appeal.

<p align="center">
  <img src="https://github.com/SherryXTChen/AID-AppEAL/blob/main/assets/teaser.png" alt="Figure 1" width="90%">
</p>


## Datasets

Create a dataset root directory called `datasets/` in this directory. Download `datasets/<dataset_name>.zip` from [here](https://www.dropbox.com/sh/t7kpyqtro7f6pgs/AAD-ayQg9ZarI0-UM8v4h6k2a?dl=0), move them to the dataset root directory, and unzip them.

For more details of how these datasets are created, see [DATASET_CREATION.md](https://github.com/SherryXTChen/AID-Appeal/blob/main/dataset_creation/DATASET_CREATION.md)

## Training

### Relative Content Appeal Score Comparator

To train the relative content appeal score comparator on the synthetic dataset for real image dataset labelling, run this command:
```
bash scripts/1_relative_appeal_score_comparison.sh <dataset_name>
```
like the following:
```
bash scripts/1_relative_appeal_score_comparison.sh food
bash scripts/1_relative_appeal_score_comparison.sh room
```
We provide the checkpoints [here](https://www.dropbox.com/scl/fo/5xwcopp1pw8bhfkp2rhia/h?rlkey=kgnrs22otr4y81a0y2zxdwz4y&dl=0) under `ckpts` following the nameing convension `ckpts/pair_with_clip_<dataset_name>/last-v1.zip`.

### Absolute Content Appeal Score Predictor

To train the absolute content appeal score predictor on the real image dataset, run this command:
```
bash scripts/2_appeal_score_prediction.sh <dataset_name>
```
We provide the checkpoints [here](https://www.dropbox.com/scl/fo/5xwcopp1pw8bhfkp2rhia/h?rlkey=kgnrs22otr4y81a0y2zxdwz4y&dl=0) under `ckpts` following the nameing convension `ckpts/singular_with_clip_<dataset_name>/last-v1.zip`.

## Inference

### Content Appeal Estimation and Heatmap Generation

To estimate the content appeal score without generating the heatmap, run this command:
```
python appeal_heatmap_generation.py --name singular_with_clip_<dataset_name> --input_dir <path_to_input_images> 
```
To estimate the content appeal score and generate the heatmap, run this command:
```
python appeal_heatmap_generation.py --name singular_with_clip_<dataset_name> --input_dir <path_to_input_images> --get_appeal_heatmap
```
Results will be saved under `outputs` by default.

### Content Appeal Enhancement

To enhance image content appeal, we use Automatic1111 [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) `> img2img > Generation > Inpaint upload` with Stable Diffusion v2.1, where aforementioned heatmaps are used as inpainting masks. For more details, pleae refer to the paper and the supplementary pdf.


## BibTeX

``` bibtex
@misc{chen2024aidappealautomaticimagedataset,
      title={AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling}, 
      author={Sherry X. Chen and Yaron Vaxman and Elad Ben Baruch and David Asulin and Aviad Moreshet and Misha Sra and Pradeep Sen},
      year={2024},
      eprint={2407.05546},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.05546}, 
}
```
