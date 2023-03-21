# Image Appeal Assessment (IPA)

## Datasets

Create a dataset root directory called `datasets/` in this directory. Download `datasets/<dataset_name>.zip` from [here](https://www.dropbox.com/sh/t7kpyqtro7f6pgs/AAD-ayQg9ZarI0-UM8v4h6k2a?dl=0), move them to the dataset root directory, and unzip them.

## Training

### Relative Appeal Score Comparator

To train the relative appeal score comparator on the synthetic dataset and label the real image dataset with appeal scores, run this command:
```
bash scripts/1_relative_appeal_score_comparison.sh <dataset_name>
```
like the following:
```
bash scripts/1_relative_appeal_score_comparison.sh food
bash scripts/1_relative_appeal_score_comparison.sh room
```

We also provide the checkpoints and real image dataset appeal labels [here](https://www.dropbox.com/sh/t7kpyqtro7f6pgs/AAD-ayQg9ZarI0-UM8v4h6k2a?dl=0). Download checkpoints from `ckpts/pair_with_clip_<dataset_name>/last-v1.zip`. Download labels from `outputs/pair_with_clip_<dataset_name>/scores_real_all.txt` and move them to `datasets/<dataset_name>/scores.txt`

### Appeal Score Predictor

To train the appeal score predictor on the real image dataset, run this command:
```
bash scripts/2_appeal_score_prediction.sh <dataset_name>
```

We also provide the checkpoints [here](https://www.dropbox.com/sh/t7kpyqtro7f6pgs/AAD-ayQg9ZarI0-UM8v4h6k2a?dl=0). Download checkpoints from `ckpts/singular_with_clip_<dataset_name>/last-v1.zip`.

## Inference

### Image Appeal Heatmap Generation

To generate the image appeal heatmap, run this command:
```
python appeal_heatmap_generation.py --name singular_with_clip_<dataset_name> --input_dir <path_to_input_images> 
```

Use generated heatmaps as masks in Stable Diffusion inpainting to enhance image appeal.

