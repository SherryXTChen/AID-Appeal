# Image Appeal Assessment (IPA)

## Data
Create a root directory for datasets called ImageAppeal/. Download datasets from [here](https://www.dropbox.com/sh/t7kpyqtro7f6pgs/AAD-ayQg9ZarI0-UM8v4h6k2a?dl=0), move them to the root directory, and unzip them.

## Training

### Relative Appeal Score Comparator

```
bash scripts/1_relative_appeal_score_comparison.sh
```

### Appeal Score Predictor

```
bash scripts/2_appeal_score_prediction.sh
```

## Inference

### Image Appeal Heatmap Generation

```
python appeal_heatmap_generation.py --name singular_with_clip_<dataset_name> --input_dir <path_to_input_images> 
```
