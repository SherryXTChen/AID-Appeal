# Image Appeal Assessment (IPA)

## Data
download dataset from [Dropbox]([https://www.dropbox.com/scl/fo/o2eebro2kqi9uzhojlkt1/h?dl=0&rlkey=xq877c336zvaeuwyrmjhzswqb](https://www.dropbox.com/sh/t7kpyqtro7f6pgs/AAD-ayQg9ZarI0-UM8v4h6k2a?dl=0), and move the directory to this folder


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
