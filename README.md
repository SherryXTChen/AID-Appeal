# Image Appeal Assessment (IPA)

## Data
download dataset from ..., and move the directory to this folder


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
