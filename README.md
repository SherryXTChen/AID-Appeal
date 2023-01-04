# SocialMediaPredictor

## Data
The dataset consists of the following 4 classes, each with 101K images (to be shared separately)

- Appealing and professionally taken images from 101 food categories (1000 image per category) taken from [https://stock.adobe.com/](https://stock.adobe.com/). Images are saved under ImageAppeal/adobe_stock/food-101/
- Appealing and non-professionally taken images from 101 food categories taken from [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). Images are saved under ImageAppeal/food-101/
- Unappealing and professionally taken images taken from [https://stock.adobe.com/](https://stock.adobe.com/). Images are saved under ImageAppeal/adobe_stock/unappealing/
- Unappealing and non-professionally taken images taken from [https://images.google.com/](https://images.google.com/). Images are saved under ImageAppeal/google/


## Training

### Siamese Network

- The network is a simple CLIP + FC layers that outputs a floating number score.
- During training, each training sample consists (appealing image, unappealing image) pair, each image is sent to the network independently.
- Loss function is applied such that scores from appealing images should be bigger than ones from unappealing images per training sample.

```
python train_v0.py --name v0_siamese_pair --model_type 'siamese' --loss_type 'pair' --appeal_root_list ImageAppeal/adobe_stock/food-101/ ImageAppeal/food-101/ --unappeal_root_list ImageAppeal/adobe_stock/unappealing/ ImageAppeal/google/ --num_epochs 2
```

### Siamese Network + Triplet loss

- Same network as above
- During training, each training sample is an image triplet. The network is first trained partially to generate image embeddings. Embeddings of appealing images should be similiar to each other; same holds for unappealing images.
- Next, the entire network is trained with image pairs as above.

```
python train_v0.py --name v0_siamese_triplet --model_type 'siamese' --loss_type 'triplet' --appeal_root_list ImageAppeal/adobe_stock/food-101/ ImageAppeal/food-101/ --unappeal_root_list ImageAppeal/adobe_stock/unappealing/ ImageAppeal/google/ --num_epochs 1

python train_v0.py --name v0_siamese_triplet --model_type 'siamese' --loss_type 'pair' --appeal_root_list ImageAppeal/adobe_stock/food-101/ ImageAppeal/food-101/ --unappeal_root_list ImageAppeal/adobe_stock/unappealing/ ImageAppeal/google/ --num_epochs 2
```

### Comparator Network

- Similiar architecture as above
- During training, each training sample consists (appealing image, unappealing image) pair, their image CLIP vectors are computed and concatenated before sent to the rest of the network
- CrossEntropy loss is used to perform a binary classification and determine which image is more appealing

```
python train_v0.py --name v0_concate --model_type 'concate' --loss_type 'pair' --appeal_root_list ImageAppeal/adobe_stock/food-101/ ImageAppeal/food-101/ --unappeal_root_list ImageAppeal/adobe_stock/unappealing/ ImageAppeal/google/ --num_epochs 2
```

## Inference

### Siamese Network

- During inference, the network takes an image and outputs a score, which is the appeal score of this image.

```
python test.py --name v0_siamese_pair --model_type 'siamese' --loss_type 'pair' --appeal_root_list ImageAppeal/adobe_stock/food-101/ ImageAppeal/food-101/ --unappeal_root_list ImageAppeal/adobe_stock/unappealing/ ImageAppeal/google/
```

### Siamese Network + Triplet loss

- Same as above

```
python test.py --name v0_siamese_triplet --model_type 'siamese' --loss_type 'triplet' --appeal_root_list ImageAppeal/adobe_stock/food-101/ ImageAppeal/food-101/ --unappeal_root_list ImageAppeal/adobe_stock/unappealing/ ImageAppeal/google/
```

### Comparator Network

- To get appeal score of Image I, compare it with k randomly selected images by sending each with Image I to the network at a time. The percentage of images out of these k images where Image I is more appealing accordingly to the model is the appeal score of Image I.

```
python test.py --name v0_concate --model_type 'concate' --loss_type 'pair' --appeal_root_list ImageAppeal/adobe_stock/food-101/ ImageAppeal/food-101/ --unappeal_root_list ImageAppeal/adobe_stock/unappealing/ ImageAppeal/google/
```

### Visualization

Code in inference section generates a text file called outputs/{name}_val/lists.txt that consists a list of images and their scores. To visualize them in the file system (outputs/{name}_val/symlinks):

```
python postprocess.py symlink_images outputs/{name}_val {absolute_path_to_images}
```

To visualize them in webpages:

```
python postprocess.py generate_html_all_food_101 outputs/{name}_val 
```

Open outputs/{name}_val/webpages/{food_category}/index.html to visualize image appeal ranking per food category.


### User study

Finally, to perform a user study and compare user preference to model outputs

```
python postprocess.py user_study outputs/{name}_val/symlinks {number_of_image_pairs}
```

User study will be saved under outputs/{name}_val/user_study/{timestamp}
