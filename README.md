# TransWeather

 <a href="https://arxiv.org/abs/2111.14813"> </a>

Code for the paper [TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions](https://arxiv.org/abs/2111.14813), CVPR 2022

Note: The train, eval, and inference scripts in this code-base are complete and runnable (`train.py`, `test_*.py`, `inference.py`).

[Paper](https://arxiv.org/abs/2111.14813) | [Website](https://jeya-maria-jose.github.io/transweather-web/)

### About this repo:

This repo hosts the implentation code for the paper "TransWeather". We also provide code for a strong transformer baseline for weather removal tasks.

## Introduction

Removing adverse weather conditions like rain, fog, and snow from images is an important problem in many applications. Most methods proposed in the literature have been designed to deal with just removing one type of degradation. Recently, a CNN-based method  using neural architecture search (All-in-One) was proposed  to remove all the weather conditions at once. However, it has a large number of parameters as it uses multiple encoders to cater to each weather removal task and still has scope for improvement in its performance. In this work, we focus on developing an efficient solution for the all adverse weather removal problem. To this end, we propose TransWeather, a transformer-based end-to-end model with just a single encoder and a decoder that can restore an image degraded by any weather condition. Specifically, we utilize a novel transformer encoder using intra-patch transformer blocks to enhance attention inside the patches to effectively remove smaller weather degradations. We also introduce a transformer decoder with learnable weather type embeddings to adjust to the weather degradation at hand. TransWeather achieves significant improvements across multiple test datasets over both All-in-One network as well as methods fine-tuned for specific tasks.

<p align="center">
  <img src="imgs/Transweather.png" width="800"/>
</p>

## News [Update]:

A change in the PSNR values are observed as the calulation is done before saving the image which is not adept with previous works. However, we still get a better performance in most datasets (27.96 vs All-in-One's 24.71). It can also be noted that the TransWeather model can be further finetuned to get a better performance on individual datasets. These updates will be reflected on the new arxiv version.

## Using the code:

The code is stable with Python >= 3.9 and PyTorch >= 2.0 (CUDA optional; CPU also works).

- Clone this repository:
```bash
git clone https://github.com/jeya-maria-jose/TransWeather
cd TransWeather
```

To install all the dependencies using pip:

```bash
pip install -r requirements.txt
```


## Datasets:

### Train Data:

TransWeather is trained on a combination of images sampled from Outdoor-Rain, Snow100K, and Raindrop datasets (similar to [All-in-One (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.pdf)), dubbed as "All-Weather", containing 18069 images. It can be downloaded from this [link](https://drive.google.com/file/d/1tfeBnjZX1wIhIFPl6HOzzOKOyo0GdGHl/view?usp=sharing).


### Dataset format:

Download the datasets and arrange them in the following format. T
```
    TransWeather
    ├── data
    |   ├── train # Training
    |   |   ├── <dataset_name>
    |   |   |   ├── input         # rain images
    |   |   |   └── gt            # clean images
    |   |   └── dataset_filename.txt
    |   └── test  # Testing
    |   |   ├── <dataset_name>
    |   |   |   ├── input         # rain images
    |   |   |   └── gt            # clean images
    |   |   └── dataset_filename.txt
```

### Text Files:

[Link](https://drive.google.com/file/d/1UsazX-P3sPcDGw3kxkyFWqUyNfhYN_AM/view?usp=sharing)


## Training:

To train TransWeather on the All-Weather dataset (expects `./data/train/allweather.txt` listing the input images):

```bash
python train.py -exp_name Transweather -train_batch_size 32 -epoch_start 0 -num_epochs 250 -train_data_dir /root/autodl-tmp/allweather -train_filename train.txt -val_data_dir /root/autodl-tmp/allweather -val_filename val.txt
```

Checkpoints are saved to `./<exp_name>/latest` every epoch and to `./<exp_name>/best` whenever the validation PSNR improves. Training logs are written to `./training_log/<exp_name>_log.txt`.

## Training on a custom all-weather dataset (input/gt/gt_val layout):

If your dataset is a single directory mixing the weather types, with training ground truth in `gt/`, validation ground truth in `gt_val/`, and all degraded images in `input/`:

```
    <data_dir>
    ├── input     # degraded images (train + val)
    ├── gt        # ground truth of the training images
    └── gt_val    # ground truth of the validation images
```

first generate the train/val file lists (pairs input and ground truth by filename, understanding the Snow100K `x.jpg`↔`x.jpg`, Raindrop `x_rain.png`↔`x_clean.png`, and Outdoor-Rain `im_x_s*_a*.png`↔`im_x.png` conventions, and reports anything it could not match):

```bash
python prepare_allweather.py -data_dir /path/to/allweather
```

This writes `train.txt`, `val.txt`, and per-weather-type lists (`val_snow.txt`, `val_raindrop.txt`, `val_outdoorrain.txt`) into the dataset directory. Then train:

```bash
python train.py -exp_name Transweather -train_batch_size 32 -num_epochs 250 \
    -train_data_dir /path/to/allweather -train_filename train.txt \
    -val_data_dir /path/to/allweather -val_filename val.txt
```

and evaluate (overall, or per weather type by swapping `-val_filename`):

```bash
python test.py -exp_name Transweather -val_data_dir /path/to/allweather -val_filename val.txt
python test.py -exp_name Transweather -val_data_dir /path/to/allweather -val_filename val_snow.txt -category snow
```

`test.py` reports PSNR, SSIM, MAE (all on the Y channel of YCbCr), LPIPS (AlexNet), and DISTS as mean ± std, saves the restored images as lossless PNGs to `./results/<category>/<exp_name>/`, and writes the summary to `metrics_<val_filename>` in the same directory. Add `-save_gt` to also save the (resized) ground truth, `-verbose` to print per-image metrics. LPIPS/DISTS download pretrained backbones on first use.

To measure the same five metrics over two existing folders of restored/ground-truth image pairs (e.g. outputs of another baseline):

```bash
python measure.py -results_dir ./results/allweather/Transweather -gt_dir ./results/allweather/Transweather_gt
```

## Evaluation:

To evaluate a trained model on the test sets (expects the corresponding text file inside `./data/test/`), run one of:

```bash
python test_raindrop.py -exp_name Transweather
python test_raindropa.py -exp_name Transweather
python test_snow100k.py -exp_name Transweather
python test_test1.py -exp_name Transweather
```

Each script reports PSNR/SSIM (computed on the Y channel of YCbCr, consistent with prior weather-removal works) and saves the restored images to `./results/<category>/<exp_name>/`.

## Inference:

To restore your own images without ground truth, point `inference.py` at a directory (searched recursively) or a single image:

```bash
python inference.py -checkpoint ./Transweather/best -input_dir ./demo_images/ -output_dir ./results/inference/
```

## Extensions:

Note that Transweather is built to solve all adverse weather problem with a single model. We observe that, additionally TransWeather can be easilty modified (removing the transformer decoder) to just focus on an individual restoration task. To train just the Transweather-encoder on other datasets (like Rain-800), organize the dataset similar to all-weather and run the following command:

```
python train-individual.py  -train_batch_size 32 -exp_name Transweather-finetune -epoch_start 0 -num_epochs 250
```

Change ```train-individual.py``` with the necesarry details of the data to be trained on. Note that the network used there is a sub-section of our original Transweather architecture without the weather queries.

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [Syn2Real](https://github.com/rajeevyasarla/Syn2Real), [Segformer](https://github.com/NVlabs/SegFormer), and [ViT](https://github.com/lucidrains/vit-pytorch).

### Citation:

```
@misc{valanarasu2021transweather,
      title={TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions},
      author={Jeya Maria Jose Valanarasu and Rajeev Yasarla and Vishal M. Patel},
      year={2021},
      eprint={2111.14813},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
