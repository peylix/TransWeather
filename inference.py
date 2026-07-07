import time
import torch
import argparse
import torch.nn as nn
import os
import numpy as np
import random
import torchvision.utils as tvu
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

from transweather_model import Transweather

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Inference on images without ground truth')
parser.add_argument('-checkpoint', help='path to the model weight to load', required=True, type=str)
parser.add_argument('-input_dir', help='directory with degraded images (or a single image path)', required=True, type=str)
parser.add_argument('-output_dir', help='directory for saving the restored images', default='./results/inference/', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif']

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))


def list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split('.')[-1]
        if '.' in entry and ext.lower() in IMG_EXTENSIONS:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(list_image_files_recursively(full_path))
    return results


def load_image(path, device):
    input_img = Image.open(path).convert('RGB')

    # Resizing image in the multiple of 16"
    wd_new, ht_new = input_img.size
    if ht_new > wd_new and ht_new > 1024:
        wd_new = int(np.ceil(wd_new * 1024 / ht_new))
        ht_new = 1024
    elif ht_new <= wd_new and wd_new > 1024:
        ht_new = int(np.ceil(ht_new * 1024 / wd_new))
        wd_new = 1024
    wd_new = int(16 * np.ceil(wd_new / 16.0))
    ht_new = int(16 * np.ceil(ht_new / 16.0))
    input_img = input_img.resize((wd_new, ht_new), Image.LANCZOS)

    transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform_input(input_img).unsqueeze(0).to(device)


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Define the network --- #
net = Transweather().to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
net.load_state_dict(torch.load(args.checkpoint, map_location=device))
net.eval()

# --- Collect input images --- #
if os.path.isdir(args.input_dir):
    image_paths = list_image_files_recursively(args.input_dir)
else:
    image_paths = [args.input_dir]

if len(image_paths) == 0:
    raise FileNotFoundError('No images found in {}'.format(args.input_dir))

os.makedirs(args.output_dir, exist_ok=True)

print('--- Inference starts! ---')
total_time = 0
with torch.no_grad():
    for path in image_paths:
        input_im = load_image(path, device)
        start_time = time.time()
        pred_image = net(input_im)
        spent = time.time() - start_time
        total_time += spent

        save_name = os.path.splitext(os.path.basename(path))[0] + '.png'
        tvu.save_image(pred_image, os.path.join(args.output_dir, save_name))
        print('processed {} ({:.4f}s)'.format(path, spent))

print('average inference time is {0:.4f}s, total image is {1}'.format(total_time / len(image_paths), len(image_paths)))
print('restored images saved to {}'.format(args.output_dir))
