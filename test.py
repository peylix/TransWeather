import time
import torch
import argparse
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData
from utils import validation_val
import os
import numpy as np
import random
from transweather_model import Transweather

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Evaluate TransWeather on an arbitrary test/validation list')
parser.add_argument('-val_data_dir', help='dataset root the paths in the list file are relative to', required=True, type=str)
parser.add_argument('-val_filename', help='text file listing the test images, inside val_data_dir', default='val.txt', type=str)
parser.add_argument('-exp_name', help='directory of the experiment whose weights are loaded', type=str)
parser.add_argument('-checkpoint', help='path to the model weight to load (defaults to ./<exp_name>/best)', default=None, type=str)
parser.add_argument('-category', help='subdirectory of ./results/ for the restored images', default='allweather', type=str)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_workers', help='number of dataloader workers (use 0 on macOS/Windows)', default=8 if sys.platform == 'linux' else 0, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name
category = args.category

checkpoint = args.checkpoint
if checkpoint is None:
    if exp_name is None:
        parser.error('either -checkpoint or -exp_name must be given')
    checkpoint = './{}/best'.format(exp_name)
if exp_name is None:
    exp_name = os.path.basename(os.path.dirname(os.path.abspath(checkpoint)))

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

# trailing separator is required because the data loaders concatenate paths
val_data_dir = os.path.join(os.path.expanduser(args.val_data_dir), '')
val_filename = args.val_filename

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Validation data loader --- #
val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=args.num_workers)

# --- Define the network --- #
net = Transweather().to(device)

net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
net.load_state_dict(torch.load(checkpoint, map_location=device))

# --- Use the evaluation model in testing --- #
net.eval()

os.makedirs('./results/{}/{}/'.format(category,exp_name), exist_ok=True)

print('--- Testing starts! ---')
print('checkpoint: {}, list: {}{}'.format(checkpoint, val_data_dir, val_filename))
start_time = time.time()
val_psnr, val_ssim = validation_val(net, val_data_loader, device, exp_name, category, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
print('restored images saved to ./results/{}/{}/'.format(category, exp_name))
