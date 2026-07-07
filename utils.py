import time
import os
from math import log10

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as tvu


# --- PSNR/SSIM metrics (adapted from https://github.com/JingyunLiang/SwinIR) --- #

def calculate_psnr(img1, img2, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Args:
        img1 (ndarray): Images with range [0, 255], order 'HWC', BGR.
        img2 (ndarray): Images with range [0, 255], order 'HWC', BGR.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=False):
    """Calculate SSIM (structural similarity).

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255], order 'HWC', BGR.
        img2 (ndarray): Images with range [0, 255], order 'HWC', BGR.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    """Convert the type and range of the input image to np.float32, [0, 1]."""
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type."""
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image using the ITU-R BT.601 conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def tensor_to_bgr_image(tensor):
    """Convert a (3, H, W) RGB tensor in [0, 1] to an HWC BGR ndarray in [0, 255].

    Values are rounded to integers so metrics match what would be computed on
    the saved 8-bit images.
    """
    img = tensor.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255.0).round().astype(np.float64)
    return img[:, :, ::-1]


def calc_psnr(pred_image, gt):
    return [calculate_psnr(tensor_to_bgr_image(pred_image[ind]), tensor_to_bgr_image(gt[ind]), test_y_channel=True)
            for ind in range(pred_image.shape[0])]


def calc_ssim(pred_image, gt):
    return [calculate_ssim(tensor_to_bgr_image(pred_image[ind]), tensor_to_bgr_image(gt[ind]), test_y_channel=True)
            for ind in range(pred_image.shape[0])]


def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def validation(net, val_data_loader, device, exp_name, save_tag=False):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, imgid = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            save_image(pred_image, imgid, exp_name, 'val')

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def validation_val(net, val_data_loader, device, exp_name, category, save_tag=False):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, imgid = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            save_image(pred_image, imgid, exp_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(pred_image, image_name, exp_name, category):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)

    save_dir = './results/{}/{}/'.format(category, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
        tvu.save_image(pred_image_images[ind], '{}{}'.format(save_dir, image_name_1))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    os.makedirs('./training_log/', exist_ok=True)
    with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 100

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
