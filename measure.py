"""Five-metric evaluation: PSNR, SSIM, MAE, LPIPS, DISTS.

Adapted from the reference implementation in ../metrics (weather-diffusion
style). PSNR/SSIM/MAE are computed on the Y channel of YCbCr; LPIPS uses the
AlexNet backbone (`pip install lpips`), DISTS comes from piq
(`pip install piq`).

Can also be run standalone to compare two folders of same-sized images
(sorted filename order must correspond):

    python measure.py -results_dir ./results/allweather/Transweather -gt_dir ./results/allweather/Transweather_gt
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torchvision.utils as tvu

from utils import calculate_psnr, calculate_ssim, to_y_channel, tensor_to_bgr_image

METRICS = ['psnr', 'ssim', 'mae', 'lpips', 'dists']
IMG_EXTS = {'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'}


def _list_images(directory):
    return sorted(f for f in os.listdir(directory)
                  if not f.startswith('.') and f.split('.')[-1].lower() in IMG_EXTS)


def calculate_mae(img1, img2, test_y_channel=True):
    """Mean absolute error on [0, 1] scale."""
    assert img1.shape == img2.shape
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    return float(np.mean(np.abs(img1 - img2)) / 255.0)


def _bgr_to_rgb_tensor(img_bgr, device):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return tensor


class PerceptualMetricComputer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lpips_model = None
        self._dists_model = None

    def _get_lpips_model(self):
        if self._lpips_model is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError("LPIPS requires `pip install lpips`.") from exc
            self._lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
        return self._lpips_model

    def _get_dists_model(self):
        if self._dists_model is None:
            try:
                from piq import DISTS
            except ImportError as exc:
                raise ImportError("DISTS requires `pip install piq`.") from exc
            self._dists_model = DISTS().to(self.device).eval()
        return self._dists_model

    @torch.no_grad()
    def calculate_lpips(self, img1, img2):
        model = self._get_lpips_model()
        pred = _bgr_to_rgb_tensor(img1, self.device) * 2.0 - 1.0
        gt = _bgr_to_rgb_tensor(img2, self.device) * 2.0 - 1.0
        return float(model(pred, gt).item())

    @torch.no_grad()
    def calculate_dists(self, img1, img2):
        model = self._get_dists_model()
        pred = _bgr_to_rgb_tensor(img1, self.device)
        gt = _bgr_to_rgb_tensor(img2, self.device)
        return float(model(pred, gt).item())


def compute_all_metrics(pred_bgr, gt_bgr, computer, test_y_channel=True):
    """All five metrics for one BGR uint8 image pair."""
    return {
        'psnr': calculate_psnr(pred_bgr, gt_bgr, test_y_channel=test_y_channel),
        'ssim': calculate_ssim(pred_bgr, gt_bgr, test_y_channel=test_y_channel),
        'mae': calculate_mae(pred_bgr, gt_bgr, test_y_channel=test_y_channel),
        'lpips': computer.calculate_lpips(pred_bgr, gt_bgr),
        'dists': computer.calculate_dists(pred_bgr, gt_bgr),
    }


def summarize_metric(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return mean, std


def summarize_all(values_by_metric):
    summary = {}
    for name in METRICS:
        mean, std = summarize_metric(values_by_metric[name])
        summary[f"{name}_mean"] = mean
        summary[f"{name}_std"] = std
        summary[name] = values_by_metric[name]
    return summary


def format_mean_std(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def metric_summary_lines(summary, decimals=4):
    return ["%s: %s" % (name.upper().ljust(5), format_mean_std(
        summary[f"{name}_mean"], summary[f"{name}_std"], decimals=decimals
    )) for name in METRICS]


def print_metric_summary(summary, decimals=4, title=None):
    if title:
        print(title)
    for line in metric_summary_lines(summary, decimals=decimals):
        print(line)


@torch.no_grad()
def evaluate_loader(net, val_data_loader, device, save_dir=None, gt_save_dir=None, verbose=False):
    """Run the model over a ValData loader and compute all five metrics.

    Metrics are computed on 8-bit rounded values, matching what would be
    measured on the saved images. Restored images (and optionally the
    resized ground truth) are saved when the corresponding dir is given.
    """
    computer = PerceptualMetricComputer(device=device)
    values = {name: [] for name in METRICS}
    count = 0
    # inference-time accounting; the first batch is dropped from the average
    # because its forward pass includes one-time CUDA warm-up/allocation cost
    infer_time_total = 0.0
    infer_images = 0

    for batch_id, val_data in enumerate(val_data_loader):
        input_im, gt, names = val_data
        input_im = input_im.to(device)
        gt = gt.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        pred = net(input_im)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        spent = time.time() - start_time
        if batch_id > 0:
            infer_time_total += spent
            infer_images += int(input_im.shape[0])

        for b in range(pred.shape[0]):
            pred_bgr = tensor_to_bgr_image(pred[b]).astype(np.uint8)
            gt_bgr = tensor_to_bgr_image(gt[b]).astype(np.uint8)
            m = compute_all_metrics(pred_bgr, gt_bgr, computer)
            for name in METRICS:
                values[name].append(m[name])

            image_name = names[b].split('/')[-1]
            # save as PNG so the files stay lossless regardless of the source extension
            save_name = os.path.splitext(image_name)[0] + '.png'
            if save_dir:
                tvu.save_image(pred[b].clamp(0, 1), os.path.join(save_dir, save_name))
            if gt_save_dir:
                tvu.save_image(gt[b].clamp(0, 1), os.path.join(gt_save_dir, save_name))

            count += 1
            if verbose:
                print("[%d] %s  PSNR=%.4f, SSIM=%.4f, MAE=%.4f, LPIPS=%.4f, DISTS=%.4f"
                      % (count, image_name, m['psnr'], m['ssim'], m['mae'], m['lpips'], m['dists']))
            elif count % 50 == 0:
                print("processed {} images".format(count))

    summary = summarize_all(values)
    summary['count'] = count
    summary['infer_time_total'] = infer_time_total
    summary['infer_images'] = infer_images
    summary['infer_time_avg'] = infer_time_total / infer_images if infer_images else float('nan')
    return summary


def evaluate_folder_pair(results_path, gt_path, device=None, verbose=True, test_y_channel=True):
    """Compute all five metrics over two folders of same-sized image pairs."""
    imgs_name = _list_images(results_path)
    gts_name = _list_images(gt_path)
    assert len(imgs_name) == len(gts_name), (
        f"Image count mismatch: results={len(imgs_name)}, gt={len(gts_name)}")

    computer = PerceptualMetricComputer(device=device)
    values = {name: [] for name in METRICS}

    for i in range(len(imgs_name)):
        res = cv2.imread(os.path.join(results_path, imgs_name[i]), cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(gt_path, gts_name[i]), cv2.IMREAD_COLOR)
        if res is None:
            raise FileNotFoundError(f"Failed to read image: {os.path.join(results_path, imgs_name[i])}")
        if gt is None:
            raise FileNotFoundError(f"Failed to read image: {os.path.join(gt_path, gts_name[i])}")

        m = compute_all_metrics(res, gt, computer, test_y_channel=test_y_channel)
        for name in METRICS:
            values[name].append(m[name])
        if verbose:
            print("%s vs %s  PSNR=%.4f, SSIM=%.4f, MAE=%.4f, LPIPS=%.4f, DISTS=%.4f"
                  % (imgs_name[i], gts_name[i], m['psnr'], m['ssim'], m['mae'], m['lpips'], m['dists']))

    summary = summarize_all(values)
    summary['count'] = len(imgs_name)
    if verbose:
        print_metric_summary(summary, decimals=4, title="Testing set metrics (mean ± std):")
        print("Total image:%d" % len(imgs_name))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute PSNR/SSIM/MAE/LPIPS/DISTS over two folders of image pairs')
    parser.add_argument('-results_dir', help='folder with restored images', required=True, type=str)
    parser.add_argument('-gt_dir', help='folder with ground-truth images (same count and sizes, matching sorted order)', required=True, type=str)
    parser.add_argument('-quiet', help='only print the final summary', action='store_true')
    args = parser.parse_args()

    summary = evaluate_folder_pair(args.results_dir, args.gt_dir, verbose=not args.quiet)
    if args.quiet:
        print_metric_summary(summary, decimals=4, title="Testing set metrics (mean ± std):")
        print("Total image:%d" % summary['count'])
