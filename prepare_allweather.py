"""Generate train/val file lists for a merged all-weather dataset.

Expects a dataset directory with the following layout:

    <data_dir>
    ├── input     # degraded images (train + val, all three weather types mixed)
    ├── gt        # ground truth for training images
    └── gt_val    # ground truth for validation images

Input and ground-truth images are paired by filename, supporting the naming
conventions of the three source datasets:
  - Snow100K:     same filename in input and gt
  - Raindrop:     <id>_rain.<ext>  <->  <id>_clean.<ext> (or <id>.<ext>)
  - Outdoor-Rain: im_<id>_s<..>_a<..>.<ext>  <->  im_<id>.<ext>

Writes into <data_dir>:
  - train.txt / val.txt      lines of "<input_path>\\t<gt_path>" (tab-separated,
                             because e.g. Snow100K file names contain spaces)
  - val_<category>.txt       per-weather-type validation lists (when detected)
"""
import argparse
import os
import re

IMG_EXTS = {'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'}

OUTDOOR_RAIN_RE = re.compile(r'^(?P<base>im_\d+)_s\d+_a\d+$', re.IGNORECASE)
RAINDROP_RE = re.compile(r'^\d+_rain$', re.IGNORECASE)

parser = argparse.ArgumentParser(description='Generate train/val lists for an input/gt/gt_val dataset')
parser.add_argument('-data_dir', help='dataset root containing input/, gt/ and gt_val/', required=True, type=str)
args = parser.parse_args()

data_dir = os.path.expanduser(args.data_dir)


def list_images(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory)
                  if not f.startswith('.') and f.split('.')[-1].lower() in IMG_EXTS)


def stem(name):
    return os.path.splitext(name)[0]


def build_index(directory):
    """Map filename stem -> filename for every image in the directory."""
    index = {}
    for f in list_images(directory):
        index[stem(f)] = f
    return index


def gt_candidates(input_stem):
    """Possible ground-truth stems for an input stem, in priority order."""
    cands = [input_stem]
    m = OUTDOOR_RAIN_RE.match(input_stem)
    if m:
        cands.append(m.group('base'))
    if input_stem.endswith('_rain'):
        cands.append(input_stem[:-5] + '_clean')
        cands.append(input_stem[:-5])
    return cands


def classify(input_stem):
    if OUTDOOR_RAIN_RE.match(input_stem):
        return 'outdoorrain'
    if RAINDROP_RE.match(input_stem) or input_stem.endswith('_rain'):
        return 'raindrop'
    return 'snow'


def find_match(input_stem, index):
    for cand in gt_candidates(input_stem):
        if cand in index:
            return index[cand]
    return None


input_files = list_images(os.path.join(data_dir, 'input'))
gt_index = build_index(os.path.join(data_dir, 'gt'))
gt_val_index = build_index(os.path.join(data_dir, 'gt_val'))

if not input_files:
    raise FileNotFoundError('No images found in {}'.format(os.path.join(data_dir, 'input')))
if not gt_index and not gt_val_index:
    raise FileNotFoundError('No images found in gt/ or gt_val/ under {}'.format(data_dir))

train_pairs = []
val_pairs = []
val_by_category = {}
unmatched = []
overlap = 0
matched_gt_val_names = set()

for f in input_files:
    s = stem(f)
    gt_val_name = find_match(s, gt_val_index)
    gt_name = find_match(s, gt_index)

    if gt_val_name is not None:
        val_pairs.append((f, gt_val_name))
        matched_gt_val_names.add(gt_val_name)
        val_by_category.setdefault(classify(s), []).append((f, gt_val_name))
        if gt_name is not None:
            # ground truth present in both gt/ and gt_val/: keep the image
            # out of the training list so the validation set stays held out
            overlap += 1
    elif gt_name is not None:
        train_pairs.append((f, gt_name))
    else:
        unmatched.append(f)


def write_list(filename, pairs, gt_dir):
    path = os.path.join(data_dir, filename)
    with open(path, 'w') as fh:
        for input_name, gt_name in pairs:
            fh.write('input/{0}\t{1}/{2}\n'.format(input_name, gt_dir, gt_name))
    return path


train_path = write_list('train.txt', train_pairs, 'gt')
val_path = write_list('val.txt', val_pairs, 'gt_val')

print('--- Dataset preparation report ---')
print('input images:  {}'.format(len(input_files)))
print('train pairs:   {}  -> {}'.format(len(train_pairs), train_path))
print('val pairs:     {}  -> {}'.format(len(val_pairs), val_path))

if len(val_by_category) > 1:
    for category, pairs in sorted(val_by_category.items()):
        path = write_list('val_{}.txt'.format(category), pairs, 'gt_val')
        print('val ({}): {}  -> {}'.format(category, len(pairs), path))

if overlap:
    print('WARNING: {} images have ground truth in both gt/ and gt_val/; '
          'they were assigned to the validation list only.'.format(overlap))

unused_gt_val = sorted(set(gt_val_index.values()) - matched_gt_val_names)
if unused_gt_val:
    print('WARNING: {} images in gt_val/ have no matching input, e.g. {}'.format(
        len(unused_gt_val), unused_gt_val[:5]))

if unmatched:
    print('WARNING: {} input images have no ground truth in gt/ or gt_val/, e.g. {}'.format(
        len(unmatched), unmatched[:5]))

if not train_pairs:
    print('WARNING: no training pairs were found - check the filename conventions above.')
