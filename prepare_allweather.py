"""Generate train/val/test file lists for a weather-restoration dataset.

Expects a dataset directory with the following layout (subdirectory names are
configurable via -input_subdir/-gt_subdir/-gt_val_subdir, so the official test
sets work directly, e.g. Raindrop's data/gt or Snow100K's synthetic/gt):

    <data_dir>
    ├── input     # degraded images (all weather types mixed)
    ├── gt        # ground truth
    └── gt_val    # ground truth for validation images (optional)

Input and ground-truth images are paired by filename, supporting the naming
conventions of the three source datasets:
  - Snow100K:     same filename in input and gt
  - Raindrop:     <id>_rain.<ext>  <->  <id>_clean.<ext> (or <id>.<ext>)
  - Outdoor-Rain: im_<id>_s<..>_a<..>.<ext>  <->  im_<id>.<ext>

Writes into <data_dir> (tab-separated "<input_path>\\t<gt_path>" lines, because
e.g. Snow100K file names contain spaces):
  - train.txt                training pairs (gt/ minus the gt_val/ overlap)
  - val.txt                  validation pairs (ground truth in gt_val/)
  - test.txt                 the full set: every pair with ground truth in gt/
  - val_<category>.txt /     per-weather-type lists (when more than one
    test_<category>.txt      weather type is detected)
"""
import argparse
import os
import re

IMG_EXTS = {'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'}

OUTDOOR_RAIN_RE = re.compile(r'^(?P<base>im_\d+)_s\d+_a\d+$', re.IGNORECASE)
RAINDROP_RE = re.compile(r'^\d+_rain$', re.IGNORECASE)

parser = argparse.ArgumentParser(description='Generate train/val/test lists for an input/gt[/gt_val] dataset')
parser.add_argument('-data_dir', help='dataset root containing the image subdirectories', required=True, type=str)
parser.add_argument('-input_subdir', help='subdirectory with the degraded images (e.g. "data" for Raindrop/Outdoor-Rain, "synthetic" for Snow100K)', default='input', type=str)
parser.add_argument('-gt_subdir', help='subdirectory with the ground-truth images', default='gt', type=str)
parser.add_argument('-gt_val_subdir', help='subdirectory with the validation ground truth (may not exist)', default='gt_val', type=str)
args = parser.parse_args()

data_dir = os.path.expanduser(args.data_dir)
input_subdir = args.input_subdir.strip('/')
gt_subdir = args.gt_subdir.strip('/')
gt_val_subdir = args.gt_val_subdir.strip('/')


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


input_files = list_images(os.path.join(data_dir, input_subdir))
gt_index = build_index(os.path.join(data_dir, gt_subdir))
gt_val_index = build_index(os.path.join(data_dir, gt_val_subdir))

if not input_files:
    raise FileNotFoundError('No images found in {}'.format(os.path.join(data_dir, input_subdir)))
if not gt_index and not gt_val_index:
    raise FileNotFoundError('No images found in {}/ or {}/ under {}'.format(gt_subdir, gt_val_subdir, data_dir))

train_pairs = []
val_pairs = []
test_pairs = []
val_by_category = {}
test_by_category = {}
unmatched = []
overlap = 0
matched_gt_val_names = set()

for f in input_files:
    s = stem(f)
    gt_val_name = find_match(s, gt_val_index)
    gt_name = find_match(s, gt_index)

    if gt_name is not None:
        # full test list: every image whose ground truth is in gt/
        test_pairs.append((f, gt_name))
        test_by_category.setdefault(classify(s), []).append((f, gt_name))

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
            fh.write('{0}/{1}\t{2}/{3}\n'.format(input_subdir, input_name, gt_dir, gt_name))
    return path


train_path = write_list('train.txt', train_pairs, gt_subdir)
val_path = write_list('val.txt', val_pairs, gt_val_subdir)
test_path = write_list('test.txt', test_pairs, gt_subdir)

print('--- Dataset preparation report ---')
print('input images:  {}'.format(len(input_files)))
print('train pairs:   {}  -> {}'.format(len(train_pairs), train_path))
print('val pairs:     {}  -> {}'.format(len(val_pairs), val_path))
print('test pairs:    {}  -> {}  (all images with ground truth in gt/)'.format(len(test_pairs), test_path))

if len(val_by_category) > 1:
    for category, pairs in sorted(val_by_category.items()):
        path = write_list('val_{}.txt'.format(category), pairs, gt_val_subdir)
        print('val ({}): {}  -> {}'.format(category, len(pairs), path))

if len(test_by_category) > 1:
    for category, pairs in sorted(test_by_category.items()):
        path = write_list('test_{}.txt'.format(category), pairs, gt_subdir)
        print('test ({}): {}  -> {}'.format(category, len(pairs), path))

if overlap:
    print('WARNING: {} images have ground truth in both gt/ and gt_val/; '
          'they were assigned to the validation list only.'.format(overlap))

# only meaningful for the single-directory layout where the same images are
# used for training; a pure train-only or test-only directory (no gt_val/)
# trivially has train.txt == test.txt
if gt_val_index:
    train_inputs = {input_name for input_name, _ in train_pairs}
    seen_in_training = sum(1 for input_name, _ in test_pairs if input_name in train_inputs)
    if seen_in_training:
        print('NOTE: {} of the {} test pairs were also used for training (they appear in train.txt). '
              'Metrics on test.txt therefore partly measure training-set performance.'.format(
                  seen_in_training, len(test_pairs)))

unused_gt_val = sorted(set(gt_val_index.values()) - matched_gt_val_names)
if unused_gt_val:
    print('WARNING: {} images in gt_val/ have no matching input, e.g. {}'.format(
        len(unused_gt_val), unused_gt_val[:5]))

if unmatched:
    print('WARNING: {} input images have no ground truth in gt/ or gt_val/, e.g. {}'.format(
        len(unmatched), unmatched[:5]))

if not train_pairs:
    print('WARNING: no training pairs were found - check the filename conventions above.')
