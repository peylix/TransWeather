"""Evaluate TransWeather on all three official test sets with one command.

For each test set this script:
  1. runs prepare_allweather.py if test.txt is missing (pass -regen to force),
  2. runs test.py (PSNR/SSIM/MAE/LPIPS/DISTS, mean ± std),
and finally prints the three metric summaries together.

Default paths match the AutoDL layout; override per test set if needed:

    python test_all.py -exp_name Transweather -data_root /root/autodl-tmp
"""
import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description='Run the three official test sets in one go')
parser.add_argument('-exp_name', help='directory of the experiment whose weights are loaded', type=str)
parser.add_argument('-checkpoint', help='path to the model weight to load (defaults to ./<exp_name>/best)', default=None, type=str)
parser.add_argument('-data_root', help='directory that contains the three test set folders', default='/root/autodl-tmp', type=str)
parser.add_argument('-raindrop_dir', help='RainDrop test_a directory (default: <data_root>/raindrop_data/test_a)', default=None, type=str)
parser.add_argument('-outdoorrain_dir', help='Outdoor-Rain test directory (default: <data_root>/CVPR19RainTrain/test)', default=None, type=str)
parser.add_argument('-snow_dir', help='Snow100K-L test directory (default: <data_root>/Snow100K-testset/jdway/GameSSD/overlapping/test/Snow100K-L)', default=None, type=str)
parser.add_argument('-val_batch_size', help='test batch size', default=1, type=int)
parser.add_argument('-regen', help='regenerate test.txt even if it already exists', action='store_true')
parser.add_argument('-results_dir', help='root directory for the restored images and metrics files (put this on a large disk for big test sets)', default='./results/', type=str)
parser.add_argument('-no_save', help='compute the metrics without saving the restored images', action='store_true')
parser.add_argument('-skip', help='test sets to skip (space-separated: raindrop outdoorrain snow100k_L)', nargs='+', default=[], type=str)
args = parser.parse_args()

if args.checkpoint is None and args.exp_name is None:
    parser.error('either -checkpoint or -exp_name must be given')

data_root = os.path.expanduser(args.data_root)

TEST_SETS = [
    {
        'name': 'raindrop',
        'dir': args.raindrop_dir or os.path.join(data_root, 'raindrop_data', 'test_a'),
        'input_subdir': 'data',
    },
    {
        'name': 'outdoorrain',
        'dir': args.outdoorrain_dir or os.path.join(data_root, 'CVPR19RainTrain', 'test'),
        'input_subdir': 'data',
    },
    {
        'name': 'snow100k_L',
        'dir': args.snow_dir or os.path.join(data_root, 'Snow100K-testset', 'jdway', 'GameSSD',
                                             'overlapping', 'test', 'Snow100K-L'),
        'input_subdir': 'synthetic',
    },
]

script_dir = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print('+ ' + ' '.join(cmd))
    result = subprocess.run(cmd, cwd=script_dir)
    if result.returncode != 0:
        raise SystemExit('command failed with exit code {}: {}'.format(result.returncode, ' '.join(cmd)))


exp_name = args.exp_name
if exp_name is None:
    exp_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))

results_root = os.path.expanduser(args.results_dir)
if not os.path.isabs(results_root):
    results_root = os.path.join(script_dir, results_root)

summaries = []
for ts in TEST_SETS:
    if ts['name'] in args.skip:
        print('\n===== {} skipped (-skip) ====='.format(ts['name']))
        continue
    print('\n===== {} ({}) ====='.format(ts['name'], ts['dir']))
    if not os.path.isdir(ts['dir']):
        raise SystemExit('test set directory not found: {}'.format(ts['dir']))

    # --- Generate the pair list if needed --- #
    list_path = os.path.join(ts['dir'], 'test.txt')
    if args.regen or not os.path.exists(list_path):
        run([sys.executable, 'prepare_allweather.py',
             '-data_dir', ts['dir'], '-input_subdir', ts['input_subdir']])

    # --- Evaluate --- #
    cmd = [sys.executable, 'test.py',
           '-val_data_dir', ts['dir'], '-val_filename', 'test.txt',
           '-category', ts['name'], '-val_batch_size', str(args.val_batch_size),
           '-results_dir', results_root]
    if args.no_save:
        cmd += ['-no_save']
    if args.checkpoint:
        cmd += ['-checkpoint', args.checkpoint]
    else:
        cmd += ['-exp_name', args.exp_name]
    run(cmd)

    summaries.append((ts['name'], os.path.join(results_root, ts['name'], exp_name,
                                               'metrics_test.txt')))

print('\n===== All test sets finished. Summary =====')
for name, metrics_path in summaries:
    print('\n[{}] ({})'.format(name, metrics_path))
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            print(f.read().rstrip())
    else:
        print('metrics file not found!')
