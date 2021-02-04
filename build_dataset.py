#!/usr/bin/env python

import argparse
from pathlib import Path
import os

from motion_transfer.paths import build_paths, create_directories
from motion_transfer.video_utils import decimate_video
from motion_transfer.labelling import fetch_models, make_labels



#
# Helpers
#

def parse_arguments():
    p = argparse.ArgumentParser(description="Transfer motion from a source video to a target video", fromfile_prefix_chars='@')
    p.add_argument('--dataroot', type=str)
    p.add_argument('-i', '--input', help='Path to the video', required=True)
    p.add_argument('--trim', help='Decimal, colon separated seconds to trim the input video to. -1 indicates the end of the video', type=str, default='0:-1')
    p.add_argument('--subsample', help='Factor to subsample the source frames, every Nth frame will be selected', type=int, default=1)
    p.add_argument('--subsample-offset', help='Offset for subsampling the source frames, every Nth+i frame will be selected', type=int, default=0)
    p.add_argument('--resize', help='Resize source to the given size')
    p.add_argument('--flip', help='Flip vertically, horizontally, or both', choices=['v', 'h', 'vh', 'hv'])
    p.add_argument('--label-with', help="Choose labelling strategy", choices=["openpose", "densepose"], default="densepose")
    p.add_argument('--exclude-landmarks', help="CSV list of facial landmarks to exclude from the labels", type=str)
    p.add_argument('--normalize', help='Output frame data for normalization', action='store_true')

    p.add_argument('--directory-prefix', help='Image and label directory prefixes for label training', default='train')
    p.add_argument('--no-label', help='Disable labeling', action='store_true')
    p.add_argument('--train-a', help="Put images in the train_A directory for non-label training", action='store_true')
    p.add_argument('--train-b', help="Put images in the train_B directory for non-label training", action='store_true')
    p.add_argument('--test-a', help="Put images in the test_A directory for non-label training", action='store_true')
    p.add_argument('--label-face', help="Add labels for the face (default on)", dest='label_face', action='store_true')
    p.add_argument('--no-label-face', help="Do not add labels for the face", dest='label_face', action='store_false')

    p.set_defaults(label_face=True, normalize=False)

    return p.parse_args()


#
# Setup
#

args = parse_arguments()
paths = build_paths(args)
paths.input = Path(args.input)


resize = tuple(map(int, args.resize.split('x'))) if args.resize is not None else None
trim = tuple(map(float, args.trim.split(':'))) if args.trim is not None else None
flip = {'v': 0, 'h': 1, 'hv': -1, 'vh': -1}[args.flip] if args.flip is not None else None
exclude_landmarks = set(args.exclude_landmarks.split(',')) if args.exclude_landmarks is not None else None
label_face = args.label_face
normalize = args.normalize

print("Creating directory hierarchy")
create_directories(paths)
print("Fetching models")
fetch_models(paths)
print("Decimating")
decimate_video(paths.input, paths.img_dir, trim=trim, subsample=args.subsample, subsample_offset=args.subsample_offset, resize=resize, flip=flip)
if not args.no_label:
    print("Labeling frames with %s" % args.label_with)

    nimgs = len(os.listdir(paths.img_dir))
    if (len(os.listdir(paths.label_dir)) >= nimgs or (normalize and paths.denorm_label_dir.exists() and len(os.listdir(paths.denorm_label_dir)) >= nimgs)) and (not normalize or len(os.listdir(paths.norm_dir)) >= nimgs):
        print("{} labels found, skipping.".format(nimgs))
    else:
        make_labels(args.label_with, paths, exclude_landmarks=exclude_landmarks, label_face=label_face, normalize=normalize)
