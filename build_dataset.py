#!/usr/bin/env python

import argparse
from pathlib import Path

from motion_transfer.paths import build_paths
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
    p.add_argument('--resize', help='Resize source to the given size')
    p.add_argument('--flip', help='Flip vertically, horizontally, or both', choices=['v', 'h', 'vh', 'hv'])
    p.add_argument('--label-with', help="Choose labelling strategy", choices=["openpose", "densepose"], default="densepose")
    p.add_argument('--exclude-landmarks', help="CSV list of facial landmarks to exclude from the labels", type=str)
    #p.add_argument('--label-face', help="Choose labelling strategy", dest='label_face', action='store_true')
    #p.add_argument('--no-label-face', help="Choose labelling strategy", dest='label_face', action='store_false')

    #p.set_defaults(label_face=True)

    return p.parse_args()

def create_directories(paths):
    for k, v in vars(paths).items():
        if k.endswith('_dir'):
            v.mkdir(exist_ok=True)


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

print("Creating directory hierarchy")
create_directories(paths)
print("Fetching models")
fetch_models(paths)
print("Decimating")
decimate_video(paths.input, paths.train_img_dir, trim=trim, subsample=args.subsample, resize=resize, flip=flip)
print("Labeling frames with %s" % args.label_with)
make_labels(args.label_with, paths, exclude_landmarks=exclude_landmarks)
