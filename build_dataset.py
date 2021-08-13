#!/usr/bin/env python

import argparse
from pathlib import Path
import os

from motion_transfer.paths import build_paths, create_directories
from motion_transfer.video_utils import decimate_and_label_video, CropCenter
from motion_transfer.labelling import fetch_models, Labeller



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
    p.add_argument('--crop', help='After resizing, crop to the given size')
    p.add_argument('--crop-center', help='Center to use for cropping', choices=[c.name for c in CropCenter], default=CropCenter.body.name)
    p.add_argument('--flip', help='Flip vertically, horizontally, or both', choices=['v', 'h', 'vh', 'hv'])
    p.add_argument('--normalize', help='Output frame data for normalization', action='store_true')

    p = Labeller.add_arguments(p)

    p.add_argument('--face-size', help='The size (squared) of faces extracted to train the face network', type=int, default=128)
    p.add_argument('--directory-prefix', help='Image and label directory prefixes for label training', default='train')
    p.add_argument('--no-label', help='Disable labeling', action='store_true')
    p.add_argument('--train-a', help="Put images in the train_A directory for non-label training", action='store_true')
    p.add_argument('--train-b', help="Put images in the train_B directory for non-label training", action='store_true')
    p.add_argument('--test-a', help="Put images in the test_A directory for non-label training", action='store_true')
    p.add_argument('--frame-offset', help="Offset all frame numbers by this number", type=int, default=0)

    p.set_defaults(normalize=False)

    return p.parse_args()


#
# Setup
#

args = parse_arguments()
paths = build_paths(args)
paths.input = Path(args.input)


resize = tuple(map(int, args.resize.split('x'))) if args.resize is not None else None
crop = tuple(map(int, args.crop.split('x'))) if args.crop is not None else None
trim = tuple(map(float, args.trim.split(':'))) if args.trim is not None else None
flip = {'v': 0, 'h': 1, 'hv': -1, 'vh': -1}[args.flip] if args.flip is not None else None
normalize = args.normalize
crop_center = CropCenter[args.crop_center]

print("Creating directory hierarchy")
create_directories(paths)
print("Fetching models")
fetch_models(paths)

if args.no_label:
    print("Decimating video")
    labeller = None
else:
    print("Decimating video and labelling with {}".format(args.label_with))
    labeller = Labeller.build_from_arguments(args, paths)

decimate_and_label_video(
        paths,
        labeller,
        trim=trim,
        subsample=args.subsample,
        subsample_offset=args.subsample_offset,
        resize=resize,
        crop=crop,
        crop_center=crop_center,
        flip=flip,
        normalize=normalize,
        frame_offset=args.frame_offset,
        face_size=args.face_size)


