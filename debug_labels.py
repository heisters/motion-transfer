#!/usr/bin/env python

import os
import argparse
import tempfile
from tqdm import tqdm
import cv2 as cv
from pathlib import Path
from motion_transfer.paths import build_paths
from motion_transfer.video_utils import video_from_frame_directory

p = argparse.ArgumentParser(description="Visualize the labels generated for a dataset")
p.add_argument('-d', '--dataroot', help='Path to the dataset', required=True)
p.add_argument('-f', '--frame', help='Frame number to visualize, defaults to the last frame found', type=int)
p.add_argument('-v', '--video', help='Generate a video including all frames', action='store_true')
p.add_argument('-o', '--out', help='Output filename basename', default='debug_label')
p.add_argument('-l', '--labels', help='The number of labels to visualize', type=int, default=35)
p.add_argument('--directory-prefix', help='Image and label directory prefixes for label training', default='train')

args = p.parse_args()
paths = build_paths(args)
out = args.out + ('.mp4' if args.video else '.png')

label_dir = paths.label_dir
img_dir = paths.img_dir


def overlay_label(img_path, label_path):
    img = cv.imread(str(img_path))
    label = cv.imread(str(label_path))

    label = cv.multiply(label, 255 / args.labels)
    colored = cv.applyColorMap(label, cv.COLORMAP_PARULA)
    blended = cv.addWeighted(img, 0.25, colored, 0.75, 0)

    return blended

if args.video:
    with tempfile.TemporaryDirectory() as dirname:
        dirpath = Path(dirname)
        for img in tqdm(os.listdir(img_dir)):
            img_path = img_dir / img
            label_path = label_dir / img
            blended = overlay_label(img_path, label_path)

            cv.imwrite(str(dirpath / img), blended)

        video_from_frame_directory(dirpath, Path(out), frame_file_glob=r"%06d.png")


else:
    img = sorted(os.listdir(label_dir))[-1] if args.frame is None else '{:06}.png'.format(args.frame)
    img_path = img_dir / img
    label_path = label_dir / img_path.name
    blended = overlay_label(img_path, label_path)
    cv.imwrite(out, blended)


print(out)

