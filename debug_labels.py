#!/usr/bin/env python

import os
import argparse
import cv2 as cv
from motion_transfer.paths import build_paths

p = argparse.ArgumentParser(description="Visualize the labels generated for a frame")
p.add_argument('-d', '--dataset', help='Name of the dataset', required=True)
p.add_argument('--source', help='The frame is a source frame', dest='source', action='store_true')
p.add_argument('--target', help='The frame is a target frame', dest='source', action='store_false')
p.add_argument('-f', '--frame', help='Frame number to visualize', type=int)
p.add_argument('-o', '--out', help='Output filename', default='debug_label.png')
p.set_defaults(source=True)

args = p.parse_args()
paths = build_paths(args)

label_dir = paths.source_label_dir if args.source else paths.target_label_dir
img_dir = paths.source_img_dir if args.source else paths.target_img_dir

img_path = img_dir / (os.listdir(img_dir)[-1] if args.frame is None else '{:05}.png'.format(args.frame))
label_path = label_dir / img_path.name

img = cv.imread(str(img_path))
label = cv.imread(str(label_path))

blended = img.copy()
blended[label > 0] = 180
cv.imwrite(args.out, blended)

print(args.out)

