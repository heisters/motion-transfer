#!/usr/bin/env python

import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from pathlib import Path
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.getcwd())
from motion_transfer.paths import build_paths, data_paths_for_idx, create_directories
from motion_transfer.labelling import Labeller
from motion_transfer.pose import Pose

TAU = 6.2831853071795864769252867665590057683943



def parse_arguments():
    p = argparse.ArgumentParser(description="Generate labels based on programmatic movement", fromfile_prefix_chars='@')
    p.add_argument('--dataroot', type=str)
    p.add_argument('-i', '--input', help='Path to a frame image that provides the basis for the generation', required=True)
    p.add_argument('-n', '--nframes', help='Number of frames to generate', required=True, type=int)
    p = Labeller.add_arguments(p)


    return p.parse_args()

args = parse_arguments()
paths = build_paths(args, directory_prefix='test')
create_directories(paths)
labeller = Labeller.build_from_arguments(args, paths)

image_path = Path(args.input)
finput = cv.imread(str(image_path))

pose = Pose(labeller.detect_pose(finput))
larm = np.atleast_2d(pose.larm.copy())
origin = np.atleast_2d(pose.lshoulder.copy())

for i in tqdm(range(0,args.nframes)):
    t = float(i) / float(args.nframes)

    theta = TAU * .25 * t - TAU * .125
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array(((c,-s),(s,c)))

    pose.larm = (rotation @ ( larm.T - origin.T ) + origin.T).T


    labels = np.zeros(finput.shape, dtype=np.uint8)
    labeller.draw_labels(labels, pose.points)

    image_path, label_path, norm_path = data_paths_for_idx(paths, i)
    cv.imwrite(str(label_path), labels)

