#!/usr/bin/env python

import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from pathlib import Path
from motion_transfer.paths import build_paths, data_paths_for_idx, create_directories
from motion_transfer.labelling import Labeller

TAU = 6.2831853071795864769252867665590057683943

class Pose(object):
    # from https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    indices = {
        "nose"       : 0,  
        "neck"       : 1,  
        "rshoulder"  : 2,  
        "relbow"     : 3,  
        "rwrist"     : 4,  
        "lshoulder"  : 5,  
        "lelbow"     : 6,  
        "lwrist"     : 7,  
        "midhip"     : 8,  
        "rhip"       : 9,  
        "rknee"      : 10, 
        "rankle"     : 11, 
        "lhip"       : 12, 
        "lknee"      : 13, 
        "lankle"     : 14, 
        "reye"       : 15, 
        "leye"       : 16, 
        "rear"       : 17, 
        "lear"       : 18, 
        "lbigtoe"    : 19, 
        "lsmalltoe"  : 20, 
        "lheel"      : 21, 
        "rbigtoe"    : 22, 
        "rsmalltoe"  : 23, 
        "rheel"      : 24, 
        "background" : 25
    }

    def __init__(self, points):
        self.points = points

    def __getattr__(self, attr):
        if attr in self.__class__.indices:
            return self.points[self.__class__.indices[attr]]
        elif attr.endswith("_idx") and attr[:-4] in self.__class__.indices:
            return self.__class__.indices[attr[:-4]]
        else:
            raise AttributeError("%r object has no attribute %r" % self.__class__.__name__, attr)

    @property
    def larm(self):
        return self.points[[self.lshoulder_idx, self.lelbow_idx, self.lwrist_idx]]

    @larm.setter
    def larm(self, values):
        self.points[[self.lshoulder_idx, self.lelbow_idx, self.lwrist_idx]] = values


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

