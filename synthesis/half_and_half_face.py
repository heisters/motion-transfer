#!/usr/bin/env python

import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())
from motion_transfer.paths import build_paths, data_paths_for_idx, create_directories
from motion_transfer.labelling import Labeller
from motion_transfer.face import Face
from motion_transfer.pose import Pose

def parse_arguments():
    p = argparse.ArgumentParser(description="Generate labels based on programmatic movement", fromfile_prefix_chars='@')
    p.add_argument('--dataroot', type=str)
    p.add_argument('-n', '--nframes', help='Number of frames to generate', required=True, type=int)
    p.add_argument('--frame-offset', help='The frame offset for the two datasets', required=True, type=int)
    p = Labeller.add_arguments(p)

    return p.parse_args()

print("Synthesizing half and half faces")

args = parse_arguments()
paths_base = build_paths(args, directory_prefix='base')
paths_out = build_paths(args, directory_prefix='test')
create_directories(paths_out)
labeller_a = Labeller.build_from_arguments(args, paths_base, label_offset=0)
labeller_b = Labeller.build_from_arguments(args, paths_base)

base_image_fns = sorted(os.listdir(paths_base.img_dir))

for i in tqdm(range(0,args.nframes)):
    t = float(i) / float(args.nframes)
    path_out_image, path_out_label, _ = data_paths_for_idx(paths_out, i)

    if path_out_label.exists():
        continue

    path_base_image_a = paths_base.img_dir / base_image_fns[i]
    path_base_image_b = paths_base.img_dir / base_image_fns[i + args.frame_offset]

    base_image_a = cv.imread(str(path_base_image_a))
    if base_image_a is None:
        raise Exception("could not read image: {}".format(path_base_image_a))
    base_image_b = cv.imread(str(path_base_image_b))
    if base_image_b is None:
        raise Exception("could not read image: {}".format(path_base_image_b))

    faces_a = [Face(labeller_a.face_labeller.landmarks, f) for f in labeller_a.face_labeller.detect(base_image_a)]
    faces_b = [Face(labeller_b.face_labeller.landmarks, f) for f in labeller_b.face_labeller.detect(base_image_b)]
    pose_a = Pose(labeller_a.detect_pose(base_image_a))
    pose_b = Pose(labeller_b.detect_pose(base_image_b))

    labels = np.zeros(base_image_a.shape, dtype=np.uint8)
    labels_a = np.zeros(labels.shape, dtype=np.uint8)
    labels_b = np.zeros(labels.shape, dtype=np.uint8)
    center = (labels.shape[1] / 2, labels.shape[0] / 2)

    for (face_a, face_b) in zip(faces_a, faces_b):
        center_a = np.mean(face_a.nose, axis=0)
        center_b = np.mean(face_b.nose, axis=0)

        offset_a = center - center_a
        offset_b = center - center_b

        for (name, shape) in face_a.shapes():
            face_a[name] = shape + offset_a
        for (name, shape) in face_b.shapes():
            face_b[name] = shape + offset_b

        labels_a = labeller_a.face_labeller.visualize_facial_landmarks(labels_a, face_a.shape, alpha=1.0)
        labels_b = labeller_b.face_labeller.visualize_facial_landmarks(labels_b, face_b.shape, alpha=1.0)

    points_a = np.where(pose_a.points >= 0, pose_a.points + offset_a, pose_a.points)
    points_b = np.where(pose_b.points >= 0, pose_b.points + offset_b, pose_b.points)
    labels_a = labeller_a.draw_labels(labels_a, points_a)
    labels_b = labeller_b.draw_labels(labels_b, points_b)


    center_x = int(center[0])
    labels[:,:center_x] = labels_a[:,:center_x]
    labels[:,center_x:] = labels_b[:,center_x:]

    cv.imwrite(str(path_out_label), labels)
