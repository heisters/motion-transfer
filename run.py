#!/usr/bin/env python

import os
import argparse
import cv2 as cv
import wget
import numpy as np
from tqdm import tqdm
from src.paths import build_paths


#
# Helpers
#

def parse_arguments():
    p = argparse.ArgumentParser(description="Transfer motion from a source video to a target video")
    p.add_argument('-d', '--dataset', help='Name of the dataset', required=True)
    p.add_argument('-s', '--source', help='Path to the source video', required=True)
    p.add_argument('-t', '--target', help='Path to the target video', required=True)
    p.add_argument('--source-from', help='Decimal seconds to start reading from source video', type=float, default=0.0)
    p.add_argument('--source-to', help='Decimal seconds to read until from source video, -1 for the end', type=float, default=-1.0)
    p.add_argument('--target-from', help='Decimal seconds to start reading from target video', type=float, default=0.0)
    p.add_argument('--target-to', help='Decimal seconds to read until from target video, -1 for the end', type=float, default=-1.0)

    return p.parse_args()

def create_directories(paths):
    for k, v in vars(paths).items():
        if k.endswith('_dir'):
            v.mkdir(exist_ok=True)

def decimate_video(path, output_dir, limit=None, trim=(0.0, -1.0)):
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened(): return

    nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if limit is not None: nframes = min(nframes, limit)
    fps = float(cap.get(cv.CAP_PROP_FPS))

    start_frame = int(fps * trim[0])
    end_frame = nframes if trim[1] < 0.0 else max(start_frame + 1, int(fps * trim[1]))

    if (end_frame - start_frame) == len(os.listdir(str(output_dir))):
        print("Found %d frames, skipping." % (end_frame - start_frame))
        return

    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    for i in tqdm(range(end_frame - start_frame)):
        # do this here to increment frame counter, regardless of whether the file exists
        if not cap.grab(): break

        imgpath = output_dir / '{:05}.png'.format(i)

        if not imgpath.exists():
            success, frame = cap.retrieve()
            if not success: break

            #frame = cv.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))
            cv.imwrite(str(imgpath), frame)

def get_models(paths):
    if not paths.pose_model.exists():
        wget.download("http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel", str(paths.pose_model))
    else:
        print("Pose model OK: %s" % paths.pose_model)
    if not paths.pose_prototxt.exists():
        wget.download("https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt", str(paths.pose_prototxt))
    else:
        print("Pose prototxt OK: %s" % paths.pose_prototxt)



def label_images(net, img_dir, label_dir):
    num_points = 25
    point_pairs = [[1, 0], [1, 2], [1, 5], 
                    [2, 3], [3, 4], [5, 6], 
                    [6, 7], [0, 15], [15, 17], 
                    [0, 16], [16, 18], [1, 8],
                    [8, 9], [9, 10], [10, 11], 
                    [11, 22], [22, 23], [11, 24],
                    [8, 12], [12, 13], [13, 14], 
                    [14, 19], [19, 20], [14, 21]]
    threshold = 0.1

    for image_path_str in tqdm(os.listdir(str(img_dir))):
        image_path = img_dir / image_path_str
        label_path = label_dir / image_path.name

        if not label_path.exists():
            image = cv.imread(str(image_path))
            if image is None:
                raise RuntimeError("Unable to read image: %s" % image_path)

            iw = image.shape[1]
            ih = image.shape[0]
            size = (368,368)#(iw, ih) # others set this to (368, 368)
            blob = cv.dnn.blobFromImage(image, 1.0 / 255.0, size, (0,0,0), swapRB=False, crop=False)
            net.setInput(blob)

            output = net.forward()
            ow = output.shape[3]
            oh = output.shape[2]

            labels = np.zeros((ih, iw), dtype=np.uint8)

            points = []
            for i in range(num_points):
                confidence = output[0, i, :, :]
                minval, prob, minloc, point = cv.minMaxLoc(confidence)
                x = ( iw * point[0] ) / ow
                y = ( ih * point[1] ) / oh

                if prob > threshold:
                    points.append((int(x), int(y)))
                else:
                    points.append(None)

            label = 1
            for a, b in point_pairs:
                if points[a] and points[b]:
                    cv.line(labels, points[a], points[b], label, 3)
                label += 1

            cv.imwrite(str(label_path), labels)


def make_labels(paths):
    net = cv.dnn.readNetFromCaffe(str(paths.pose_prototxt), str(paths.pose_model))
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    print("Labeling source")
    label_images(net, paths.source_img_dir, paths.source_label_dir)
    print("Labeling target")
    label_images(net, paths.target_img_dir, paths.target_label_dir)

#
# Setup
#

args = parse_arguments()
paths = build_paths(args)
paths.source = Path(args.source)
paths.target = Path(args.target)



print("Creating directory hierarchy")
create_directories(paths)
print("Fetching models")
get_models(paths)
print("Decimating source video")
decimate_video(paths.source, paths.source_img_dir, trim=(args.source_from, args.source_to))
print("Decimating target video")
decimate_video(paths.target, paths.target_img_dir, trim=(args.target_from, args.target_to))
print("Labeling frames")
make_labels(paths)
