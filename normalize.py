#!/usr/bin/env python

import argparse
import numpy as np
import math
from scipy import stats
from PIL import Image
import sys
import os
import shutil
import cv2 as cv

from motion_transfer.paths import build_paths, data_paths
from motion_transfer.labelling import OpenposeLabeller, pose_point_is_valid

def isinvalid(p):
    return not pose_point_is_valid(p)

def parse_arguments():
    p = argparse.ArgumentParser(description="Normalize source video labels to the space of the target video")
    p.add_argument('--dataroot', type=str)
    p.add_argument('--limit', help="Only process the first N data samples", type=int)
    p.add_argument('--window', help="Starting ankle position window size as a floating-point percentage", type=float, default=0.05)
    p.add_argument('--zthreshold', help="Starting z-score threshold for height consideration", type=float, default=1.0)
    p.add_argument('--dry-run', help="Print calculated statistics and quit", action="store_true")
    return p.parse_args()

idx_lankle = 14
idx_rankle = 11
idx_nose = 0
idx_neck = 1
idx_hip = 8

def calculate_mean_ankle_for_pose(points):
    lankle = points[idx_lankle]
    rankle = points[idx_rankle]

    if isinvalid(lankle) or isinvalid(rankle):
        return np.array([-1,-1])

    lankle = np.array(lankle)
    rankle = np.array(rankle)

    return (lankle + rankle) / 2.0

def calculate_metrics_for_pose(points):

    nose = points[idx_nose]
    neck = points[idx_neck]
    hip = points[idx_hip]
    mean_ankle = calculate_mean_ankle_for_pose(points)

    if isinvalid(nose) or isinvalid(mean_ankle) or isinvalid(hip) or isinvalid(neck):
        return None


    nose = np.array(nose)

    # do not consider poses where the body is inverted, even partially
    yvalues = [nose[1], neck[1], hip[1], mean_ankle[1]]
    if sorted(yvalues) != yvalues:
        return None

    height = np.sqrt(np.sum((nose - mean_ankle) ** 2))

    return (mean_ankle, nose, height)

def calculate_metrics(paths, limit=None):
    ankles, noses, heights, images = [], [], [], []

    for i, (image_path, label_path, norm_path) in enumerate(data_paths(paths, normalize=True)):
        points = np.load(norm_path)
        metric = calculate_metrics_for_pose(points)

        if metric is not None:
            ankle, nose, height = metric
            ankles.append(ankle)
            noses.append(nose)
            heights.append(height)
            images.append(str(image_path))

        if args.limit is not None and i >= args.limit:
            break

    return np.array(ankles, float), np.array(noses, float), np.array(heights, float), np.array(images)

def calculate_ankle_and_height(ankles, noses, heights, method=np.amin, window=0.05, zthreshold=1.0):
    ankle_y = method(ankles[:,1])

    ankle_nearby = ( ankles[:,1] > ( ankle_y * ( 1.0 - window ) ) ) & ( ankles[:,1] < ( ankle_y * ( 1.0 + window ) ) )
    heights_nearby = heights[ankle_nearby]
    z = stats.zscore(heights_nearby)
    heights_over_z = heights_nearby[z >= zthreshold]

    height = math.nan
    if len(heights_over_z):
        height = np.median(heights_over_z) # was np.amax ... parameterize?

    return ankle_y, height

def calculate_close_and_far_transformations(ankles, noses, heights, start_window=0.05, start_zthreshold=1.0):
    window_inc = 0.025
    max_window = 0.3 # % of the image (doubled)
    zthreshold_inc = -0.1
    min_zthreshold = 0 # mean
    success = False

    window = start_window
    while not success and window <= max_window:
        zthreshold = start_zthreshold
        while not success and zthreshold >= min_zthreshold:
            a_close, h_close = calculate_ankle_and_height(ankles, noses, heights, method=np.amax, window=window, zthreshold=zthreshold)
            a_far, h_far = calculate_ankle_and_height(ankles, noses, heights, method=np.amin, window=window, zthreshold=zthreshold)

            if not math.isnan(h_close) and not math.isnan(h_far) and a_close > a_far and h_close > h_far:
                success = True


            zthreshold += zthreshold_inc
        window += window_inc

    if not success:
        raise Exception("Could not find a window and z-score threshold that result in expected transformations")

    return a_close, h_close, a_far, h_far, window, zthreshold



args = parse_arguments()
source_paths = build_paths(args, directory_prefix='test')
target_paths = build_paths(args, directory_prefix='train')

if source_paths.norm_calculations.exists():
    print("Loading normalization calculations from {}".format(source_paths.norm_calculations))
    calculations = np.load(source_paths.norm_calculations)
else:
    s_ankles, s_noses, s_heights, s_images = calculate_metrics(source_paths, limit=args.limit)
    t_ankles, t_noses, t_heights, t_images = calculate_metrics(target_paths, limit=args.limit)

    s_calc = calculate_close_and_far_transformations(s_ankles, s_noses, s_heights, start_window=args.window, start_zthreshold=args.zthreshold)
    t_calc = calculate_close_and_far_transformations(t_ankles, t_noses, t_heights, start_window=args.window, start_zthreshold=args.zthreshold)
    calculations = [s_calc, t_calc]

    if not args.dry_run:
        np.save(source_paths.norm_calculations, np.array(calculations, dtype=np.float))

s_close_ankle, s_close_height, s_far_ankle, s_far_height, s_win, s_thrsh = calculations[0]
t_close_ankle, t_close_height, t_far_ankle, t_far_height, t_win, t_thrsh = calculations[1]

scale_close = t_close_height / s_close_height
scale_far = t_far_height / s_far_height
scale_diff = scale_close - scale_far
s_ankle_diff = s_close_ankle - s_far_ankle
t_ankle_diff = t_close_ankle - t_far_ankle

if args.dry_run:
    print("Source params\twin: {:.2f}\tz-threshold: {:.2f}".format(s_win, s_thrsh))
    print("Source far\tankle: {:.2f}\theight: {:.2f}".format(s_far_ankle, s_far_height))
    print("Source close\tankle: {:.2f}\theight: {:.2f}".format(s_close_ankle, s_close_height))
    print("")
    print("Target params\twin: {:.2f}\tz-threshold: {:.2f}".format(t_win, t_thrsh))
    print("Target far\tankle: {:.2f}\theight: {:.2f}".format(t_far_ankle, t_far_height))
    print("Target close\tankle: {:.2f}\theight: {:.2f}".format(t_close_ankle, t_close_height))
    print("")
    print("Scales\tfar: {:.2f}\tclose: {:.2f}\tdelta: {:.2f}".format(scale_far, scale_close, scale_diff))

    sys.exit()




prev_y = -1

if not source_paths.denorm_label_dir.exists():
    print("Moving denormalized source labels to {}".format(source_paths.denorm_label_dir))
    shutil.move(str(source_paths.label_dir), str(source_paths.denorm_label_dir))

source_paths.label_dir.mkdir(exist_ok=True)

if len(os.listdir(source_paths.label_dir)) == len(os.listdir(source_paths.denorm_label_dir)):
    print("{} normalized labels found, skipping normalization".format(len(os.listdir(source_paths.label_dir))))

else:
    for i, (image_path, label_path, norm_path) in enumerate(data_paths(source_paths, normalize=True)):
        if label_path.exists():
            continue

        points = np.load(norm_path)

        mean_ankle = calculate_mean_ankle_for_pose(points)
        y = prev_y if isinvalid(mean_ankle) else mean_ankle[1]

        if y == -1:
            origin = np.array([0,0])
            scale = 1
            tx = 0
        else:
            prev_y = y
            origin = mean_ankle
            a = (y - s_far_ankle) / s_ankle_diff
            scale = scale_far + a * scale_diff
            tx = np.array([origin[0], t_far_ankle + a * t_ankle_diff], dtype=float)


        # apply transform
        points = np.where(points >= 0, (points - origin) * scale + tx, points)

        with Image.open(str(image_path)) as img:
            w, h = img.size

        labels = np.zeros((h, w, 3), dtype=np.uint8)

        OpenposeLabeller.draw_labels(labels, points.round().astype(np.int32))

        cv.imwrite(str(label_path), labels)
