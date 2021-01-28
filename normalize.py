#!/usr/bin/env python

import argparse
import numpy as np
import math
from scipy import stats
import sys

from motion_transfer.paths import build_paths, data_paths


def parse_arguments():
    p = argparse.ArgumentParser(description="Normalize source video labels to the space of the target video")
    p.add_argument('--dataroot', type=str)
    p.add_argument('--limit', help="Only process the first N data samples", type=int)
    p.add_argument('--window', help="Starting ankle position window size as a floating-point percentage", type=float, default=0.05)
    p.add_argument('--zthreshold', help="Starting z-score threshold for height consideration", type=float, default=1.0)
    p.add_argument('--dry-run', help="Print calculated statistics and quit", action="store_true")
    return p.parse_args()

def calculate_metrics_for_pose(points):
    idx_lankle = 14
    idx_rankle = 11
    idx_nose = 0
    idx_neck = 1
    idx_hip = 8

    lankle = points[idx_lankle]
    rankle = points[idx_rankle]
    nose = points[idx_nose]
    neck = points[idx_neck]
    hip = points[idx_hip]

    if nose is None or lankle is None or rankle is None or hip is None or neck is None:
        return None


    nose = np.array(nose)
    lankle = np.array(lankle)
    rankle = np.array(rankle)

    mean_ankle = ( lankle + rankle ) / 2.0

    # do not consider poses where the body is inverted, even partially
    yvalues = [nose[1], neck[1], hip[1], mean_ankle[1]]
    if sorted(yvalues) != yvalues:
        return None

    height = np.sqrt(np.sum((nose - mean_ankle) ** 2))

    return (mean_ankle, nose, height)

def calculate_metrics(paths, limit=None):
    ankles, noses, heights, images = [], [], [], []

    for i, (image_path, label_path, norm_path) in enumerate(data_paths(paths, normalize=True)):
        points = np.load(norm_path, allow_pickle=True)
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
        height = np.amax(heights_over_z)

    return ankle_y, height

def calculate_close_and_far_transformations(ankles, noses, heights, window=0.05, zthreshold=1.0):
    window_inc = 0.05
    max_window = 0.3
    zthreshold_inc = -0.1
    min_zthreshold = -1.0

    while True:
        a_close, h_close = calculate_ankle_and_height(ankles, noses, heights, method=np.amax, window=window, zthreshold=zthreshold)
        a_far, h_far = calculate_ankle_and_height(ankles, noses, heights, method=np.amin, window=window, zthreshold=zthreshold)

        if not math.isnan(h_close) and not math.isnan(h_far) and a_close > a_far and h_close > h_far:
            break


        zthreshold += zthreshold_inc
        window += window_inc

        if window > max_window or zthreshold < min_zthreshold:
            raise Exception("Could not find a window and z-score threshold that result in expected transformations")

    return a_close, h_close, a_far, h_far, window, zthreshold



args = parse_arguments()
source_paths = build_paths(args, directory_prefix='test')
target_paths = build_paths(args, directory_prefix='train')

s_ankles, s_noses, s_heights, s_images = calculate_metrics(source_paths, limit=args.limit)
t_ankles, t_noses, t_heights, t_images = calculate_metrics(target_paths, limit=args.limit)

s_close_ankle, s_close_height, s_far_ankle, s_far_height, s_win, s_thrsh = calculate_close_and_far_transformations(s_ankles, s_noses, s_heights, window=args.window, zthreshold=args.zthreshold)
t_close_ankle, t_close_height, t_far_ankle, t_far_height, t_win, t_thrsh = calculate_close_and_far_transformations(t_ankles, t_noses, t_heights, window=args.window, zthreshold=args.zthreshold)

if args.dry_run:
    print("Source close\tankle: {:.2f}\theight: {:.2f}".format(s_close_ankle, s_close_height))
    print("Source far\tankle: {:.2f}\theight: {:.2f}".format(s_far_ankle, s_far_height))
    print("Source params\twin: {:.2f}\tz-threshold: {:.2f}".format(s_win, s_thrsh))
    print("Target close\tankle: {:.2f}\theight: {:.2f}".format(t_close_ankle, t_close_height))
    print("Target far\tankle: {:.2f}\theight: {:.2f}".format(t_far_ankle, t_far_height))
    print("Target params\twin: {:.2f}\tz-threshold: {:.2f}".format(t_win, t_thrsh))

    #print("s_close:")
    #print(" ".join(s_images[s_heights == s_close_height]))
    #print("s_far:")
    #print(" ".join(s_images[s_heights == s_far_height]))
    #print("t_close:")
    #print(" ".join(t_images[t_heights == s_close_height]))
    #print("t_far:")
    #print(" ".join(t_images[t_heights == s_far_height]))
    sys.exit()

