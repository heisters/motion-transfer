#!/usr/bin/env python

import os
import sys
import argparse
import cv2 as cv
import dlib
import wget
import bz2
import numpy as np
from imutils import face_utils
from tqdm import tqdm
from pathlib import Path

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
sys.path.append('./vendor/detectron2/projects/DensePose')
from densepose import add_densepose_config, add_hrnet_config
from densepose.vis.densepose_results import DensePoseMaskedColormapResultsVisualizer, _extract_i_from_iuvarr
from densepose.vis.extractor import CompoundExtractor, DensePoseResultExtractor, create_extractor

from motion_transfer.paths import build_paths



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
    p.add_argument('--label-face', help="Choose labelling strategy", dest='label_face', action='store_true')
    p.add_argument('--no-label-face', help="Choose labelling strategy", dest='label_face', action='store_false')

    p.set_defaults(label_face=True)

    return p.parse_args()

def create_directories(paths):
    for k, v in vars(paths).items():
        if k.endswith('_dir'):
            v.mkdir(exist_ok=True)

def decimate_video(path, output_dir, limit=None, trim=(0.0, -1.0), subsample=1, resize=None, flip=None):
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened(): return

    nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if limit is not None: nframes = min(nframes, limit)
    fps = float(cap.get(cv.CAP_PROP_FPS))

    start_frame = int(fps * trim[0])
    end_frame = nframes if trim[1] < 0.0 else max(start_frame + 1, int(fps * trim[1]))

    if (end_frame - start_frame) // subsample == len(os.listdir(str(output_dir))):
        print("Found %d frames, skipping." % (end_frame - start_frame))
        return

    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    for i in tqdm(range(end_frame - start_frame)):
        # do this here to increment frame counter, regardless of whether the file exists
        if not cap.grab(): break
        if i % subsample != 0: continue

        imgpath = output_dir / '{:05}.png'.format(i)

        if not imgpath.exists():
            success, frame = cap.retrieve()
            if not success: break

            if resize is not None: frame = cv.resize(frame, resize, interpolation=cv.INTER_AREA)
            if flip is not None: frame = cv.flip(frame, flip)
            cv.imwrite(str(imgpath), frame)

def get_model(name, path, url):
    if path.exists():
        print("%s OK: %s" % (name, path))
    else:
        wget.download(url, str(path))
        if url.endswith(".bz2"):
            z = bz2.BZ2File(str(path))
            data = z.read()
            open(str(path), 'wb').write(data)

def get_models(paths):
    get_model("Pose model", paths.pose_model, "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel")
    get_model("Pose prototxt", paths.pose_prototxt, "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt")
    get_model("Densepose base config", paths.densepose_base_cfg, "https://raw.githubusercontent.com/facebookresearch/detectron2/master/projects/DensePose/configs/Base-DensePose-RCNN-FPN.yaml")
    get_model("Denspose config", paths.densepose_cfg, "https://raw.githubusercontent.com/facebookresearch/detectron2/master/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml")
    get_model("Densepose model", paths.densepose_model, "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2_s1x/173294801/model_final_6e1ed1.pkl")
    get_model("DLIB face landmarks", paths.dlib_face_landmarks, "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    get_model("DLIB CNN face detector", paths.dlib_face_detector, "http://dlib.net/files/mmod_human_face_detector.dat.bz2")


def label_images(img_dir, label_dir):
    for image_path_str in tqdm(os.listdir(str(img_dir))):
        image_path = img_dir / image_path_str
        label_path = label_dir / image_path.name

        if not label_path.exists():
            image = cv.imread(str(image_path))
            if image is None:
                raise RuntimeError("Unable to read image: %s" % image_path)

            yield image, image_path, label_path

def label_images_with_openpose(net, img_dir, label_dir):
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

    for image, image_path, label_path in label_images(img_dir, label_dir):
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

def label_images_with_densepose(pose_predictor, face_detector, face_predictor, img_dir, label_dir):
    gammut = np.append(np.arange(1, 256, 1, dtype=np.uint8), [255]).reshape(256, 1)
    cmap = np.dstack((gammut, gammut, gammut)).astype(np.uint8)
    face_colors = [tuple(map(int, r[0])) for r in cmap[26:34]]

    for image, image_path, label_path in label_images(img_dir, label_dir):
        labels = np.zeros(image.shape, np.uint8)

        # Pose detection
        outputs = pose_predictor(image)['instances']
        visualizer = DensePoseMaskedColormapResultsVisualizer(_extract_i_from_iuvarr, _extract_i_from_iuvarr, cmap=cmap, alpha=1.0, val_scale=1.0)
        extractor = create_extractor(visualizer)
        data = extractor(outputs)
        if data is not None and data[1] is not None:
            visualizer.visualize(labels, data)
        else:
            print("No pose detected for frame {}".format(image_path))

        # Face detection
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        face_rects = face_detector(gray, 1)
        for (i, rect) in enumerate(face_rects):
            shape = face_predictor(gray, rect.rect)
            shape = face_utils.shape_to_np(shape)

            labels = face_utils.visualize_facial_landmarks(labels, shape, colors=face_colors, alpha=1.0)

        cv.imwrite(str(label_path), labels)


def make_labels_with_openpose(paths):
    net = cv.dnn.readNetFromCaffe(str(paths.pose_prototxt), str(paths.pose_model))
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    print("Labeling")
    label_images_with_openpose(net, paths.train_img_dir, paths.train_label_dir)

def make_labels_with_densepose(paths):
    cfg = get_cfg()
    add_densepose_config(cfg)
    add_hrnet_config(cfg)
    cfg.merge_from_file(str(paths.densepose_cfg))
    cfg.MODEL.WEIGHTS = str(paths.densepose_model)
    cfg.freeze()
    pose_predictor = DefaultPredictor(cfg)

    face_detector = dlib.cnn_face_detection_model_v1(str(paths.dlib_face_detector))
    face_predictor = dlib.shape_predictor(str(paths.dlib_face_landmarks))


    print("Labeling")
    label_images_with_densepose(pose_predictor, face_detector, face_predictor, paths.train_img_dir, paths.train_label_dir)

def make_labels(labeller, paths):
    if labeller == 'openpose':
        make_labels_with_openpose(paths)
    elif labeller == "densepose":
        make_labels_with_densepose(paths)

#
# Setup
#

args = parse_arguments()
paths = build_paths(args)
paths.input = Path(args.input)


resize = tuple(map(int, args.resize.split('x'))) if args.resize is not None else None
trim = tuple(map(float, args.trim.split(':'))) if args.trim is not None else None
flip = {'v': 0, 'h': 1, 'hv': -1, 'vh': -1}[args.flip] if args.flip is not None else None

print("Creating directory hierarchy")
create_directories(paths)
print("Fetching models")
get_models(paths)
print("Decimating")
decimate_video(paths.input, paths.train_img_dir, trim=trim, subsample=args.subsample, resize=resize, flip=flip)
print("Labeling frames with %s" % args.label_with)
make_labels(args.label_with, paths)
