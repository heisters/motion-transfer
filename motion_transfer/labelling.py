import wget
import dlib
import bz2
import cv2 as cv
import numpy as np
from imutils import face_utils
import sys
from .paths import data_paths

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


def fetch_model(name, path, url):
    if path.exists():
        print("%s OK: %s" % (name, path))
    else:
        wget.download(url, str(path))
        if url.endswith(".bz2"):
            z = bz2.BZ2File(str(path))
            data = z.read()
            open(str(path), 'wb').write(data)

def fetch_models(paths):
    fetch_model("Pose model", paths.pose_model, "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel")
    fetch_model("Pose prototxt", paths.pose_prototxt, "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt")
    fetch_model("Densepose base config", paths.densepose_base_cfg, "https://raw.githubusercontent.com/facebookresearch/detectron2/master/projects/DensePose/configs/Base-DensePose-RCNN-FPN.yaml")
    fetch_model("Denspose config", paths.densepose_cfg, "https://raw.githubusercontent.com/facebookresearch/detectron2/master/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml")
    fetch_model("Densepose model", paths.densepose_model, "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2_s1x/173294801/model_final_6e1ed1.pkl")
    fetch_model("DLIB face landmarks", paths.dlib_face_landmarks, "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    fetch_model("DLIB CNN face detector", paths.dlib_face_detector, "http://dlib.net/files/mmod_human_face_detector.dat.bz2")


def label_images(paths, normalize=False):
    for image_path, label_path, norm_path in data_paths(paths, normalize=normalize):

        if not label_path.exists() or ( normalize and norm_path is not None and not norm_path.exists() ):
            image = cv.imread(str(image_path))
            if image is None:
                raise RuntimeError("Unable to read image: %s" % image_path)

            yield image, image_path, label_path, norm_path

def build_label_colormap():
    gammut = np.append(np.arange(1, 256, 1, dtype=np.uint8), [255]).reshape(256, 1)
    cmap = np.dstack((gammut, gammut, gammut)).astype(np.uint8)
    return cmap

def build_face_detector(paths, exclude_landmarks=None):
    face_detector = dlib.cnn_face_detection_model_v1(str(paths.dlib_face_detector))
    face_predictor = dlib.shape_predictor(str(paths.dlib_face_landmarks))

    landmarks = face_utils.FACIAL_LANDMARKS_IDXS.copy()
    if exclude_landmarks is not None:
        for name in exclude_landmarks:
            del landmarks[name]

    return (face_detector, face_predictor, landmarks)

def visualize_facial_landmarks(image, shape, colors, landmarks, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(landmarks.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = landmarks[name]
        pts = shape[j:k]

        # check if are supposed to draw the jawline
        if name == "jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv.convexHull(pts)
            cv.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return output


def label_face(face_detector, face_predictor, landmarks, image, labels, face_colors):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_rects = face_detector(gray, 1)
    for (i, rect) in enumerate(face_rects):
        shape = face_predictor(gray, rect.rect)
        shape = face_utils.shape_to_np(shape)

        labels = visualize_facial_landmarks(labels, shape, face_colors, landmarks, alpha=1.0)

    return labels


def make_labels_with_openpose(paths, exclude_landmarks=None, label_face=True, normalize=False):
    net = cv.dnn.readNetFromCaffe(str(paths.pose_prototxt), str(paths.pose_model))
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

    face_detector, face_predictor, landmarks = (None,None,None)
    if label_face:
        face_detector, face_predictor, landmarks = build_face_detector(paths, exclude_landmarks=exclude_landmarks)


    # from https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    # {0,  "Nose"},
    # {1,  "Neck"},
    # {2,  "RShoulder"},
    # {3,  "RElbow"},
    # {4,  "RWrist"},
    # {5,  "LShoulder"},
    # {6,  "LElbow"},
    # {7,  "LWrist"},
    # {8,  "MidHip"},
    # {9,  "RHip"},
    # {10, "RKnee"},
    # {11, "RAnkle"},
    # {12, "LHip"},
    # {13, "LKnee"},
    # {14, "LAnkle"},
    # {15, "REye"},
    # {16, "LEye"},
    # {17, "REar"},
    # {18, "LEar"},
    # {19, "LBigToe"},
    # {20, "LSmallToe"},
    # {21, "LHeel"},
    # {22, "RBigToe"},
    # {23, "RSmallToe"},
    # {24, "RHeel"},
    # {25, "Background"}

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

    cmap = build_label_colormap()
    if face_detector is not None:
        face_colors = [tuple(map(int, r[0])) for r in cmap[num_points : num_points + len(landmarks)]]

    for image, image_path, label_path, norm_path in label_images(paths, normalize=normalize):
        labels = np.zeros(image.shape, dtype=np.uint8)

        # Pose detection
        iw = image.shape[1]
        ih = image.shape[0]
        size = (368,368)#(iw, ih) # others set this to (368, 368)
        blob = cv.dnn.blobFromImage(image, 1.0 / 255.0, size, (0,0,0), swapRB=False, crop=False)
        net.setInput(blob)

        output = net.forward()
        ow = output.shape[3]
        oh = output.shape[2]


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

        # labelling
        label = 1
        for a, b in point_pairs:
            if points[a] and points[b]:
                cv.line(labels, points[a], points[b], label, 3)
            label += 1

        # Face detection
        if face_detector is not None:
            labels = label_face(face_detector, face_predictor, landmarks, image, labels, face_colors)

        cv.imwrite(str(label_path), labels)

        # normalization data
        if norm_path is not None:
            np.save(norm_path, np.array(points))


def make_labels_with_densepose(paths, exclude_landmarks=None, label_face=True, normalize=False):
    if normalize:
        raise NotImplementedError("Normalization is not yet supported with the densepose labeller")


    cfg = get_cfg()
    add_densepose_config(cfg)
    add_hrnet_config(cfg)
    cfg.merge_from_file(str(paths.densepose_cfg))
    cfg.MODEL.WEIGHTS = str(paths.densepose_model)
    cfg.freeze()
    pose_predictor = DefaultPredictor(cfg)

    face_detector, face_predictor, landmarks = (None,None,None)
    if label_face:
        face_detector, face_predictor, landmarks = build_face_detector(paths, exclude_landmarks=exclude_landmarks)

    cmap = build_label_colormap()

    if face_detector is not None:
        face_colors = [tuple(map(int, r[0])) for r in cmap[26 : 26 + len(landmarks)]]

    for image, image_path, label_path, _ in label_images(paths):
        labels = np.zeros(image.shape, dtype=np.uint8)

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
        if face_detector is not None:
            labels = label_face(face_detector, face_predictor, landmarks, image, labels, face_colors)

        cv.imwrite(str(label_path), labels)

def make_labels(labeller, paths, exclude_landmarks=None, label_face=True, normalize=False):
    if labeller == 'openpose':
        make_labels_with_openpose(paths, exclude_landmarks=exclude_landmarks, label_face=label_face, normalize=normalize)
    elif labeller == "densepose":
        make_labels_with_densepose(paths, exclude_landmarks=exclude_landmarks, label_face=label_face, normalize=normalize)
