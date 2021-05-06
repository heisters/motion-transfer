import wget
import dlib
import bz2
import cv2 as cv
import math
import numpy as np
from enum import Enum
from imutils import face_utils
import sys
from .paths import data_paths
from .video_utils import crop_frame

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

def build_label_colormap(offset):
    start = 1 + offset
    n = 256 - offset
    gammut = np.append(np.arange(start, 256, 1, dtype=np.uint8), [255]).reshape(n, 1)
    cmap = np.dstack((gammut, gammut, gammut)).astype(np.uint8)
    return cmap

def pose_point_is_valid(point):
    return (point >= 0).all()


class FaceLabeller(object):
    def __init__(self, paths, cmap, label_base, exclude_landmarks=None):
        self.detector, self.predictor, self.landmarks = self.build_face_detector(paths, exclude_landmarks=exclude_landmarks)
        self.colors = self.build_face_label_colormap(cmap, label_base, len(self.landmarks))

    def detect(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        face_rects = self.detector(gray, 1)
        faces = []
        for (i, rect) in enumerate(face_rects):
            face = self.predictor(gray, rect.rect)
            face = face_utils.shape_to_np(face, dtype=np.int32)
            faces.append(face)

        faces = np.array(faces, dtype=np.int32)
        return faces

    def label(self, image, labels):
        for face in self.detect(image):

            labels = self.visualize_facial_landmarks(labels, face, alpha=1.0)

        return labels

    @classmethod
    def build_face_label_colormap(self, cmap, base, nlabels):
        return [tuple(map(int, r[0])) for r in cmap[base : base + nlabels]]

    @classmethod
    def build_face_detector(self, paths, exclude_landmarks=None):
        face_detector = dlib.cnn_face_detection_model_v1(str(paths.dlib_face_detector))
        face_predictor = dlib.shape_predictor(str(paths.dlib_face_landmarks))

        landmarks = face_utils.FACIAL_LANDMARKS_IDXS.copy()
        if exclude_landmarks is not None:
            for name in exclude_landmarks:
                del landmarks[name]

        return (face_detector, face_predictor, landmarks)

    def visualize_facial_landmarks(self, image, face, alpha=0.75):
        # create two copies of the input image -- one for the
        # overlay and one for the final output image
        overlay = image.copy()
        output = image.copy()

        # loop over the facial landmark regions individually
        for (i, name) in enumerate(self.landmarks.keys()):
            # grab the (x, y)-coordinates associated with the
            # face landmark
            (j, k) = self.landmarks[name]
            pts = face[j:k]

            # check if are supposed to draw the jawline
            if name == "jaw":
                # since the jawline is a non-enclosed facial region,
                # just draw lines between the (x, y)-coordinates
                for l in range(1, len(pts)):
                    ptA = tuple(pts[l - 1])
                    ptB = tuple(pts[l])
                    cv.line(overlay, ptA, ptB, self.colors[i], 2)

            # otherwise, compute the convex hull of the facial
            # landmark coordinates points and display it
            else:
                hull = cv.convexHull(pts)
                cv.drawContours(overlay, [hull], -1, self.colors[i], -1)

        # apply the transparent overlay
        cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # return the output image
        return output

class Labeller(object):
    Strategies = Enum('Strategies', 'densepose openpose')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--label-with', help="Choose labelling strategy", choices=["openpose", "densepose"], default="openpose")
        p.add_argument('--exclude-landmarks', help="CSV list of facial landmarks to exclude from the labels", type=str)
        p.add_argument('--label-face', help="Add labels for the face (default on)", dest='label_face', action='store_true')
        p.add_argument('--no-label-face', help="Do not add labels for the face", dest='label_face', action='store_false')
        p.add_argument('--label-offset', help="Offset to add to all labels for combining datasets with different label ranges", type=int, default=0)

        p.set_defaults(label_face=True)
        return p

    @classmethod
    def build_from_arguments(cls, args, paths, label_offset=None):
        exclude_landmarks = set(args.exclude_landmarks.split(',')) if args.exclude_landmarks is not None else None
        label_face = args.label_face
        label_offset = args.label_offset if label_offset is None else label_offset
        labeller = Labeller.build(Labeller.Strategies[args.label_with], paths, exclude_landmarks=exclude_landmarks, label_face=label_face, label_offset=label_offset)
        return labeller

    @classmethod
    def build(cls, strategy, paths, exclude_landmarks=None, label_face=True, label_offset=0):
        if strategy == cls.Strategies.densepose:
            return DenseposeLabeller(paths, exclude_landmarks=exclude_landmarks, label_face=label_face, label_offset=label_offset)
        elif strategy == cls.Strategies.openpose:
            return OpenposeLabeller(paths, exclude_landmarks=exclude_landmarks, label_face=label_face, label_offset=label_offset)
        else:
            raise NotImplementedError("Unrecognized labeling strategy: {}".format(strategy))



class DenseposeLabeller(Labeller):
    num_points = 26
    def __init__(self, paths, exclude_landmarks=None, label_face=True, label_offset=0):

        cfg = get_cfg()
        add_densepose_config(cfg)
        add_hrnet_config(cfg)
        cfg.merge_from_file(str(paths.densepose_cfg))
        cfg.MODEL.WEIGHTS = str(paths.densepose_model)
        cfg.freeze()
        self.pose_predictor = DefaultPredictor(cfg)

        cmap = build_label_colormap(label_offset)

        if label_face:
            self.face_labeller = FaceLabeller(paths, cmap, self.num_points, exclude_landmarks=exclude_landmarks)

        self.visualizer = DensePoseMaskedColormapResultsVisualizer(_extract_i_from_iuvarr, _extract_i_from_iuvarr, cmap=cmap, alpha=1.0, val_scale=1.0)
        self.extractor = create_extractor(visualizer)

    def label_image(self, image, image_path, label_path, norm_path):
        labels = np.zeros(image.shape, dtype=np.uint8)

        # Pose detection
        outputs = self.pose_predictor(image)['instances']
        data = self.extractor(outputs)
        if data is not None and data[1] is not None:
            self.visualizer.visualize(labels, data)
        else:
            print("No pose detected for frame {}".format(image_path))

        # Face detection
        if self.face_labeller is not None:
            labels = self.face_labeller.label(image, labels)

        cv.imwrite(str(label_path), labels)

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
class OpenposeLabeller(Labeller):
    num_points = 25
    threshold = 0.1
    point_pairs = [[1, 0], [1, 2], [1, 5], 
                    [2, 3], [3, 4], [5, 6], 
                    [6, 7], [0, 15], [15, 17], 
                    [0, 16], [16, 18], [1, 8],
                    [8, 9], [9, 10], [10, 11], 
                    [11, 22], [22, 23], [11, 24],
                    [8, 12], [12, 13], [13, 14], 
                    [14, 19], [19, 20], [14, 21]]

    def __init__(self, paths, exclude_landmarks=None, label_face=True, label_offset=0):

        self.net = cv.dnn.readNetFromCaffe(str(paths.pose_prototxt), str(paths.pose_model))
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        self.cmap = build_label_colormap(label_offset)
        if label_face:
            self.face_labeller = FaceLabeller(paths, self.cmap, self.num_points, exclude_landmarks=exclude_landmarks)

    def draw_labels(self, cvimg, points):
        for li, (a, b) in enumerate(self.point_pairs):
            label = (int(self.cmap[li][0][0]), int(self.cmap[li][0][1]), int(self.cmap[li][0][2]))

            pa = points[a]
            pb = points[b]
            if pose_point_is_valid(pa) and pose_point_is_valid(pb):
                coords = np.array([pa, pb])
                center = tuple(np.round(np.mean(coords, 0)).astype(int))
                D = pa - pb
                L = np.linalg.norm(D)
                angle = math.degrees(math.atan2(D[1], D[0]))
                poly = cv.ellipse2Poly(center, (int(L / 2), 4), int(angle), 0, 360, 1)
                cv.fillConvexPoly(cvimg, poly, label)

        return cvimg

    def detect_pose(self, image):
        iw = image.shape[1]
        ih = image.shape[0]
        size = (368,368)
        blob = cv.dnn.blobFromImage(image, 1.0 / 255.0, size, (0,0,0), swapRB=False, crop=False)
        self.net.setInput(blob)

        output = self.net.forward()
        ow = output.shape[3]
        oh = output.shape[2]


        points = []
        for i in range(self.num_points):
            confidence = output[0, i, :, :]
            minval, prob, minloc, point = cv.minMaxLoc(confidence)
            x = ( iw * point[0] ) / ow
            y = ( ih * point[1] ) / oh

            if prob > self.threshold:
                points.append([int(x), int(y)])
            else:
                points.append([-1, -1])

        points = np.array(points, dtype=np.int32)
        return points

    def label_image(self, image, image_path, label_path, norm_path, resize=None, crop=None):
        labels = np.zeros(image.shape, dtype=np.uint8)

        iw = image.shape[1]
        ih = image.shape[0]

        points = self.detect_pose(image)


        # labelling
        self.draw_labels(labels, points)

        # Face detection
        if self.face_labeller is not None:
            labels = self.face_labeller.label(image, labels)

        center = None
        if crop is not None:
            center = np.ma.masked_less(points, 0).mean(axis=0)
            if center.all() is np.ma.masked: center = np.array([iw / 2, ih / 2])
            center = list(center.round().astype(int))
            points = points - center + [crop[0]/2,crop[1]/2]
            labels = crop_frame(labels, crop, center)

        cv.imwrite(str(label_path), labels)

        # normalization data
        if norm_path is not None:
            np.save(norm_path, points)

        return center
