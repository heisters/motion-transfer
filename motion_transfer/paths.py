from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
import os

def build_paths(args, directory_prefix=None):
    paths = SimpleNamespace(root_dir = Path("./"))
    try:
        if directory_prefix is None: directory_prefix = args.directory_prefix
    except AttributeError:
        pass

    paths.models_dir            = paths.root_dir / "models"
    paths.pose_prototxt         = paths.models_dir / "body_25_pose_deploy.prototxt"
    paths.pose_model            = paths.models_dir / "body_25_pose_iter_584000.caffemodel"
    paths.densepose_base_cfg    = paths.models_dir / "Base-DensePose-RCNN-FPN.yaml"
    paths.densepose_cfg         = paths.models_dir / "densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml"
    paths.densepose_model       = paths.models_dir / "model_final_6e1ed1.pkl"
    paths.dlib_face_landmarks   = paths.models_dir / "shape_predictor_68_face_landmarks.dat"
    paths.dlib_face_detector    = paths.models_dir / "mmod_human_face_detector.dat"


    paths.dataset_dir           = Path(args.dataroot)
    try:
        paths.img_dir           = paths.dataset_dir / "{}_img".format(directory_prefix)
    except AttributeError:
        paths.img_dir           = paths.dataset_dir / "train_img"

    try:
        if args.train_a is True:
            paths.img_dir = paths.dataset_dir / "train_A"
        elif args.train_b is True:
            paths.img_dir = paths.dataset_dir / "train_B"
        elif args.test_a is True:
            paths.img_dir = paths.dataset_dir / "test_A"
    except AttributeError:
        pass

    try:
        paths.label_dir         = paths.dataset_dir / "{}_label".format(directory_prefix)
    except AttributeError:
        paths.label_dir         = paths.dataset_dir / "train_label"

    try:
        paths.norm_dir          = paths.dataset_dir / "{}_norm".format(directory_prefix)
    except AttributeError:
        paths.norm_dir          = paths.dataset_dir / "train_norm"

    try:
        if args.name is not None and args.results_dir is not None:
            paths.results_dir   = paths.root_dir / args.results_dir / args.name
    except AttributeError:
        pass

    return paths

def data_paths(paths, normalize=False):
    img_dir = paths.img_dir
    label_dir = paths.label_dir
    norm_dir = paths.norm_dir

    for image_path_str in tqdm(os.listdir(str(img_dir))):
        image_path = img_dir / image_path_str
        label_path = label_dir / image_path.name
        norm_path = None
        if norm_dir is not None and normalize:
            norm_path = norm_dir / image_path.with_suffix('.npy').name

        yield image_path, label_path, norm_path
