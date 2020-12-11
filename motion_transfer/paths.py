from types import SimpleNamespace
from pathlib import Path

def build_paths(args):
    paths = SimpleNamespace(root_dir = Path("./"))

    paths.models_dir            = paths.root_dir / "models"
    paths.pose_prototxt         = paths.models_dir / "body_25_pose_deploy.prototxt"
    paths.pose_model            = paths.models_dir / "body_25_pose_iter_584000.caffemodel"
    paths.densepose_base_cfg    = paths.models_dir / "Base-DensePose-RCNN-FPN.yaml"
    paths.densepose_cfg         = paths.models_dir / "densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml"
    paths.densepose_model       = paths.models_dir / "model_final_6e1ed1.pkl"
    paths.dlib_face_landmarks   = paths.models_dir / "shape_predictor_68_face_landmarks.dat"
    paths.dlib_face_detector    = paths.models_dir / "mmod_human_face_detector.dat"


    paths.dataset_dir           = Path(args.dataroot)
    paths.train_img_dir         = paths.dataset_dir / "train_img"
    paths.train_label_dir       = paths.dataset_dir / "train_label"

    try:
        if args.name is not None and args.results_dir is not None:
            paths.results_dir   = paths.root_dir / args.results_dir / args.name
    except AttributeError:
        pass

    return paths
