from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
import os

def build_paths(args, directory_prefix=None, dataroot=None):
    paths = SimpleNamespace(root_dir = Path("./"))

    if directory_prefix is None:
        try:
            directory_prefix = args.directory_prefix
        except AttributeError:
            directory_prefix = 'train'

    paths.models_dir            = paths.root_dir / "models"
    paths.pose_prototxt         = paths.models_dir / "body_25_pose_deploy.prototxt"
    paths.pose_model            = paths.models_dir / "body_25_pose_iter_584000.caffemodel"
    paths.densepose_base_cfg    = paths.models_dir / "Base-DensePose-RCNN-FPN.yaml"
    paths.densepose_cfg         = paths.models_dir / "densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml"
    paths.densepose_model       = paths.models_dir / "model_final_6e1ed1.pkl"
    paths.dlib_face_landmarks   = paths.models_dir / "shape_predictor_68_face_landmarks.dat"
    paths.dlib_face_detector    = paths.models_dir / "mmod_human_face_detector.dat"


    dataroot                    = args.dataroot if dataroot is None else dataroot
    paths.dataset_dir           = Path(dataroot)
    paths.img_dir               = paths.dataset_dir / "{}_img".format(directory_prefix)

    try:
        if args.train_a is True:
            paths.img_dir = paths.dataset_dir / "train_A"
        elif args.train_b is True:
            paths.img_dir = paths.dataset_dir / "train_B"
        elif args.test_a is True:
            paths.img_dir = paths.dataset_dir / "test_A"
    except AttributeError:
        pass

    paths.label_dir             = paths.dataset_dir / "{}_label".format(directory_prefix)

    paths.norm_dir              = paths.dataset_dir / "{}_norm".format(directory_prefix)
    paths.denorm_label_dir      = paths.dataset_dir / "{}_denorm_label".format(directory_prefix)
    paths.norm_calculations     = paths.norm_dir / "calculations.npy"

    try:
        if args.results_dir is not None:
            name = None
            if args.results_name is not None:
                name = args.results_name
            elif args.name is not None:
                name = args.name

            if name is not None:
                paths.results_dir = paths.root_dir / args.results_dir / name
    except AttributeError:
        pass

    return paths

def create_directories(paths):
    for k, v in vars(paths).items():
        if k.endswith('_dir') and k != 'denorm_label_dir':
            v.mkdir(exist_ok=True)


def data_paths_for_image(paths, image_path, normalize=False):
    img_dir = paths.img_dir
    label_dir = paths.label_dir
    norm_dir = paths.norm_dir

    image_path = img_dir / image_path
    label_path = label_dir / image_path.name
    norm_path = None
    if norm_dir is not None and normalize:
        norm_path = norm_dir / image_path.with_suffix('.npy').name

    return image_path, label_path, norm_path

def data_paths_for_idx(paths, idx, normalize=False):
    image_path = '{:06}.png'.format(idx)
    return data_paths_for_image(paths, image_path, normalize=normalize)

def data_paths(paths, normalize=False):
    for image_path_str in tqdm(os.listdir(str(paths.img_dir))):
        yield data_paths_for_image(paths, image_path_str, normalize=normalize)
