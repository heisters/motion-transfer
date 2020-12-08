### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import sys
from tqdm import tqdm
from PIL import Image
import torch
import shutil
import subprocess
import shlex

from src.paths import build_paths

# pix2pixHD imports
sys.path.append('./vendor/pix2pixHD/')
from options.base_options import BaseOptions
from models.models import create_model
from data.data_loader import CreateDataLoader
import util.util as util


# edited version of TestOptions
class VideoOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        #self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        #self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--results_dir', type=str, default='results', help='saves results here.')
        #self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        #self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
        self.parser.add_argument('--how_many', type=int, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.parser.add_argument('-d', '--dataset', help='Name of the dataset', required=True) # needed for paths, temporarily
        self.parser.add_argument('--fps', type=float, default=24., help='frame per second for video generation')
        self.isTrain = False

opt = VideoOptions()
opt = opt.parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.resize_or_crop = "none"

paths = build_paths(opt)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()


nframes = opt.how_many if opt.how_many is not None else len(dataset)
duration_s = nframes / opt.fps
video_id = "epoch-%s_%s_%ds_%dfps" % (
    str(opt.which_epoch),
    opt.name,
    duration_s,
    opt.fps
)

video_path = (paths.results_dir / video_id).with_suffix(".mp4")
i = 0
while video_path.exists():
    i += 1
    video_path = (paths.results_dir / (video_id + ("-%d" % (i)))).with_suffix(".mp4")
frame_dir = video_path.with_suffix('')
if frame_dir.exists(): shutil.rmtree(frame_dir)
frame_dir.mkdir(parents=True, exist_ok=True)

model = create_model(opt)

def save_tensor(tensor, path):
    """Saving a Torch image tensor into an image (with text)"""
    img_nda = util.tensor2im(tensor.data[0])
    img_pil = Image.fromarray(img_nda)

    if text != "":
        if text_pos == "auto":
            # top-right corner
            text_xpos = img_pil.width - 28 * len(text)
            text_ypos = 30
        else:
            text_xpos, text_ypos = text_pos

        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 50)
        draw.text((text_xpos, text_ypos), text, text_color, font=font)

    img_pil.save(path)

def video_from_frame_directory(frame_dir, video_path, frame_file_glob=r"frame-%05d.jpg", framerate=24, ffmpeg_verbosity=16, crop_to_720p=True, reverse=False):
    """Build a mp4 video from a directory frames
        note: crop_to_720p crops the top of 1280x736 images to get them to 1280x720
    """
    command = """ffmpeg -v %d -framerate %d -i %s -ss 1 -q:v 2%s %s%s""" % (
        ffmpeg_verbosity,
        framerate,
        str(frame_dir / frame_file_glob),
        ' -filter:v "crop=1280:720:0:16"' if crop_to_720p else "",
        "-vf reverse " if reverse else "",
        video_path
    )
    print(command)
    print("building video from frames")
    p = subprocess.Popen(shlex.split(command), shell=False)
    p.communicate()

for i, data in enumerate(tqdm(dataset)):
    if opt.how_many is not None and i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()

    inferred = model.inference(data['label'], data['inst'], data['image'])
    img_nda = util.tensor2im(inferred.data[0])
    img_pil = Image.fromarray(img_nda)
    img_pil.save(frame_dir / ("frame-%s.jpg" % str(i + 1).zfill(5)))


video_from_frame_directory(
    frame_dir, 
    video_path, 
    framerate=opt.fps, 
    crop_to_720p=False,
    reverse=False
)

print("video ready:\n%s" % video_path)
