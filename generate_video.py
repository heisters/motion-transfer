#!/usr/bin/env python
import os
import sys
from tqdm import tqdm
from PIL import Image
import torch
import shutil

from motion_transfer.paths import build_paths
from motion_transfer.video_options import VideoOptions
from motion_transfer.video_utils import video_from_frame_directory, video_filename_for_codec, Codec

# pix2pixHD imports
sys.path.append('./vendor/pix2pixHD/')
from models.models import create_model
from data.data_loader import CreateDataLoader
import util.util as util



opt = VideoOptions()
opt = opt.parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.resize_or_crop = "none"

paths = build_paths(opt)

codec = Codec[opt.codec]

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()


nframes = opt.how_many if opt.how_many is not None else len(dataset)
duration_s = nframes / opt.fps
video_id = "epoch-%s_%s_%ds_%dfps%s" % (
    str(opt.which_epoch),
    opt.name,
    duration_s,
    opt.fps,
    opt.output_suffix
)

frame_dir = paths.results_dir / video_id
video_path = video_filename_for_codec(paths.results_dir / video_id, codec)
frame_dir.mkdir(parents=True, exist_ok=True)

model = create_model(opt)


for i, data in enumerate(tqdm(dataset)):
    fn = frame_dir / ("frame-%s.png" % str(i + 1).zfill(5))
    if fn.exists(): continue

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
    img_pil.save(fn)


if not video_path.exists():
    video_from_frame_directory(
        frame_dir, 
        video_path, 
        framerate=opt.fps,
        codec=codec
    )

print("video ready:\n%s" % video_path)
