### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import sys
from tqdm import tqdm
from PIL import Image
import torch
import shutil

from src.paths import build_paths

# pix2pixHD imports
sys.path.append('./vendor/pix2pixHD/')
from options.test_options import TestOptions
from models.models import create_model
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
import util.util as util


opt = TestOptions()
opt.parser.add_argument('-d', '--dataset', help='Name of the dataset', required=True)
opt.parser.add_argument('--fps', type=float, default=24., help='frame per second for video generation')
opt = opt.parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# additional enforced options for video
opt.video_mode = True
opt.label_nc = 0
opt.no_instance = True
opt.resize_or_crop = "none"

paths = build_paths(opt)

# this directory will contain the generated videos
output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# this directory will contain the frames to build the video
frame_dir = os.path.join(opt.checkpoints_dir, opt.name, 'frames')
if os.path.isdir(frame_dir):
    shutil.rmtree(frame_dir)
os.mkdir(frame_dir)


#frame_index = 1
#
#if opt.start_from == "noise":
#    # careful, default value is 1024x512
#    t = torch.rand(1, 3, opt.fineSize, opt.loadSize)
#
#elif opt.start_from  == "video" and opt.gen_strategy == "recursive":
#    # use initial frames from the dataset
#    print("Generating seed tensors...")
#    for data in tqdm(dataset):
#        t = data['left_frame']
#        video_utils.save_tensor(
#            t,
#            frame_dir + "/frame-%s.jpg" % str(frame_index).zfill(5),
#            text="original video",
#        )
#        frame_index += 1
#
#
#else:
#    # use specified image
#    filepath = opt.start_from
#    if os.path.isfile(filepath):
#        t = video_utils.im2tensor(Image.open(filepath))
#        for i in range(50):
#            video_utils.save_tensor(
#                t,
#                frame_dir + "/frame-%s.jpg" % str(frame_index).zfill(5),
#            )
#            frame_index += 1
#
#current_frame = None
#if opt.gen_strategy == "recursive":
#    current_frame = t
#elif opt.gen_strategy == "map":
#    frame_index = 1

duration_s = opt.how_many / opt.fps
video_id = "epoch-%s_%s_%.1f-s_%.1f-fps" % (
    str(opt.which_epoch),
    opt.name,
    duration_s,
    opt.fps
)

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
        frame_dir + "/" + frame_file_glob,
        ' -filter:v "crop=1280:720:0:16"' if crop_to_720p else "",
        "-vf reverse " if reverse else "",
        video_path
    )
    print(command)
    print("building video from frames")
    p = subprocess.Popen(shlex.split(command), shell=False)
    p.communicate()

for i, framefn in tqdm(zip(range(opt.how_many), os.listdir(paths.source_label_dir)), total=opt.how_many):
    label_img = Image.open(str(paths.source_label_dir / framefn))
    label_tensor = get_transform(opt, get_params(opt, label_img.size))(label_img.convert('RGB'))
    inferred = model.inference(label_tensor, None, None)

    img_nda = util.tensor2im(inferred.data[0])
    img_pil = Image.fromarray(img_nda)
    img_pil.save(frame_dir + "/frame-%s.jpg" % str(i + 1).zfill(5))

video_path = output_dir + "/" + video_id + ".mp4"
while os.path.isfile(video_path):
    video_path = video_path[:-4] + "-.mp4"

video_from_frame_directory(
    frame_dir, 
    video_path, 
    framerate=opt.fps, 
    crop_to_720p=False,
    reverse=False
)

print("video ready:\n%s" % video_path)
