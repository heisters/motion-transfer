import subprocess
import shlex
import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from enum import Enum
from .paths import data_paths_for_image

Codec = Enum('Codec', 'x264 prores')

def video_filename_for_codec(path, codec):
    if codec == Codec.x264:
        return path.with_suffix('.mp4')
    elif codec == Codec.prores:
        return path.with_suffix('.mov')
    else:
        raise Exception('unrecognized codec: {}'.format(codec))


def video_from_frame_directory(frame_dir, video_path, codec=Codec.x264, frame_file_glob=r"frame-%05d.png", framerate=24, ffmpeg_verbosity=16):
    """Build a mp4 video from a directory frames
    """
    if codec == Codec.x264:
        encoding = '-vcodec libx264 -crf 20 -pix_fmt yuv420p'
    elif codec == Codec.prores:
        encoding = '-c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le'
    else:
        raise Exception('unrecognized codec: {}'.format(codec))

    command = """ffmpeg -v %d -framerate %d -f image2 -i %s %s %s""" % (
        ffmpeg_verbosity,
        framerate,
        str(frame_dir / frame_file_glob),
        encoding,
        video_path
    )
    print(command)
    print("building video from frames")
    p = subprocess.Popen(shlex.split(command), shell=False)
    p.communicate()

def crop_frame(image, dims, center):
    wa = dims[0] // 2
    ha = dims[1] // 2
    wb = dims[0] - wa
    hb = dims[1] - ha

    c = np.array(center) + [wa, ha] # account for padding

    xa = c[0] - wa
    xb = c[0] + wb
    ya = c[1] - ha
    yb = c[1] + hb

    o = np.pad(image, ((ha,hb),(wa,wb),(0,0)), mode='edge')
    o = o[ya:yb, xa:xb].copy()

    return o

def decimate_and_label_video(paths, labeller, limit=None, trim=(0.0, -1.0), subsample=1, subsample_offset=0, resize=None, crop=None, flip=None, normalize=False):
    cap = cv.VideoCapture(str(paths.input))
    if not cap.isOpened(): return

    nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if limit is not None: nframes = min(nframes, limit)
    fps = float(cap.get(cv.CAP_PROP_FPS))

    start_frame = int(fps * trim[0])
    end_frame = nframes if trim[1] < 0.0 else max(start_frame + 1, int(fps * trim[1]))


    frames_needed = (end_frame - start_frame) // subsample
    if (
            frames_needed == len(os.listdir(str(paths.img_dir))) and
            (
                labeller is None or
                (
                    frames_needed <= len(os.listdir(str(paths.label_dir))) or
                    (paths.denorm_label_dir.exists() and frames_needed <= len(os.listdir(str(paths.denorm_label_dir))))
                )
            ) and
            (not normalize or frames_needed == len(os.listdir(str(paths.norm_dir))))
       ):
        print("Found %d frames, skipping." % (end_frame - start_frame))
        return


    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    for i in tqdm(range(end_frame - start_frame)):
        # do this here to increment frame counter, regardless of whether the file exists
        if not cap.grab(): break
        if (i + subsample_offset) % subsample != 0: continue

        image_path = paths.img_dir / '{:05}.png'.format(i)
        _, label_path, norm_path = data_paths_for_image(paths, image_path.name, normalize=normalize)

        if not image_path.exists() or not label_path.exists() or ( norm_path is not None and not norm_path.exists() ):
            success, frame = cap.retrieve()
            if not success: break

            if resize is not None: frame = cv.resize(frame, resize, interpolation=cv.INTER_AREA)
            if flip is not None: frame = cv.flip(frame, flip)

            center = None
            if labeller is not None:
                center = labeller.label_image(frame, image_path, label_path, norm_path, resize=resize, crop=crop)

            if center is not None:
                frame = crop_frame(frame, crop, center)


            cv.imwrite(str(image_path), frame)
