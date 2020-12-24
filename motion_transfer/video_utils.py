import subprocess
import shlex
import cv2 as cv
import os
from tqdm import tqdm
from enum import Enum

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

def decimate_video(path, output_dir, limit=None, trim=(0.0, -1.0), subsample=1, subsample_offset=0, resize=None, flip=None):
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
        if (i + subsample_offset) % subsample != 0: continue

        imgpath = output_dir / '{:05}.png'.format(i)

        if not imgpath.exists():
            success, frame = cap.retrieve()
            if not success: break

            if resize is not None: frame = cv.resize(frame, resize, interpolation=cv.INTER_AREA)
            if flip is not None: frame = cv.flip(frame, flip)
            cv.imwrite(str(imgpath), frame)
