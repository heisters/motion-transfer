import subprocess
import shlex
import cv2 as cv
import os
from tqdm import tqdm

def video_from_frame_directory(frame_dir, video_path, frame_file_glob=r"frame-%05d.jpg", framerate=24, ffmpeg_verbosity=16):
    """Build a mp4 video from a directory frames
        note: crop_to_720p crops the top of 1280x736 images to get them to 1280x720
    """
    command = """ffmpeg -v %d -framerate %d -f image2 -i %s -vcodec libx264 -crf 20 -pix_fmt yuv420p %s""" % (
        ffmpeg_verbosity,
        framerate,
        str(frame_dir / frame_file_glob),
        video_path
    )
    print(command)
    print("building video from frames")
    p = subprocess.Popen(shlex.split(command), shell=False)
    p.communicate()

def decimate_video(path, output_dir, limit=None, trim=(0.0, -1.0), subsample=1, resize=None, flip=None):
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
        if i % subsample != 0: continue

        imgpath = output_dir / '{:05}.png'.format(i)

        if not imgpath.exists():
            success, frame = cap.retrieve()
            if not success: break

            if resize is not None: frame = cv.resize(frame, resize, interpolation=cv.INTER_AREA)
            if flip is not None: frame = cv.flip(frame, flip)
            cv.imwrite(str(imgpath), frame)
