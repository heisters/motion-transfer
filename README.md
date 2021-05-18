Motion Transfer
===============

A set of tools that help train and run motion transfer models. Based on
[Everbody Dance Now](https://carolineec.github.io/everybody_dance_now/) (2019),
these tools facilitate the labelling of input video, tuning of training
parameters, training, and synthesis of video.

Overview
--------

`run.py` provides a high level interface to the multiple steps required for
training and using a model. `run.py` takes a YAML configuration file,
constructs a number of commands to task-specific scripts, and executes them.
The task scripts are idempotent, allowing the user to easily manage multiple
projects without tracking their current state.

The tasks for a basic motion transfer project are:

1. label data
2. normalize labels (optional)
3. train low-resolution model
4. train high-resolution model
5. generate outputs

Setup
-----

Clone the repo:

```bash
git clone --recurse-submodules https://github.com/heisters/motion-transfer.git
cd motion-transfer
```

Setup the environment with a fresh virtualenv:
```bash
mkvirtualenv motionxfer
workon motionxfer
python -m pip install -r requirements.txt
python -m pip install -r optional-requirements.txt # installs apex for fp16 training
```

Basic Usage
-----------

Put something like the following in a file called `config.yml`:

```yaml
SubjectA_to_SubjectB:
    width: 1024
    height: 576
    labels: 34
    data:
        - source: ~/media/SubjectA.mov
        - source: ~/media/SubjectB.mov
          options:
            - directory-prefix test
    normalize: false
    train:
        global:
            epochs: 12
            options:
                - no_flip
        local:
            epochs: 40
            options:
                - no_flip
    generate: true
```

This will label the Subject A video with OpenPose (poses) and dlib (faces),
train a model on Subject A and its labels, then generate a video using the
labels from the Subject B video without normalization. To execute it, simply
run:

```bash
./run.py SubjectA_to_SubjectB
```

### Basic Configuration

`config.yml` supports multiple projects, each under a key that provides its
name. This name will be used to name data, models, and generated video. Every
project requires three configuration parameters:

* `width` and `height` must be divisible by 32. This will set the default
  height of the data, the high-resolution model, and the generated video. The
  low-resolution model defaults to running at half this resolution. These
  values have a significant effect on memory usage, so they are limited by the
  memory available on your GPU(s). For help finding a resolution that is
  divisible by 32 but preserves a given aspect ratio, you can run
  `resize_divisible_by.py`
* `labels` is the number of labels in your training data, and depends on
  whether you are labelling with OpenPose or DensePose, labelling faces, and/or
  including multiple distinct people/label spaces in your data. **If you are
  building an unlabelled model (straight RGB values), this should be set to
  0.** Otherwise, the values are:

  * OpenPose: 26 (25 labels + 1 for "nothing")
  * DensePose: 27
  * Faces: add 8


Advanced Usage
--------------

`./run.py` and the underlying scripts do their best to avoid unnecessary work,
and pick up where they left off quickly if you had to abort a run for some
reason. If you have changed your config, and you want to rebuild some part of
the project, you will need to remove underlying data manually. For instance,
to rebuild a project from scratch:

```bash
rm -r data/SubjectA checkpoints/SubjectA_{local,global} results/SubjectA
./run.py SubjectA
```

If you wanted to only rebuild test data and rerun video generation:

```bash
rm -r data/SubjectA/test_* results/SubjectA
./run.py SubjectA --only data,generate
```

Advanced Configuration
----------------------

Each part of the configuration accepts a number of options that allow you to
fine tune your work. Following are some examples:

### Data

An unlabelled model that uses RGB values directly:

```yaml
SubjectA:
    width: 1024
    height: 576
    labels: 0
    data:
        - source: ~/media/SubjectA_training.mov
          options:
              - train-a
        - source: ~/media/SubjectB.mov
          options:
              - train-b
        # this video will be used as the input to the model during video generation
        - source: ~/media/SubjectA_generate.mov
          options:
              - test-a
```

Subsampling data to use only every other frame, do not label faces:

```yaml
SubjectA:
    width: 1024
    height: 576
    labels: 26
    data:
        - source: ~/media/SubjectA.mov
          options:
              - subsample 2
              - no-label-face
```

Use the first 5 minutes of the video for training, the rest for generation:

```yaml
SubjectA:
    width: 1024
    height: 576
    labels: 26
    data:
        - source: ~/media/SubjectA.mov
          options:
              - trim 0:300
              - no-label-face
        - source: ~/media/SubjectA.mov
          options:
              - trim 300:-1
              - no-label-face
              - directory-prefix test
```

Train a model on two faces, labelling each in a different label-space, and
cropping centered on the face:

```yaml
Hybrid:
    width: 1024
    height: 1024
    labels: 68
    data:
        - source: ~/media/SubjectA.mov
          # resizes before cropping
          resize: 1344x1344
          # crops to the model size of 1024x1024
          crop: true
          options:
              - label-with openpose
              - label-face
              # the labels for this subject will be 0 - 33
              - label-offset 0
              # tell cropping to center on the face
              - crop-center face
              - trim 0:360
        - source: ~/media/SubjectB.mov
          resize: 1344x1344
          crop: true
          options:
              # this should be set to the length of the previous video in
              # seconds, multiplied by its FPS (eg. 360 * 24)
              - frame-offset 8640
              - label-with openpose
              - label-face
              # the labels for this subject will be 34 - 67
              - label-offset 34
              - crop-center face
              - trim 0:360
```

### Models

You can also train a model, and then use it to generate multiple videos:


```yaml
# Train model
SubjectA:
    width: 1024
    height: 576
    labels: 26
    data:
        - source: ~/media/SubjectA.mov
          options:
            - no-label-face
    normalize: false
    train:
        global:
            epochs: 12
            options:
                - no_flip
        local:
            epochs: 40
            options:
                - no_flip

# Generate by inputting Subject B labels into the Subject A model
Transfer_SubjectB:
    width: 1024
    height: 576
    labels: 26
    data:
        - source: ~/media/SubjectA.mov
          options:
            - no-label-face
            - directory-prefix test
    normalize: false
    generate:
        model: SubjectA

# Generate by inputting Subject C labels into the Subject A model
Transfer_SubjectC:
    width: 1024
    height: 576
    labels: 26
    data:
        - source: ~/media/SubjectC.mov
          options:
            - no-label-face
            - directory-prefix test
    normalize: false
    generate:
        model: SubjectA
```

To see all the available options, you can run each of the task scripts directly:

```bash
./build_dataset.py --help
./normalize.py --help
./train.py --help
./generate_video.py --help
```


