#!/usr/bin/env python
import argparse
import sys
import os
from math import ceil
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

parser = argparse.ArgumentParser()
parser.add_argument("name", help="the name of the configuration to load from the file provided")
parser.add_argument("--config", default="config.yml", help="path to the config YAML file")
parser.add_argument("--dry-run", help="print the command to be executed and exit", action='store_true')
parser.add_argument("--only", help="Comma separated list of commands to run from data, normalize, train_global, train_local, synthesize, and generate. Defaults to all.", default="data,normalize,train_global,train_local,synthesize,generate")
args = parser.parse_args()

only = set(tuple(args.only.split(',')))

with open(args.config) as config_file:
    config = yaml.load(config_file.read(), Loader=Loader)

config = config[args.name]
if config is None:
    sys.exit("{} not found in {}".format(args.name, args.config))

name = args.name
width = config["width"]
height = config["height"]
labels = config.get('labels', 34)
normalize = config.get('normalize', False)
global_ngf = 64
local_ngf = 32

commands = []

def build_options(yaml):
    yaml = [] if yaml is None else yaml
    return ["--{}".format(o) for o in yaml]
#
# Data
#
if 'data' in only and "data" in config:
    for recipe in config["data"]:
        source = recipe["source"]
        data = recipe.get('name', name)
        resize = recipe.get('resize', True)
        if resize is True: # ie., explicitly true, or unspecified
            resize = "{}x{}".format(width, height)
        crop = recipe.get('crop', False)
        if crop is True:
            crop = "{}x{}".format(width, height)

        options = build_options(recipe.get('options'))
        if labels == 0: options.append("--no-label")
        if normalize: options.append('--normalize')
        if resize: options.append('--resize {}'.format(resize))
        if crop: options.append('--crop {}'.format(crop))
        options = " ".join(options)
        commands.append("./build_dataset.py --dataroot data/{} -i {} {}".format(data, source, options))

#
# Normalization
#

if 'normalize' in only and normalize:
    try:
        options = normalize.get('options')
    except (AttributeError, TypeError):
        options = None

    options = build_options(options)
    target = normalize.get('target', None)
    if target: options.append('--target-dataroot data/{}'.format(target))
    options = " ".join(options)
    commands.append("./normalize.py --dataroot data/{} {}".format(name, options))

#
# Training
#
if 'train_global' in only and "train" in config and "global" in config["train"]:

    # global
    train = config["train"]["global"]
    data = train.get('data', name)
    niter = ceil(train["epochs"] / 2)
    niter_decay = train["epochs"] - niter
    global_ngf = train.get('ngf', global_ngf)
    options = build_options(train.get('options'))
    if os.path.exists("checkpoints/{}_global/latest_net_G.pth".format(name)): options.append('--continue_train')
    options.append('--ngf {}'.format(global_ngf))


    options = " ".join(options)

    commands.append("./train.py --dataroot data/{} --name {}_global "
            "--model pix2pixHDts --num_D 3  --label_nc {} --no_instance --fp16 --netG global --loadSize {:.0f} --fineSize {:.0f} "
            "--niter {} --niter_decay {} "
            "--resize_or_crop scale_width {}".format(
                data, name, labels, width / 2, width / 2, niter, niter_decay, options))


if 'train_local' in only and "train" in config and "local" in config["train"]:
    # local
    train = config["train"]["local"]
    data = train.get('data', name)
    niter = ceil(train["epochs"] / 2)
    niter_decay = train["epochs"] - niter
    niter_fix_global = ceil(train["epochs"] / 5)
    ngf = train.get('ngf', local_ngf)
    options = build_options(train.get('options'))
    if os.path.exists("checkpoints/{}_local/latest_net_G.pth".format(name)): options.append('--continue_train')
    options.append('--ngf {}'.format(ngf))


    options = " ".join(options)

    commands.append("./train.py --dataroot data/{} --name {}_local "
            "--model pix2pixHDts --num_D 3 --label_nc {} --no_instance --fp16 --netG local --loadSize {} --fineSize {} "
            "--niter {} --niter_decay {} --niter_fix_global {} "
            "--resize_or_crop none --load_pretrain checkpoints/{}_global {}".format(
                data, name, labels, width, width, niter, niter_decay, niter_fix_global, name, options))

#
# Synthesis
#
if 'synthesize' in only and 'synthesize' in config:
    synthesize = config['synthesize']
    script = synthesize['script']
    data = synthesize.get('data', name)
    options = build_options(synthesize.get('options'))
    options = " ".join(options)
    commands.append("{} --dataroot data/{} {}".format(script, data, options))

#
# Generation
#

if 'generate' in only and 'generate' in config:
    generate = config['generate']
    ngf = generate.get('ngf', local_ngf)
    try:
        options = generate.get('options')
        data = generate.get('data', name)
        model = generate.get('model', name)
    except (AttributeError, TypeError):
        options = None
        data = name
        model = name
    options = build_options(options)
    options = " ".join(options)
    commands.append("./generate_video.py --dataroot data/{} --name {}_local --results_name {} "
            "--model pix2pixHDts --label_nc {} --no_instance --fp16 --netG local --fineSize {} "
            "--ngf {} --resize_or_crop none {}".format(
                data, model, name, labels, width, ngf, options))

command = " && \\\n".join(commands)

#
# Go
#
print(command)

if not args.dry_run:
    ret = os.system(command)
    if ret != 0:
        sys.exit("Run failed")
