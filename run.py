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
args = parser.parse_args()

with open(args.config) as config_file:
    config = yaml.load(config_file.read(), Loader=Loader)

config = config[args.name]
if config is None:
    sys.exit("{} not found in {}".format(args.name, args.config))

name = args.name
width = config["width"]
height = config["height"]
labels = config.get('labels', 34)

commands = []

def build_options(yaml):
    yaml = [] if yaml is None else yaml
    return ["--{}".format(o) for o in yaml]
#
# Data
#
for recipe in config["data"]:
    source = recipe["source"]
    data = recipe.get('name', name)
    options = build_options(recipe.get('options'))
    if labels == 0: options.append("--no-label")
    options = " ".join(options)
    commands.append("./build_dataset.py --dataroot data/{} -i {} --resize {}x{} {}".format(data, source, width, height, options))


#
# Training
#

# global
train = config["train"]["global"]
data = train.get('data', name)
niter = ceil(train["epochs"] / 2)
niter_decay = train["epochs"] - niter
options = build_options(train.get('options'))
if os.path.exists("checkpoints/{}_global".format(name)): options.append('--continue_train')
options = " ".join(options)

commands.append("./train.py --dataroot data/{} --name {}_global "
        "--num_D 3 --label_nc {} --no_instance --fp16 --netG global --fineSize {:.0f} "
        "--ngf 64 --niter {} --niter_decay {} "
        "--resize_or_crop scale_width_and_crop {}".format(
            data, name, labels, width / 2, niter, niter_decay, options))


# local
train = config["train"]["local"]
data = train.get('data', name)
niter = ceil(train["epochs"] / 2)
niter_decay = train["epochs"] - niter
niter_fix_global = ceil(train["epochs"] / 5)
options = build_options(train.get('options'))
if os.path.exists("checkpoints/{}_local".format(name)): options.append('--continue_train')
options = " ".join(options)

commands.append("./train.py --dataroot data/{} --name {}_local "
        "--num_D 3 --label_nc {} --no_instance --fp16 --netG local --fineSize {} "
        "--ngf 32 --niter {} --niter_decay {} --niter_fix_global {} "
        "--resize_or_crop none --load_pretrain checkpoints/{}_global {}".format(
            data, name, labels, width, niter, niter_decay, niter_fix_global, name, options))

#
# Generation
#

generate = config['generate']
data = generate.get('data', name)
options = build_options(generate.get('options'))
options = " ".join(options)
commands.append("./generate_video.py --dataroot data/{} --name {}_local "
        "--label_nc {} --no_instance --fp16 --netG local --fineSize {} "
        "--ngf 32 --resize_or_crop none {}".format(
            data, name, labels, width, options))

command = " && \\\n".join(commands)

#
# Go
#
print(command)

if not args.dry_run:
    ret = os.system(command)
    if ret != 0:
        sys.exit("Run failed")
