#!/usr/bin/env python

import argparse
from math import floor

parser = argparse.ArgumentParser(description='Resize two given dimensions to the nearest size that is divisible by N without changing aspect ratio.')
parser.add_argument('width', type=int, help='original width')
parser.add_argument('height', type=int, help='original height')
parser.add_argument('divisor', type=int, help='divisor')

args = parser.parse_args()


ow = args.width
oh = args.height
D = args.divisor

w = floor(ow / D) * D
found = False

while w > D:
    h = w / ow * oh
    ok = h % D == 0
    if ok:
        print("%d x %d" % (w, h))
        found = True
    w -= D


if not found:
    print("No solution found.")

