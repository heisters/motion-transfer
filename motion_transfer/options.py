### Some sections:
### Copyright (c) 2019 Caroline Chan
### All rights reserved. 
import argparse

def add_motion_transfer_options(parser):
    # for face discriminator
    parser.add_argument('--face_discrim', action='store_true', help='if specified, add a face discriminator')
    parser.add_argument('--niter_fix_main', type=int, default=0, help='number of epochs that we only train the face discriminator')
    
    #for face generator
    parser.add_argument('--face_generator', action='store_true', help='if specified, add a face residual prediction generator')
    parser.add_argument('--faceGtype', type=str, default='unet', help='selects architecture to use for face generator, choose from a UNet generator or global generator [unet|global]')

