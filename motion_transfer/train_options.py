### Some sections:
### Copyright (c) 2019 Caroline Chan
### All rights reserved. 
import argparse
import sys
from .options import add_motion_transfer_options
sys.path.append('./vendor/pix2pixHD/')
from options.train_options import TrainOptions as P2PTrainOptions

class TrainOptions(P2PTrainOptions):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    def initialize(self):
        P2PTrainOptions.initialize(self)

        add_motion_transfer_options(self.parser)

        #for L1 loss
        self.parser.add_argument('--use_l1', action='store_true', help='use L1 loss instead of VGG')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_F', type=float, default=1.0, help='weight flow loss')

        # for face training
        self.parser.add_argument('--face', action='store_true', help='train face model')
