import argparse
import sys
sys.path.append('./vendor/pix2pixHD/')
from options.train_options import TrainOptions as P2PTrainOptions

class TrainOptions(P2PTrainOptions):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(fromfile_prefix_chars='@')



