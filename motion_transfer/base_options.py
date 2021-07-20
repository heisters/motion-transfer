import sys
import argparse
sys.path.append('./vendor/pix2pixHD/')
from .options import add_motion_transfer_options
from options.base_options import BaseOptions as P2PBaseOptions

class BaseOptions(P2PBaseOptions):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(fromfile_prefix_chars='@')



