import argparse
import sys
sys.path.append('./vendor/pix2pixHD/')
from options.base_options import BaseOptions as P2PBaseOptions

class BaseOptions(P2PBaseOptions):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(fromfile_prefix_chars='@')



