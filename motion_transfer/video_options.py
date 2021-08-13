from .base_options import BaseOptions
from .options import add_motion_transfer_options

# edited version of TestOptions
class VideoOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        #self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        #self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--results_dir', type=str, default='results', help='saves results here.')
        self.parser.add_argument('--results_name', type=str, help='name for the subdirectory under results_dir to save the results. Defaults to --name.')
        #self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        #self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
        self.parser.add_argument('--how_many', type=int, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.parser.add_argument('--fps', type=float, default=24., help='frame per second for video generation')
        self.parser.add_argument('--codec', choices=['x264', 'prores'], default='x264')
        self.parser.add_argument('--output-suffix', type=str, help='Append the provided suffix to the output directory and file', default='')
        self.isTrain = False
        self.parser.add_argument('--face', action='store_true', help='use face model for inferences')
        add_motion_transfer_options(self.parser)

