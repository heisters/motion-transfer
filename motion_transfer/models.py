### Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
### BSD License. All rights reserved. 
import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from models.pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHDts':
        from .pix2pixHDts_model import Pix2PixHDtsModel
        model = Pix2PixHDtsModel()
    else:
        from .ui_model import UIModel
        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
