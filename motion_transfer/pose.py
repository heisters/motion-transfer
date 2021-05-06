class Pose(object):
    # from https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    indices = {
        "nose"       : 0,  
        "neck"       : 1,  
        "rshoulder"  : 2,  
        "relbow"     : 3,  
        "rwrist"     : 4,  
        "lshoulder"  : 5,  
        "lelbow"     : 6,  
        "lwrist"     : 7,  
        "midhip"     : 8,  
        "rhip"       : 9,  
        "rknee"      : 10, 
        "rankle"     : 11, 
        "lhip"       : 12, 
        "lknee"      : 13, 
        "lankle"     : 14, 
        "reye"       : 15, 
        "leye"       : 16, 
        "rear"       : 17, 
        "lear"       : 18, 
        "lbigtoe"    : 19, 
        "lsmalltoe"  : 20, 
        "lheel"      : 21, 
        "rbigtoe"    : 22, 
        "rsmalltoe"  : 23, 
        "rheel"      : 24, 
        "background" : 25
    }

    def __init__(self, points):
        self.points = points

    def __getattr__(self, attr):
        if attr in self.__class__.indices:
            return self.points[self.__class__.indices[attr]]
        elif attr.endswith("_idx") and attr[:-4] in self.__class__.indices:
            return self.__class__.indices[attr[:-4]]
        else:
            raise AttributeError("%r object has no attribute %r" % self.__class__.__name__, attr)

    @property
    def larm(self):
        return self.points[[self.lshoulder_idx, self.lelbow_idx, self.lwrist_idx]]

    @larm.setter
    def larm(self, values):
        self.points[[self.lshoulder_idx, self.lelbow_idx, self.lwrist_idx]] = values
