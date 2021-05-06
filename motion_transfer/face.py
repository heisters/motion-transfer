from imutils import face_utils

class Face(object):
    def __init__(self, landmarks, shape):
        super().__setattr__('landmarks', landmarks)
        super().__setattr__('shape', shape)

    def __getattr__(self, attr):
        if attr in self.landmarks:
            (j,k) = self.landmarks[attr]
            return self.shape[j:k]
        elif attr.endswith("_idxs") and attr[:-5] in self.landmarks:
            return self.landmarks[attr[:-5]]
        else:
            raise AttributeError("%r object has no attribute %r" % self.__class__.__name__, attr)

    def __setattr__(self, attr, value):
        if attr in self.landmarks:
            (j,k) = self.landmarks[attr]
            self.shape[j:k] = value
        else:
            super().__setattr__(attr, value)

    def __getitem__(self, key):
        if key in self.landmarks:
            (j,k) = self.landmarks[key]
            return self.shape[j:k]
        else:
            return None

    def __setitem__(self, key, value):
        if key in self.landmarks:
            (j,k) = self.landmarks[key]
            self.shape[j:k] = value
        else:
            raise AttributeError("%r object has no landmark %r" % self.__class__.__name__, attr)

    def shapes(self):
        for name in self.landmarks.keys():
            (j,k) = self.landmarks[name]
            yield (name, self.shape[j:k])
        
