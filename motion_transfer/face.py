from imutils import face_utils
import numpy as np

class Face(object):
    def __init__(self, landmarks, shape, rect, confidence):
        super().__setattr__('landmarks', landmarks)
        super().__setattr__('shape', shape)
        super().__setattr__('rect', rect)
        super().__setattr__('confidence', confidence)

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

    def center(self):
        return np.mean(self.nose, axis=0)


    def padded_bounds(self, osize, *, w, h):
        c = self.center()
        minx, miny, maxx, maxy = self.rect.left(), self.rect.top(), self.rect.right(), self.rect.bottom()
        rectw = maxx - minx
        recth = maxy - miny
        dw = osize - rectw
        dh = osize - recth

        if dw < 0 or dh < 0:
            raise RuntimeError("face rectangle {}x{} is larger than configured face size".format(rectw, recth))

        maxx = min(maxx + dw // 2, w)
        minx = max(maxx - osize, 0)
        maxx += osize - (maxx - minx)

        maxy = min(maxy + dh // 2, h)
        miny = max(maxy - osize, 0)
        maxy += osize - (maxx - minx)

        return minx, miny, maxx, maxy

