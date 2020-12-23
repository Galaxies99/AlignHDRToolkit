import cv2


class Tonemapper:
    def __init__(self, tonemapper):
        self.tonemapper = tonemapper

    def process(self, hdr):
        ldr = self.tonemapper.process(hdr)
        return ldr * 255


class TonemapperReinhard(Tonemapper):
    def __init__(self, **kwargs):
        tonemapper = cv2.createTonemapReinhard(**kwargs)
        super().__init__(tonemapper)


class TonemapperDrago(Tonemapper):
    def __init__(self, **kwargs):
        tonemapper = cv2.createTonemapDrago(**kwargs)
        super().__init__(tonemapper)


class TonemapperMantiuk(Tonemapper):
    def __init(self, **kwargs):
        tonemapper = cv2.createTonemapMantiuk(**kwargs)
        super().__init__(tonemapper)
