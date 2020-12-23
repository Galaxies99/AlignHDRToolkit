import cv2


class Merger:
    def __init__(self):
        pass

    def process(self, images, exposures):
        pass


class RealMerger(Merger):
    def __init__(self, calibrater, merger):
        super().__init__()
        self.calibrater = calibrater
        self.merger = merger

    def process(self, images, exposures):
        response = self.calibrater.process(images, exposures)
        hdr = self.merger.process(images, exposures, response)
        return hdr


class FakeMerger(Merger):
    def __init__(self, merger):
        super().__init__()
        self.merger = merger

    def process(self, images, exposures=None):
        fake_hdr = self.merger(images)
        return fake_hdr


# built-in method of merger in opencv2: Debevec
class MergerDebevec(RealMerger):
    def __init__(self, **kwargs):
        calibrater = cv2.createCalibrateDebevec(**kwargs)
        merger = cv2.createMergeDebevec()
        super().__init__(calibrater, merger)


# built-in method of merger in opencv2: Robertson
class MergerRobertson(RealMerger):
    def __init__(self, **kwargs):
        calibrater = cv2.createCalibrateRobertson(**kwargs)
        merger = cv2.createMergeRobertson()
        super().__init__(calibrater, merger)


# built-in method of merger in opencv2: Mertens (fake HDR)
class MergerMertens(FakeMerger):
    def __init__(self, **kwargs):
        merger = cv2.createMergeMertens(**kwargs)
        super().__init__(merger)
