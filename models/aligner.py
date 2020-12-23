import cv2
import numpy as np


class Aligner:
    def __init__(self):
        pass

    def process(self, images, ref_id=None):
        return images


# built-in method of alignment method in cv2
class AlignMTB(Aligner):
    def __init__(self, **kwargs):
        super().__init__()
        self.align_mtb = cv2.createAlignMTB(**kwargs)

    def process(self, images, ref_id=None):
        self.align_mtb.process(images, images)
        return images


# Reference:
#   Greg Ward,
#     Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Hand-Held Exposures,
#       JGT 2003.
class AlignMTBPyramid(Aligner):
    def __init__(self, grey_approx='G', threshold_range=10):
        super().__init__()
        self.grey_approx = grey_approx
        self.threshold_range = threshold_range

    def process(self, images, ref_id=None):
        # Construct grey-level images
        image_stack = []
        for image in images:
            if self.grey_approx == 'G':
                image_stack.append(image[:, :, 1])
            elif self.grey_approx == 'RGB':
                # Here image is in BGR mode
                image_stack.append((19 * np.array(image[:, :, 0]).astype(np.int16) +
                                    183 * np.array(image[:, :, 1]).astype(np.int16) +
                                    54 * np.array(image[:, :, 2]).astype(np.int16)) / 256.0)
            else:
                print('Error: Undefined Grey-level Approximation Method.')
                return None

        # Construct image pyramids
        pyramid_level = max(0, int(np.log2(min(np.array(image_stack).shape[1], np.array(image_stack).shape[2]) / 64)))
        image_pyramids = []
        for image in image_stack:
            pyramid = [image]
            cur_image = image
            for i in range(pyramid_level):
                cur_image = cv2.pyrDown(cur_image)
                pyramid.insert(0, cur_image)
            image_pyramids.append(pyramid)

        # Calculate reference image MTB
        if ref_id is None:
            ref_id = int((len(image_stack) - 1) / 2)
        ref_imgs_MTB = []
        for img in image_pyramids[ref_id]:
            med = np.median(img)
            _, ref_img_MTB = cv2.threshold(img, med, 255, cv2.THRESH_BINARY)
            ref_imgs_MTB.append(ref_img_MTB)

        # Calculate offset
        image_offset = []
        for i, pyramid in enumerate(image_pyramids):
            offset = [0, 0]
            for j, img in enumerate(pyramid):
                med = np.median(img)
                # Median Threshold Bitmap
                _, img_MTB = cv2.threshold(img, med, 255, cv2.THRESH_BINARY)
                # Exclusion Bitmap
                img_EB = cv2.inRange(img, med - self.threshold_range, med + self.threshold_range)
                # Reference Median Threshold Bitmap
                ref_img_MTB = ref_imgs_MTB[j]
                cur_offset = [0, 0]
                cur_diff = float('Inf')
                for offset_x in range(-1, 1):
                    for offset_y in range(-1, 1):
                        c_offset_x = offset_x + offset[0]
                        c_offset_y = offset_y + offset[1]
                        img_trans = cv2.warpAffine(img_MTB,
                                                   np.array([[1, 0, c_offset_x],
                                                             [0, 1, c_offset_y]]).astype(np.float32),
                                                   (img_MTB.shape[1], img_MTB.shape[0]))
                        diff = np.sum(np.logical_and(np.logical_xor(ref_img_MTB, img_trans), img_EB))
                        if diff < cur_diff:
                            cur_diff = diff
                            cur_offset = [c_offset_x, c_offset_y]
                offset = cur_offset
            image_offset.append(offset)

        res = []
        for i, img in enumerate(images):
            offset = image_offset[i]
            img_trans = cv2.warpAffine(img,
                                       np.array([[1, 0, offset[0]],
                                                 [0, 1, offset[1]]]).astype(np.float32),
                                       (img.shape[1], img.shape[0]))
            res.append(img_trans)
        return res


# Align the images according to the features of the images.
class AlignFeature(Aligner):
    def __init__(self, detector, min_matches=9, match_percent=0.15):
        super().__init__()
        self.detector = detector
        self.min_matches = min_matches
        self.match_percent = match_percent

    def process(self, images, ref_id=None):
        if ref_id is None:
            ref_id = int((len(images) - 1) / 2)
        res_images = []
        for i, img in enumerate(images):
            if i == ref_id:
                res_images.append(img)
                continue
            img = self.alignFeaturesPair(img, images[ref_id])
            res_images.append(img)
        return res_images

    def alignFeaturesPair(self, img, ref_img):
        # Convert to gray-level images
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        keypoints, descriptor = self.detector.detectAndCompute(img_gray, None)
        ref_keypoints, ref_descriptor = self.detector.detectAndCompute(ref_img_gray, None)
        descriptor = descriptor.astype(np.uint8)
        ref_descriptor = ref_descriptor.astype(np.uint8)

        # Matcher
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        # Match points
        matches = matcher.match(descriptor, ref_descriptor, None)
        matches.sort(key=lambda x: x.distance, reverse=False)
        if len(matches) < self.min_matches:
            return img
        match_number = max(int(len(matches) * self.match_percent), self.min_matches)
        matches = matches[:match_number]

        # Find the points in keypoints
        points = np.zeros((match_number, 2), dtype=np.float32)
        ref_points = np.zeros((match_number, 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points[i, :] = keypoints[match.queryIdx].pt
            ref_points[i, :] = ref_keypoints[match.trainIdx].pt

        # Find the homography matrix
        h, _ = cv2.findHomography(points, ref_points, cv2.RANSAC)
        ret_img = cv2.warpPerspective(img, h, (ref_img.shape[1], ref_img.shape[0]))

        return ret_img


# Reference:
#   Rublee, Ethan, et al,
#     ORB: An efficient alternative to SIFT or SURF.
#       ICCV 2011
class AlignFeatureORB(AlignFeature):
    def __init__(self, min_matches=9, match_percent=0.15, **kwargs):
        self.detector = cv2.ORB_create(**kwargs)
        super().__init__(self.detector, min_matches, match_percent)


# Reference:
#   Lowe, David G.，
#     Distinctive image features from scale-invariant keypoints，
#       IJCV 2004
class AlignFeatureSIFT(AlignFeature):
    def __init__(self, min_matches=9, match_percent=0.15, **kwargs):
        self.detector = cv2.xfeatures2d.SIFT_create(**kwargs)
        super().__init__(self.detector, min_matches, match_percent)


# Reference:
#   GD Evangelidis, et al,
#     Parametric image alignment using enhanced correlation coefficient maximization,
#       PAMI 2008
class AlignECC(Aligner):
    def __init__(self, warp_mode=cv2.MOTION_TRANSLATION, iteration=5000, terminal_eps=1e-10, gaussFiltSize=5):
        super().__init__()
        self.warp_mode = warp_mode
        assert warp_mode == cv2.MOTION_AFFINE or warp_mode == cv2.MOTION_TRANSLATION or \
            warp_mode == cv2.MOTION_EUCLIDEAN or warp_mode == cv2.MOTION_HOMOGRAPHY
        self.iteration = iteration
        self.terminal_eps = terminal_eps
        self.gaussFiltSize = gaussFiltSize

    def process(self, images, ref_id=None, mask=None):
        if ref_id is None:
            ref_id = int((len(images) - 1) / 2)
        res_images = []
        for i, img in enumerate(images):
            if i == ref_id:
                res_images.append(img)
                continue
            img = self.alignECCPair(img, images[ref_id], mask)
            res_images.append(img)
        return res_images

    def alignECCPair(self, img, ref_img, mask=None):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        ref_img_shape = ref_img.shape

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iteration, self.terminal_eps)

        (cc, warp_matrix) = cv2.findTransformECC(ref_img_gray, img_gray, warp_matrix, self.warp_mode, criteria,
                                                 mask, self.gaussFiltSize)

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            img_aligned = cv2.warpPerspective(img, warp_matrix, (ref_img_shape[1], ref_img_shape[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            img_aligned = cv2.warpAffine(img, warp_matrix, (ref_img_shape[1], ref_img_shape[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return img_aligned

