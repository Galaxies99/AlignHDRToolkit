import os
import numpy as np
import cv2
import rawpy
import exifread


def readImages(file_directory):
    files = []

    directory = 'data/' + file_directory + '/'
    file_list = getAllFile(directory)
    exposures_filename = directory + 'exposure.info'

    if exposures_filename in file_list:
        exposures_list = []
        with open(exposures_filename) as exposure_file:
            for line in exposure_file:
                pos = line.find(' ')
                if pos == -1:
                    continue
                files.append('data/' + file_directory + '/' + line[:pos])
                line = line[pos + 1:]
                line = line[: line.find(' ')]
                exposures_list.append(1.0 / float(line))

        exposures = np.array(exposures_list, dtype=np.float32)

        images = []
        for img_file in files:
            if 'dng' in img_file:
                images.append(readRawImage(img_file))
            else:
                images.append(cv2.imread(img_file))
    else:
        images = []
        exposures = []
        for img_file in file_list:
            if img_file == 'reference.info':
                continue
            if 'dng' in img_file:
                images.append(readRawImage(img_file))
            else:
                images.append(cv2.imread(img_file))
            contents = exifread.process_file(open(img_file, 'rb'), details=False, strict=True)
            if 'EXIF ExposureTime' in contents.keys():
                exposure_str = contents['EXIF ExposureTime'].printable
            elif 'Image ExposureTime' in contents.keys():
                exposure_str = contents['Image ExposureTime'].printable
            else:
                print('Error: No exposure informations.')
                return None, None, None
            if '/' in exposure_str:
                x, y = exposure_str.split('/')
                exposure = float(x) / float(y)
            else:
                exposure = float(exposure_str)
            exposures.append(exposure)

    images, exposures = sortByExposure(images, exposures)
    exposures = np.array(exposures, dtype=np.float32)

    ref_id = None
    reference_filename = directory + 'reference.info'
    if reference_filename in file_list:
        with open(reference_filename) as reference_file:
            ref_filename = reference_file.read()
        ref_filename = ref_filename.rstrip('\n').rstrip(' ').lstrip(' ')
        ref_filename = directory + ref_filename
        if ref_filename in file_list:
            ref_img = cv2.imread(ref_filename)
            for i, img in enumerate(images):
                if (img == ref_img).all():
                    ref_id = i

    return images, exposures, ref_id


def getAllFile(directory):
    res = []
    for file in os.listdir(directory):
        if file[0] == '.':
            continue
        t_file = directory + file
        if os.path.isdir(t_file):
            continue
        res.append(t_file)
    return res


def readRawImage(file):
    with rawpy.imread(file) as raw:
        rgb = raw.postprocess()
    # Changing to BGR
    rgb[:, :, 0], rgb[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
    return rgb


def getResultFolder(file_directory):
    res_path = os.getcwd().strip()
    res_path = res_path.rstrip('/')
    res_path = res_path + '/data/' + file_directory + '/result'

    is_exists = os.path.exists(res_path)

    if not is_exists:
        os.mkdir(res_path)

    return res_path


def sortByExposure(images, exposures):
    zipped = zip(images, exposures)
    sort_zipped = sorted(zipped, key=lambda x: x[1])
    result = zip(*sort_zipped)
    images, exposures = [list(x) for x in result]
    return images, exposures
