import cv2
import HDR
import utils
from tqdm import tqdm
from models import aligner


if __name__ == '__main__':
    data_list = ['Big_City_Lights', 'Hall', 'High_Five', 'Izmir_Harbor', 'The_Marble_Hall']
    for file_directory in tqdm(data_list):
        images, exposures, ref_id = utils.readImages(file_directory)
        hdr = HDR.hdrSynthesis(images, exposures, ref_id)
        cv2.imwrite(utils.getResultFolder(file_directory) + '/res.hdr', hdr)
        ldr = HDR.hdrTonemapping(hdr)
        cv2.imwrite(utils.getResultFolder(file_directory) + '/ldr-tonemapping.jpg', ldr)
