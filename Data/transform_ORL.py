import cv2 as cv
import numpy as np
import re
import os
def do_transform(root_dir):
    # TODO: complete transfrom of ORL
    # read in image as numpy array from files
    file_list = os.listdir(root_dir)
    pattern = re.compile('s\d+')
    im_list = [ [] for i in range(10)]
    for file_name in file_list:
        if pattern.match(file_name):
            object_dir = root_dir +"\\"+file_name + "\\"
            im_name_list = os.listdir(object_dir)

            pointer = 0
            for im_name in im_name_list:
                im = cv.imdecode(np.fromfile(object_dir + im_name, dtype=np.uint8), -1)
                im_list[pointer].append(im)
                pointer += 1
    # testusage: show im in file list
    for view in range(10):
            cv.imshow("display", im_list[view][0])
            cv.waitKey(500)

    pass