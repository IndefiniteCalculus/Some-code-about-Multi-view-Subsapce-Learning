import cv2 as cv
import numpy as np
import re
import os
def do_transform(root_dir):
    # read in image as numpy array from files, extract label from file_name
    file_list = os.listdir(root_dir)
    pattern = re.compile('s\d+')
    im_list = [ [] for i in range(10)]
    labels = [ [] for i in range(10)]
    for file_name in file_list:
        if pattern.match(file_name):
            # read in all the image under this dir
            object_dir = root_dir +"\\"+file_name + "\\"
            im_name_list = os.listdir(object_dir)
            pointer = 0
            for im_name in im_name_list:
                im = cv.imdecode(np.fromfile(object_dir + im_name, dtype=np.uint8), -1)
                im_list[pointer].append(im)
                pointer += 1
            # and extract file name as label by the way
            for view_labels in labels:
                view_labels.append(int(file_name[1:]))

    # transform label lists to numpy arrays
    for idx in range(len(labels)):
        labels[idx] = np.array(labels[idx])

    # flatten matrices into vectors
    n_sample = len(im_list[0])
    data = []
    for view in range(len(im_list)):
        data_mat = np.zeros((n_sample, 92*112))
        im_idx = 0
        for im in im_list[view]:
            vector = im.reshape(1,-1)
            data_mat[im_idx][:] = vector
            im_idx += 1
        data.append(data_mat)

    return data, labels
