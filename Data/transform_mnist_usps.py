import cv2 as cv
import numpy as np
import re
import os


def readin_flatten(root_dir, sub_dir, loadin_mode):
    path = root_dir + "\\" + sub_dir + "\\"
    labels = []
    l = int(sub_dir)
    file_list = os.listdir(path)
    im = cv.imdecode(np.fromfile(path + file_list[0], dtype=np.uint8), -1)
    data_mat = np.zeros((100, im.shape[0] * im.shape[1]))

    for i in range(100):
        im = cv.imdecode(np.fromfile(path + file_list[i], dtype=np.uint8), -1)
        if loadin_mode == "resize":
            im = cv.resize(im, (16,16))
        vec = im.reshape(1, -1)
        labels.append(l)
        data_mat[i, :] = vec

    return data_mat, labels


def do_transform(root_dir, loadin_mode = "native"):
    # read in image in numpy array
    # consider file name as label
    file_list = os.listdir(root_dir)
    Mv_Data = []
    labels = []

    pattern = re.compile('\d')
    for f_name in file_list:
        data_list = []
        data = None
        label = []
        view_dir = root_dir + "\\" + f_name
        digit_list = os.listdir(view_dir)
        for d_name in digit_list:
            if pattern.match(d_name):
                data_mat, temp_l = readin_flatten(view_dir, d_name, loadin_mode=loadin_mode)
                data_list.append(data_mat)
                label = label + temp_l
        # combine datalist to numpy matrix
        for data_mat in data_list:
            if data is None:
                data = data_mat
            else:
                data = np.concatenate((data, data_mat),axis=0)

        Mv_Data.append(data)
        label = np.array(label)
        labels.append(label)
    return Mv_Data, labels


if __name__ == "__main__":
    Mv_Data, labels = do_transform("E:\\Works\\数据集\\mnist-usps")
