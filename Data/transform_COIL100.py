import os
import cv2 as cv
import numpy as np
import re
def readin_flatten(root_dir):
    path = root_dir
    im = cv.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    im = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
    im = cv.resize(im,(48,48))
    return im.reshape((1,-1))

def list2numpy(vec_list):
    vec = None
    for v in vec_list:
        if vec is None:
            vec = v
        else:
            vec = np.concatenate((vec, v), axis=0)
    return vec

def split(MvData:list,Mvlabel:list,train_rate=0.7, test_rate=0.3, va_rate=0):
    total = train_rate + test_rate +va_rate
    train_rate = train_rate/total
    test_rate = test_rate/total
    va_rate = va_rate/total
    n_sample = max(Mvlabel[0].shape)
    tr_MvData, te_MvData, va_MvData = [],[],[]
    tr_MvLabel, te_MvLabel, va_MvLabel = [],[],[]
    for v in range(len(MvData)):
        tr_end, te_end, va_end= int(train_rate*n_sample), int((train_rate+test_rate)*n_sample), int(train_rate+test_rate+va_rate)*n_sample

        tr_MvData.append(MvData[v][ : tr_end, : ])
        te_MvData.append(MvData[v][tr_end : te_end, : ])
        tr_MvLabel.append(Mvlabel[v][:,:tr_end])
        te_MvLabel.append(Mvlabel[v][:,tr_end:te_end])

        if te_end != va_end:
            va_MvData.append(MvData[v][te_end: va_end, :])
            va_MvLabel.append(Mvlabel[v][te_end:va_end])
        else:
            va_MvLabel, va_MvData = None, None
    return tr_MvData,tr_MvLabel, te_MvData,te_MvLabel, va_MvData, va_MvLabel


def do_transform(root_dir, loadin_mode = "native"):
    # read in image in numpy array
    # consider file name as label
    file_list = os.listdir(root_dir)

    is_image = re.compile('(.*)\.png')
    pop_indices = []
    for f_name in file_list:
        if re.match(is_image,f_name) is None:
            idx = file_list.index(f_name)
            pop_indices.append(idx-len(pop_indices))
    for idx in pop_indices:
        file_list.pop(idx)

    obj_pattern = re.compile("obj(\d)+__(\d)*")
    class_pattern = re.compile("obj(\d)+")
    angle_pattern = re.compile("(\d)+\.png")
    view_range = [(0,15),(40,55),(105,120),(160,175)]
    Mv_Data = [[],[],[],[]]
    Mv_label = [[],[],[],[]]
    for f_name in file_list:
        label = []
        # extract info of image
        l = re.search(class_pattern, f_name)
        angle = re.search(angle_pattern, f_name)
        l = f_name[l.span()[0]: l.span()[1]]
        l = l[3:]
        l = int(l)
        angle = f_name[angle.span()[0] : angle.span()[1]]
        angle = angle[:-4]
        angle = int(angle)
        view = None
        for v in range(len(view_range)):
            angle_range = view_range[v]
            if angle >= angle_range[0] and angle <= angle_range[1]:
                view = v
                break
        # read in image and flatten
        if view is not None:
            vec = readin_flatten(root_dir + "\\" + f_name)
            Mv_Data[view].append(vec)
            Mv_label[view].append(l)
    # 整理数据为numpy矩阵
    for v in range(len(Mv_Data)):
        Mv_Data[v] = list2numpy(Mv_Data[v])

        Mv_label[v].sort()
        Mv_label[v] = np.array(Mv_label[v]).reshape(1,-1)
    return Mv_Data, Mv_label


if __name__ == "__main__":
    Mv_Data, labels = do_transform("E:\\Works\\数据集\\COIL\\coil-100")
    tr_MvData,tr_MvLabel, te_MvData,te_MvLabel, va_MvData, va_MvLabel = split(Mv_Data, labels)
