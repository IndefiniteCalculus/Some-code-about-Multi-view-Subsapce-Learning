import numpy as np
import cv2 as cv
import os
root_dir1 = r"E:/Works/数据集/CMU_PIE_for_MvCCDA/view_files"
root_dir2 = r"E:/Works/数据集/CMU_PIE_for_MvCCDA/sample_files"
def save_view_file(tr_MvList, tr_labels, te_MvList, te_labels):
    num_view = len(tr_MvList)
    for v in range(len(tr_labels)):
        tr_label = tr_labels[v]
        tr_label = np.ravel(tr_label)
        for l_idx in range(len(tr_label)):
            if os.path.exists(root_dir1 + "\\" + str(tr_label[l_idx])) is False:
                os.mkdir(root_dir1 + "\\" + str(tr_label[l_idx]))
            im = tr_MvList[v][l_idx]
            path = root_dir1 + "/" + str(tr_label[l_idx]) + "/" + str(v) + ".png"
            cv.imencode('.png', im)[1].tofile(path)
        te_label = te_labels[v]
        te_label = np.ravel(te_label)
        for l_idx in range(len(te_label)):
            im = te_MvList[v][l_idx]
            path = root_dir1 + "/" + str(te_label[l_idx])+ "/" + str(v+num_view)+ ".png"
            cv.imencode('.png', im)[1].tofile(path)

    pass

def save_sample_file(tr_MvList, tr_labels, te_MvList, te_labels):
    num_view = len(tr_MvList)
    for v in range(len(tr_labels)):
        tr_label = tr_labels[v]
        tr_label = np.ravel(tr_label)
        for l_idx in range(len(tr_label)):
            if os.path.exists(root_dir2 + "\\" + str(v)) is False:
                os.mkdir(root_dir2 + "\\" + str(v))
            im = tr_MvList[v][l_idx]
            path = root_dir2  + "/" + str(v)+ "/" + str(tr_label[l_idx]) + ".png"
            cv.imencode('.png', im)[1].tofile(path)
        te_label = te_labels[v]
        te_label = np.ravel(te_label)
        for l_idx in range(len(te_label)):
            im = te_MvList[v][l_idx]
            path = root_dir2 + "/" + str(v) + "/" + str(te_label[l_idx]) + ".png"
            cv.imencode('.png', im)[1].tofile(path)

def reconstruct(MvData,labels):
    # reconstruct data vector to data mat
    MvImage = []
    num_sample = MvData[0].shape[0]
    for v in range(len(MvData)):
        data = MvData[v]
        images_in_view = []
        for idx in range(num_sample):
            im = data[idx,:]
            w_h = int(np.sqrt(max(im.shape)))
            im = im.reshape((w_h, w_h))
            im = cv.rotate(im,cv.ROTATE_90_CLOCKWISE)
            im = cv.flip(im, 1)
            cv.imshow("display"+str(v)+" "+str(labels[v][0,idx]), im)
            cv.waitKey(800)
            images_in_view.append(im)
        MvImage.append(images_in_view)
    return MvImage
    pass

if __name__ == '__main__':
    from Data import dataloader
    tr_MvData, te_MvData, labels = dataloader.load_data("matlabrow")
    tr_labels = labels.get('train')
    te_labels = labels.get('test')
    print("reconstructing data")
    tr_Mvlist = reconstruct(tr_MvData,tr_labels)
    te_Mvlist = reconstruct(te_MvData,te_labels)
    print("saving data")
    save_view_file(tr_Mvlist, tr_labels, te_Mvlist, te_labels)
    # save_sample_file(tr_Mvlist, tr_labels, te_Mvlist, te_labels)
    print("saving complete")
pass