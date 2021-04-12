import numpy as np

def combine_list2mat(data_list):
    data = None
    for sub_data in data_list:
        if data is None:
            data = sub_data
        else:
            data = np.concatenate((data, sub_data), axis=0)
    return data

def do_split(MvData, labels):
    # 每个类在每个视角上有100个，50个被选作train，25个被选作validation，剩下的作为test
    l_set = set(labels[0])
    tr_MvData, tr_labels = [], []
    va_MvData, va_labels = [], []
    te_MvData, te_labels = [], []
    for v in range(len(MvData)):
        tr_data, va_data, te_data = [], [], []
        tr_label, va_label, te_label = [], [], []
        for l in l_set:
            # get the indices of a number
            l_indices = labels[v] == l
            tr_indices, va_indices, te_indices\
            = np.zeros_like(l_indices), np.zeros_like(l_indices), np.zeros_like(l_indices)
            # get the indices of train test validation
            true_indices_num = np.arange(0, 1000)
            true_indices_num = true_indices_num[l == labels[v]].tolist()
            tr_indices[true_indices_num[0:50]] = l_indices[true_indices_num[0:50]]
            va_indices[true_indices_num[50:75]] = l_indices[true_indices_num[50:75]]
            te_indices[true_indices_num[75:100]] = l_indices[true_indices_num[75:100]]

            tr_l, te_va_l = [l for i in range(50)], [l for i in range(25)]
            tr_label += tr_l
            va_label += te_va_l
            te_label += te_va_l
            # split each data and add them to list
            tr_data.append(MvData[v][tr_indices, :])
            va_data.append(MvData[v][va_indices, :])
            te_data.append(MvData[v][te_indices, :])
        # combine mat in list to one mat
        tr_data = combine_list2mat(tr_data)
        te_data = combine_list2mat(te_data)
        va_data = combine_list2mat(va_data)

        tr_MvData.append(tr_data)
        va_MvData.append(va_data)
        te_MvData.append(te_data)

        tr_labels.append(np.array(tr_label).reshape(1,-1))
        va_labels.append(np.array(va_label).reshape(1,-1))
        te_labels.append(np.array(te_label).reshape(1,-1))
        # TODO processing Labels to fit with split data

    return tr_MvData, va_MvData, te_MvData, tr_labels, va_labels, te_labels
