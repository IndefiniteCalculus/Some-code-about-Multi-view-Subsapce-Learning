import numpy as np
# split samples into 10 separated parts

def k_fold(k, MvData, sub_MvData = None):
    num_sample = MvData[0]
    pass

def get_merge_part_idx():

    pass

def get_split_range(num_sample):
    each_n_sample = num_sample // 10
    unfull_num = num_sample % 10

    # generate indices of samples
    slice_indices = []
    for i in range(10):
        slice_indices.append([i * each_n_sample, (i + 1) * each_n_sample])

    # balance the proportion of split sample sets
    if unfull_num > 5:
        used_unfull_num = 0
        for indices in slice_indices:
            if used_unfull_num < unfull_num:
                if indices[0] != 0:
                    indices[0] = indices[0] + 1 + used_unfull_num
                indices[1] = indices[1] + 1 + used_unfull_num
                used_unfull_num += 1
    else:
        slice_indices[-1][1] += unfull_num
    pass

if __name__ == "__main__":
    MvData = [
        np.outer(np.arange(0,100), np.ones(100)),
        np.outer(np.arange(100,200), np.ones(100)),
        np.outer(np.arange(300,400), np.ones(100))
    ]
    pass