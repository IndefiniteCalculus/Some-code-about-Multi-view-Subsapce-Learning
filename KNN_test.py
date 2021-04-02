import numpy as np

def self_test(map_matrixes, data, labels, data_param):
    # the way of testing is find each mapped data vectors nearest vector (use Eucilid distance)
    # if the nearest vector and selected vector shared with same label, we think the map of selected vector is a success
    # if not, we think the mapping of this vector is a failure
    mapped_data = []
    for vi in range(data_param.get("num_view")):
        mapped_vi = np.dot(data[vi], map_matrixes[vi].T)
        mapped_data.append(mapped_vi)

    adjacency_list = [] # each element idx is the (i * num_view + j) -th view data's adjacency mat between view i and view j
    for vi in range(data_param.get("num_view")):
        for vj in range(data_param.get("num_view")):
            Ni = mapped_data[vi]
            Nj = mapped_data[vj]
            temp = None
            for idx in range(data_param.get("num_sample")):
                data_vector = Ni[idx, :]
                extent_vector = np.outer(np.ones(data_param.get("num_sample")),data_vector)
                dif = extent_vector - Nj
                dif = dif * dif
                dif = np.sum(dif, 1).reshape(-1,1)
                if temp is None:
                    temp = dif
                else:
                    temp = np.concatenate((temp, dif),1)
            adjacency_list.append(temp)
            pass
    i = 0
    acc_list = []
    # access to each available adjacency matrixes among pair of each views
    while i < data_param.get("num_view"):
        j = i + 1
        while j < data_param.get("num_view"):
            adj_mat = adjacency_list[i * data_param.get("num_view") + j]
            # estimate if data vectors among this pair of views shared with same label or not
            label_i = labels[i]
            label_j = labels[j]

            # access to each pair of data vectors' distance among the adjacency mat
            acc = 0
            for selected_vector in range(data_param.get("num_sample")):
                indices = [a for a in range(data_param.get("num_sample")) if a != selected_vector]
                dist_array = adj_mat[indices,selected_vector]
                # select min value's idx in dist_array
                min_idx = 0
                for idx in range(max(dist_array.shape)):
                    if dist_array[min_idx] > dist_array[idx]:
                        min_idx = idx
                # if the vector that min_idx pointing to and selected vector shared with same value
                nearest_vector = indices[min_idx]
                # label_j[0,nearest_vector]
                if label_j[0,nearest_vector] == label_i[0,selected_vector]:
                    acc += 1
            acc_list.append(acc)
            j+=1
        i+=1
    pass

if __name__ == "__main__":
    # datas =
    pass