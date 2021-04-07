import numpy as np
from Data import dataloader
from scipy.linalg import eigh


def num2onehot(label_set):
    # 以(num_sample, num_classes)的格式组织独热编码，假设训练数据都包含了所有的类
    num_classes, max_l, min_l = obtain_numclass(label_set)
    v_labels = {}
    for data_type in label_set.keys():
        v_labels.update({data_type: None})
    for data_type in label_set.keys():
        labels = label_set.get(data_type)
        onehots = []
        for view in labels:
            onehot = np.zeros((view.shape[1], num_classes))
            for dim1 in range(view.shape[1]):
                l = view[0, dim1] - min_l
                onehot[dim1, l] = 1
            onehots.append(onehot)
        v_labels.update({data_type:onehots})
    pass
    return v_labels, num_classes


def get_hotkernel_mat(label_set, train_data, data_param):
    _, max_l, min_l = obtain_numclass(label_set)
    # obtain similarity matrix on multi_view
    Sv = None
    label = label_set.get("train")
    for view_label, view_data in zip(label, train_data):
        # for each view
        similarity_mat = np.zeros((data_param.get("num_sample"), data_param.get("num_sample")))
        for i in range(data_param.get("num_sample")):
            # for each pair of vectors in data
            vector_i = view_data[i, :]
            for j in range(data_param.get("num_sample")):
                vector_j = view_data[j, :]
                e_v = vector_i - vector_j
                similarity_mat[i, j] = np.exp(-np.dot(e_v, e_v.T)/sigma)
        if Sv is None:
            Sv = similarity_mat
        else:
            Sv += similarity_mat
    Sv = Sv / data_param.get("num_view")
    # obtain hot kernel matrix on the basis of similarity matrix and respective data labels
    for i in range(data_param.get("num_sample")):
        for j in range(data_param.get("num_sample")):
            if label[0][0,i] != label[0][0,j]:  # only consider the similarity of data shared with same labels
                Sv[i, j] = 0
    return Sv


def obtain_numclass(label_set):
    # obtain number of class in the label_set. Assuming all classes have been present in the label_set
    min_l, max_l = float("inf"), 0
    # obtain max label and min label among all dataset
    for data_type in label_set.keys():
        temp_min, temp_max = np.min(label_set.get(data_type)), \
                             np.max(label_set.get(data_type))
        if temp_max > max_l:
            max_l= temp_max
        if temp_min < min_l:
            min_l = temp_min
    num_classes = max_l - min_l + 1
    return num_classes, max_l, min_l


def obtain_numsample(data):
    # obtain number of samples in the label vectors
    return data.shape[0]


def obtain_numdim(data):
    # obtain dimension of data
    return data.shape[1]


def obtain_numview(data_set):
    # obtain number of view the dataset have
    return len(data_set)


def train(train_data, map_matrixes, common_comp, onehots, hotkernel_mat, data_param, hyper_param):
    # iterative update reduced_data and map_matrix
    all_coveraged = None
    iteration = 0
    while True:

        # update all common component of each sample vector
        for i in range(data_param.get("num_sample")):
            weight_map_i = np.sum(hotkernel_mat[i, :])
            weighted_comp_i = np.dot(hotkernel_mat[i, :].reshape(1,-1), common_comp)

            # optimize the component based on map matrix
            optimized_count = 0
            coveraged = None
            # optimize common component zi
            while True:
                r = []
                Q = []
                # calculate the r of objective function
                for v_id in range(data_param.get("num_view")):
                    mapped_data = np.dot(map_matrixes[v_id], train_data[v_id][i, :].T)
                    r.append(common_comp[i,:] - mapped_data)

                # calculate the weight function Q in the objective function
                for v in range(data_param.get("num_view")):
                    Q.append(1/(np.dot(r[v],r[v].T) + hyper_param.get("alpha")**2))
                sum_Q = sum(Q)

                # assemble the two part of the objective function
                a = data_param.get("num_view") * hyper_param.get("lambda2") \
                + data_param.get("num_view") * hyper_param.get("lambda3") * weight_map_i \
                    + sum_Q

                b = num_view * hyper_param.get("lambda2") * onehots.get("train")[0][i, :] + num_view * hyper_param.get("lambda3") * weighted_comp_i
                for v in range(data_param.get("num_view")):
                    b = b + Q[v] * np.dot(map_matrixes[v], train_data[v][i, :])

                # estimate loss of this component and break the loop if coverage
                updated_common_comp_i = np.linalg.inv(np.eye(hyper_param.get("subspace_dim"))*a).dot(b.reshape(-1,1))
                loss = updated_common_comp_i.reshape(1,-1) - common_comp[i, :]
                loss = np.dot(loss, loss.T)
                loss = np.sqrt(loss)
                # update
                common_comp[i, :] = updated_common_comp_i.T
                optimized_count += 1
                if loss < 0.0001:
                    coveraged = True
                    break
                if optimized_count > 200:
                    coveraged = False
                    break
        # optimize map matrix based on common_component
        optimized_count = 0
        coveraged = None

        # update component in view width
        last_map_matrixes = tuple(map_matrixes)
        for v in range(data_param.get("num_view")):
            map_mat_v = map_matrixes[v]
            # optimize mapping matrix on view v
            while True:
                G = []
                r = []
                # calculate the G of objective function
                for sample_id in range(data_param.get("num_sample")):
                    mapped_data = np.dot(map_mat_v, train_data[v][sample_id, :].T)
                    r_i = common_comp[sample_id, :] - mapped_data
                    r.append(r_i)
                    G.append(1 / (np.dot(r_i, r_i.T) + hyper_param.get("alpha") ** 2))
                data_X = train_data[v]
                G = np.array(G).reshape(-1,1)
                # assuming the structure of objective function should be like b / a,
                # the assemble of 'a' should be
                XG = np.dot(G * np.eye(data_param.get("num_sample")),
                            data_X)
                XGX = np.dot(data_X.T, XG)
                a = XGX + hyper_param.get("lambda1") * data_param.get("num_sample") * np.eye(data_param.get("num_dim"))
                # the assemble of 'b' should be
                # b = np.dot(common_comp.T, XG)
                GZ = np.dot(G * np.eye(data_param.get("num_sample")),
                            common_comp)
                b = np.dot(common_comp.T,XG)
                updated_map_mat_v = np.dot(np.linalg.inv(a),b.T).T

                # use Frobenius norm to estimate loss of mapping matrix
                loss = updated_map_mat_v - map_mat_v
                loss = loss * loss
                loss = np.sum(loss)
                loss = np.sqrt(loss)
                optimized_count += 1

                # update for next loop
                map_mat_v = updated_map_mat_v

                if loss < 1e-4:
                    coveraged = True
                    break
                if optimized_count > 200:
                    coveraged = False
                    break
                pass
            # obtain optimized map vector
            map_matrixes[v] = updated_map_mat_v
        iteration += 1
        # now estimate the max loss of map matrixes
        max_loss = 0
        for map_id in range(len(map_matrixes)):
            loss = last_map_matrixes[map_id] - map_matrixes[map_id]
            loss = loss * loss
            loss = np.sqrt(np.sum(loss))
            max_loss = max(loss, max_loss)
        if max_loss < 5e-7:
            all_coveraged = True
            break
        if iteration > 200:
            all_coveraged = False
            break
        pass
    return map_matrixes, common_comp


def test(data:list, labels, data_param):
    # the way of testing is find each mapped data vectors nearest vector (use Eucilid distance)
    # if the nearest vector and selected vector shared with same label, we think the map of selected vector is a success
    # if not, we think the mapping of this vector is a failure
    from sklearn.neighbors import KNeighborsClassifier
    acc_list = []
    for vi in range(data_param.get("num_view")):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(data[vi], labels[vi].reshape(-1, 1))
        for vj in range(data_param.get("num_view")):
            if vi != vj:
                pred = neigh.predict(data[vj])
                label = labels[vj]
                t = pred - label
                count = 0
                for ele_idx in range(max(t.shape)):
                    if t[0,ele_idx] == 0:
                        count += 1
                acc = count / data_param.get("num_sample")
                acc_list.append(acc)
    acc_ave = sum(acc_list) / len(acc_list) * 100
    return acc_ave


# hyperparameter setting
alpha = 2.3849
lambda1 = 0.6
lambda2 = 0.002
lambda3 = 0.0005
sigma = 2000
subspace_dim = 45
# pca_dim = 80

if __name__ == "__main__":
    train_data, test_data, labels = dataloader.load_data("matlabpca")

    num_sample = obtain_numsample(train_data[0])
    num_dim = obtain_numdim(train_data[0])
    num_view = obtain_numview(train_data)
    train_data_param = {"num_sample": num_sample, "num_dim": num_dim, "num_view": num_view}

    num_sample = obtain_numsample(test_data[0])
    num_dim = obtain_numdim(test_data[0])
    num_view = obtain_numview(test_data)
    test_data_param = {"num_sample": num_sample, "num_dim": num_dim, "num_view": num_view}

    hyperparameters = {"alpha": alpha, "lambda1": lambda1, "lambda2": lambda2, "lambda3": lambda3, "sigma": sigma, "subspace_dim": subspace_dim}

    onehots, _ = num2onehot(labels)
    hotkernel_mat = get_hotkernel_mat(labels, train_data, train_data_param)

    # initialization of parameters to be learned, size of row data should be (num_sample, num_dim)
    param = dataloader.load_param()
    common_comp = param.get("Z").T
    map_matrixes = param.get("P")
    for m_idx in range(len(map_matrixes)):
        map_matrixes[m_idx] = map_matrixes[m_idx].T

    map_matrixes, common_comp = train(train_data, map_matrixes, common_comp, onehots, hotkernel_mat, train_data_param, hyperparameters)
    # project test data to subspace
    mapped_data = []
    for vi in range(test_data_param.get("num_view")):
        mapped_vi = np.dot(test_data[vi], map_matrixes[vi].T)
        mapped_data.append(mapped_vi)

    mapped_ave_acc = test(mapped_data, labels.get("test"), test_data_param)
    unmapped_ave_acc = test(test_data, labels.get("test"), test_data_param)

    print(
        "percentage of how many vector and it's nearest vector shared with same label among unmapped dataset: \n" + str(
            unmapped_ave_acc) + "%")
    print("percentage of how many vector and it's nearest vector shared with same label among mapped dataset: \n" + str(
        mapped_ave_acc) + "%")
    pass