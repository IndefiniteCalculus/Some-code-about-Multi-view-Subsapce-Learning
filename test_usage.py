from Data import dataloader
from MvCCDA_OOP import MvCCDA
from Preprocessing import MvPCA_matlab
import pickle
import numpy as np
def do_train(dataset_name, algorithm):
    #
    pca_dim = 80
    tr_MvData, tr_label, te_MvData, te_label = None, None, None, None
    if dataset_name == "cmu_pie":
        # cmu_pie
        tr_MvData, te_MvData, labels = dataloader.load_data("pca_pie")
        tr_label = labels.get("train")
        te_label = labels.get("test")
    elif dataset_name == "mnist-usps":
        # mnist-usps
        tr_MvData, va_MvData, te_MvData, tr_label, va_label, te_label = dataloader.load_data("pca_mnist-usps", pca_dim)
    elif dataset_name == "COIL":
        # COIL
        tr_MvData, tr_label, te_MvData, te_label, va_MvData, va_label = dataloader.load_data("pca_COIL", pca_dim)
    else:
        print("dataset "+dataset_name+" is not available yet")
        return None, None, None, None
    print("data load in, start training")
    import time
    is_MvCCDA = False
    if algorithm == "LPDP":
        model = MvCCDA(
            algorithm="LPDP", t = 1, sigma = 200, lambda1=0.06, lambda2=0.2, lambda3=5e-5, lambda4=5e-5
        )
        is_MvCCDA = True

    elif algorithm == "MMC":
        model = MvCCDA(
            algorithm="MMC", t=1, sigma=200, lambda1=0.6, lambda2=0.2, lambda3=0, lambda4=5e-4
        )
        is_MvCCDA = True

    elif algorithm == "LPP":
        model = MvCCDA(
            algorithm="LPP", sigma=200, lambda1=0.6, lambda2=0.02, lambda3=5e-2
        )
        is_MvCCDA = True

    elif algorithm == "PCA":
        pass
    elif algorithm == "CCA":
        import rcca
        num_class = len(set(np.ravel(tr_label[0]).tolist()))
        CCA = rcca.CCA(kernelcca = False, reg = 0., numCC = num_class)

    # training to find map_matrice based on different algorithm
    if is_MvCCDA:
        t1 = time.time()
        map_matrices, common_comp = model.train(tr_MvData, tr_label, rand_seed=0, )
        t2 = time.time()
        t = t2 - t1
    elif algorithm == "PCA":
        map_matrices, common_comp,t = None, None, None
        model = MvCCDA(
            algorithm="MMC", t=0, sigma=0, lambda1=0, lambda2=0, lambda3=0, lambda4=0
        )
    elif algorithm == "CCA":
        t1 = time.time()
        CCA.train(tr_MvData)
        t2 = time.time()
        t = t2 - t1
        model = CCA
        map_matrices = CCA.ws
        common_comp = CCA.comps
        for v in range(len(map_matrices)):
            map_matrices[v] = map_matrices[v].T

    print("training done")
    import winsound
    winsound.Beep(600, 1000)
    return (model,map_matrices, common_comp), (tr_MvData, tr_label), (te_MvData, te_label), t


def test(test_data, labels, map_matrices, mode="project and test", K_Near=1):
        # testing method aim to find each mapped data vectors nearest vector (use Eucilid distance). If the nearest
        # vector and selected vector shared with same label, the selected vector was mapped to a proper manifold.
        # TODO: design a independent cross-validation method(or refactor train, isolate the part related to num_class
        #  values, and optimized part as a independent method). The frame work need to be implement is：
        #  1: random split 2:k-fold

        # map data to manifold described by mapping matrices
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics.cluster import normalized_mutual_info_score
        num_view = len(test_data)
        num_sample = test_data[0].shape[0]
        if mode == "project and test":
            mapped_data = []
            for vi in range(num_view):
                mapped_vi = np.dot(test_data[vi], map_matrices[vi].T)
                mapped_data.append(mapped_vi)
                labels[vi] = np.ravel(labels[vi])
        else:
            mapped_data = test_data
            for vi in range(num_view):
                labels[vi] = np.ravel(labels[vi])
        # count the accuracy of data in the manifold
        acc_list = []
        NMI_list = []
        for vi in range(num_view):
            neigh = KNeighborsClassifier(n_neighbors=K_Near)
            _ =neigh.fit(mapped_data[vi], labels[vi])
            for vj in range(num_view):
                    if vi != vj:
                        pred = neigh.predict(mapped_data[vj])
                        label = labels[vj]
                        label = np.ravel(label)
                        t = pred - label
                        count = 0
                        for ele_idx in range(max(t.shape)):
                            if t[ele_idx] == 0:
                                count += 1
                        acc = count / num_sample
                        acc_list.append(acc)
                        NMI_score = normalized_mutual_info_score(pred, label)
                        NMI_list.append(NMI_score)

        acc_ave = sum(acc_list) / len(acc_list) * 100
        NMI_ave = sum(NMI_list) / len(NMI_list) * 100
        D_acc = []
        for acc in acc_list:
            D_acc.append((acc-acc_ave/100)**2)
        D_acc = sum(D_acc)**1/2 / len(D_acc) * 100
        return acc_ave, NMI_ave, D_acc

def knn_test(model, map_matrices, te_MvData, te_label,t,algorithm,K_range=(1,5,1), test_data_name="", mode="project and test"):
    # using knn algorithm to evaluate the quality of mapped data
    if algorithm != "PCA" and algorithm != "CCA":
        # print MvCCDA algorithm parameter setting
        print("algorithm:" + str(algorithm) + " t: " + str(model.t) + ", l1: " + str(model.lambda1) + ", l2: " + str(
            model.lambda2) + ", l3: " + str(model.lambda3) + ", l4: " + str(model.lambda4))
        print("time: " + str(t))

    if algorithm == "CCA":

        for i in range(K_range[0], K_range[1], K_range[2]):
            ave_acc, ave_nmi, D_acc = test(te_MvData, te_label, map_matrices, K_Near=i,mode=mode)
            print("k: "+str(i)+" ave_acc: " + str(ave_acc) +" D_acc: "+str(D_acc))

    if algorithm != "CCA":
        for i in range(K_range[0], K_range[1], K_range[2]):
            ave_acc, ave_nmi, D_acc = test(te_MvData, te_label, map_matrices, K_Near=i,mode=mode)
            print("k: "+str(i)+" ave_acc: " + str(ave_acc) +" D_acc: "+str(D_acc))

def save_to_recording(recording:dict):
    with open("recording.pickle","wb") as f:
        pickle.dump(recording,f)

def load_recoding():
    recording = None
    with open("recording.pickle","wb") as f:
        recording = pickle.load(f)
    if recording is None:
        recording = {"mnist-usps":None,"COIL":None,"CMU_PIE":None,"random_2D_sample": None}
    return recording

def visualization():
    from matplotlib.pyplot import plot

if __name__ == "__main__":
    dataset_name = "mnist-usps"
    algorithm = "CCA"
    (model,map_matrices, common_comp), (tr_MvData, tr_label), (te_MvData, te_label), t = do_train(dataset_name, algorithm)
    knn_test(model, map_matrices, te_MvData, te_label, t, algorithm = algorithm, K_range=(1, 17, 2),test_data_name=dataset_name)
    pass

pass
