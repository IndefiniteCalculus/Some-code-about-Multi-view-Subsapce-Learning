from Data import dataloader
from MvCCDA_OOP import MvCCDA
from Preprocessing import MvPCA_matlab

def do_train(dataset_name):
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
    model = MvCCDA(
        algorithm="LPP", sigma = 200, lambda1=0.6, lambda2=0.02, lambda3=5e-2 # /k=1 5s 82.0% 6590 73% / k=5 21s 81.2% 6462 73% / k=7 21s 81.4% 6494 73%
    )
    t1 = time.time()
    map_matrices, common_comp = model.train(tr_MvData, tr_label, rand_seed=0, )
    t2 = time.time()
    t = t2 - t1
    print("training done")
    import winsound
    winsound.Beep(600, 1000)
    return (model,map_matrices, common_comp), (tr_MvData, tr_label), (te_MvData, te_label), t

def knn_test(model, map_matrices, te_MvData, te_label,t,K_range=(1,5,1), test_data_name="original"):
    # using knn algorithm to evaluate the quality of mapped data
    ave_acc, ave_nmi, ave_D_acc = model.test(te_MvData, te_label, map_matrices, mode="don't project", K_Near=1)
    print("algorithm:" + str(model.algorithm) + " t: " + str(model.t) + ", l1: " + str(model.lambda1) + ", l2: " + str(
        model.lambda2) + ", l3: " + str(model.lambda3) + ", l4: " + str(model.lambda4))
    print(test_data_name+" ave_acc: " + str(ave_acc) + " ave_nmi: " + str(ave_nmi))
    print("time: " + str(t))
    for i in range(K_range[0], K_range[1], K_range[2]):
        ave_acc, ave_nmi, ave_D_acc = model.test(te_MvData, te_label, map_matrices, K_Near=i)
        print("k: "+str(i)+" ave_acc: " + str(ave_acc) + " ave_nmi: " + str(ave_nmi))

if __name__ == "__main__":
    (model,map_matrices, common_comp), (tr_MvData, tr_label), (te_MvData, te_label), t = do_train("mnist-usps")
    knn_test(model, map_matrices, te_MvData, te_label, t, K_range=(1, 5, 1))
    pass
# print("percentage of how many vector and it's nearest vector shared with same label among unmapped dataset: \n"+str(unmapped_ave_acc)+"%")
# print("percentage of how many vector and it's nearest vector shared with same label among mapped dataset: \n" + str(mapped_ave_acc)+"%")
pass
