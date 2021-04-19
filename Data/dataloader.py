import pickle
import os
from Data import transform_ORL
from Data import transform_mnist_usps
from Preprocessing import DataType_Transformation as pca
def load_data(data_name, pca_dim = None):

    if data_name == "matlabtest":
        root_dir = os.getcwd() + "\\Data"
        with open(root_dir+"\\train_set.pickle", 'rb') as train_f, open(root_dir+"\\test_set.pickle",'rb') as test_f:
            train_set = pickle.load(train_f)
            test_set = pickle.load(test_f)

    elif data_name == "matlabrow":
        root_dir = os.getcwd() + "\\Data"
        with open(root_dir+"\\row_data.pickle","rb") as data_f, open(root_dir+"\\labels.pickle","rb") as label_f:
            row_data = pickle.load(data_f)
            train_set = row_data.get("train")
            test_set = row_data.get("test")
            label_set = pickle.load(label_f)

    elif data_name == "matlabpca":
        root_dir = os.getcwd() + "\\Data"
        with open(root_dir + "\\dim_reduced_data.pickle", 'rb') as dim_reduced_f, open(root_dir+"\\labels.pickle","rb") as label_f:
            dim_reduced_f = pickle.load(dim_reduced_f)
            train_set = dim_reduced_f.get("train")
            test_set = dim_reduced_f.get("test")
            label_set = pickle.load(label_f)

    elif data_name == "ORL":
        root_dir = "E:\\Works\\数据集\\ORL\\The ORL Database of Faces" # edit root_dir as the path of ORL database
        data, label = transform_ORL.do_transform(root_dir)
        return data, label

    elif data_name == "pie":
        root_dir = r"E:\\Works\\python_work\\Some code about Multi-view Subsapce Learning\\Data"
        with open(root_dir + "\\row_data.pickle", "rb") as data_f, open(root_dir + "\\labels.pickle", "rb") as label_f:
            row_data = pickle.load(data_f)
            train_set = row_data.get("train")
            test_set = row_data.get("test")
            label_set = pickle.load(label_f)

    elif data_name == "pca_mnist-usps":
        Mv_Data, labels = transform_mnist_usps.do_transform("E:\\Works\\数据集\\mnist-usps")
        from Data import mnist_usps_split
        tr_MvData, va_MvData, te_MvData, tr_labels, va_labels, te_labels \
            = mnist_usps_split.do_split(Mv_Data, labels)
        tr_MvPCA, te_MvPCA = pca.load_pca(tr_MvData, te_MvData, pca_dim)
        _, va_MvPCA = pca.load_pca(tr_MvData, va_MvData, pca_dim)
        return tr_MvPCA, va_MvPCA, te_MvPCA, tr_labels, va_labels, te_labels

    elif data_name == "mnist-usps":
        MvData, labels = transform_mnist_usps.do_transform("E:\\Works\\数据集\\mnist-usps")
        from Data import mnist_usps_split
        tr_MvData, va_MvData, te_MvData, tr_labels, va_labels, te_labels \
            = mnist_usps_split.do_split(MvData, labels)
        return tr_MvData, va_MvData, te_MvData, tr_labels, va_labels, te_labels

    elif data_name == "resized_mnist-usps":
        MvData, labels = transform_mnist_usps.do_transform("E:\\Works\\数据集\\mnist-usps", loadin_mode="resize")
        from Data import mnist_usps_split
        tr_MvData, va_MvData, te_MvData, tr_labels, va_labels, te_labels \
            = mnist_usps_split.do_split(MvData, labels)
        return tr_MvData, va_MvData, te_MvData, tr_labels, va_labels, te_labels

    elif data_name == "COIL":
        from Data import transform_COIL100
        Mv_Data, labels = transform_COIL100.do_transform("E:\\Works\\数据集\\COIL\\coil-100")
        tr_MvData, tr_MvLabel, te_MvData, te_MvLabel, va_MvData, va_MvLabel = transform_COIL100.split(Mv_Data, labels)
        return tr_MvData, tr_MvLabel, te_MvData, te_MvLabel, va_MvData, va_MvLabel

    elif data_name == "pca_COIL":
        from Data import transform_COIL100
        Mv_Data, labels = transform_COIL100.do_transform("E:\\Works\\数据集\\COIL\\coil-100")
        tr_MvData, tr_MvLabel, te_MvData, te_MvLabel, va_MvData, va_MvLabel = transform_COIL100.split(Mv_Data, labels)
        tr_MvPCA, te_MvPCA = pca.load_pca(tr_MvData, te_MvData, pca_dim)
        if va_MvData is not None:
            _, va_MvPCA = pca.load_pca(tr_MvData, va_MvData, pca_dim)
        return tr_MvPCA, tr_MvLabel,te_MvPCA, te_MvLabel, va_MvData, va_MvLabel

    else:
        return None, None, None

    return train_set, test_set, label_set

def load_param():
    root_dir = os.getcwd() + "\\Data"
    with open(root_dir+"\\parameter_initial.pickle", 'rb') as param_f:
        param = pickle.load(param_f)
        return param

if __name__ == "__main__":
    train_set, test_set, label_set = load_data("pca_mnist-usps")

    pass