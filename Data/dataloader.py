import pickle
import os
import transform_ORL
def load_data(data_name):

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
        # TODO: after transform_ORL complete, format data to meet the requirement of model
        root_dir = "E:\\Works\\数据集\\ORL\\The ORL Database of Faces" # edit root_dir as the path of ORL database
        transform_ORL.do_transform(root_dir)

    else:
        return None, None, None

    return train_set, test_set, label_set

def load_param():
    root_dir = os.getcwd() + "\\Data"
    with open(root_dir+"\\parameter_initial.pickle", 'rb') as param_f:
        param = pickle.load(param_f)
        return param

if __name__ == "__main__":
    train_set, test_set, label_set = load_data("ORL")

    pass