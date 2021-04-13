function preprocess()
    DATA = load("E:\\Works\\数据集\\temp.mat"); % 暂时引入mat文件，功能测试完毕后会导入temp.mat文件直接使用
    data = DATA.data.';
    data = double(data);
    num_view = DATA.num_view;
    dims_list = DATA.dim_list;
    pca_dim = DATA.pca_dim;
    swap_dir = ".";
    % split data into multiview data
    MvData = cell(1,num_view);
    shape = size(data);
    for i = 1:num_view
        if(i-1 < 1)
            start_dim = 1;
            end_dim = dims_list(1);
        else
            start_dim = sum(dims_list(1:i-1)) + 1;
            end_dim = sum(dims_list(1:i));
        end
        range = [start_dim, end_dim];
        MvData{i} = data(start_dim : end_dim, : );
    end
    %DATA = load([swap_dir,'temp.mat']);
    %data = DATA.data;
    
    [x_mean x_var eig_value W_pca] = MvPCA(MvData);
    MvData = MvPCA_Projection(MvData, x_mean, x_var, W_pca, pca_dim);    
    save('E:\\Works\\数据集\\temp.mat', 'MvData');