function preprocess()
    DATA = load("E:\\Works\\数据集\\temp.mat"); % 暂时引入mat文件，功能测试完毕后会导入temp.mat文件直接使用
    tr_data = DATA.tr_data.';
    te_data = DATA.te_data.';
    tr_data = double(tr_data);
    te_data = double(te_data);
    num_view = DATA.num_view;
    dims_list = DATA.dim_list;
    pca_dim = DATA.pca_dim;
    
    % split tr_data into multiview tr_data
    tr_MvData = cell(1,num_view);
    te_MvData = cell(1,num_view);
    shape = size(tr_data);
    for i = 1:num_view
        if(i-1 < 1)
            start_dim = 1;
            end_dim = dims_list(1);
        else
            start_dim = sum(dims_list(1:i-1)) + 1;
            end_dim = sum(dims_list(1:i));
        end
        tr_MvData{i} = tr_data(start_dim : end_dim, : );
        te_MvData{i} = te_data(start_dim : end_dim, : );
    end
    %DATA = load([swap_dir,'temp.mat']);
    %tr_data = DATA.tr_data;
    
    [x_mean x_var eig_value W_pca] = MvPCA(tr_MvData);
    tr_MvData = MvPCA_Projection(tr_MvData, x_mean, x_var, W_pca, pca_dim);    
    te_MvData = MvPCA_Projection(te_MvData, x_mean, x_var, W_pca, pca_dim);
    save('E:\\Works\\数据集\\temp.mat', 'tr_MvData', 'te_MvData');