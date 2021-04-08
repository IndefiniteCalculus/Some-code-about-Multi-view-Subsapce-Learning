function preprocess()
    DATA = load("E:\\Works\\数据集\\temp.mat"); % 暂时引入mat文件，功能测试完毕后会导入temp.mat文件直接使用
    data = DATA.data.';
    data = double(data);
    num_view = DATA.num_view;
    pca_dim = 80;
    swap_dir = "."
    % split data into nultiview mode
    MvData = cell(1,num_view);
    shape = size(data);
    each_num = shape(2) / num_view;
    for i = 1:num_view
        MvData{i} = data(: ,(i-1)*each_num+1 : i*each_num);
    end
    %DATA = load([swap_dir,'temp.mat']);
    %data = DATA.data;
    
    [x_mean x_var eig_value W_pca] = MvPCA(MvData);
    MvData = MvPCA_Projection(MvData, x_mean, x_var, W_pca, pca_dim);    
    save('E:\\Works\\数据集\\temp.mat', 'MvData');