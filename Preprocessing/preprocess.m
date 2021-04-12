function preprocess()
    DATA = load("E:\\Works\\���ݼ�\\temp.mat"); % ��ʱ����mat�ļ������ܲ�����Ϻ�ᵼ��temp.mat�ļ�ֱ��ʹ��
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
        MvData{i} = data((i-1)*dims_list(i)+1 : i*dims_list(i), : );
    end
    %DATA = load([swap_dir,'temp.mat']);
    %data = DATA.data;
    
    [x_mean x_var eig_value W_pca] = MvPCA(MvData);
    MvData = MvPCA_Projection(MvData, x_mean, x_var, W_pca, pca_dim);    
    save('E:\\Works\\���ݼ�\\temp.mat', 'MvData');