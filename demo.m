clc; clear; 
addpath(genpath('.\'))
DataName={'music_emotion'};
for d=1:length(DataName)
    dataset=DataName{d};
    path=strcat('./data/',dataset);
    load(path);
  
    data = zscore(data);
    [num_data, dim] = size(data);
    randorder = randperm(num_data);
    cv_num = 5;
    selectratio = 0.025:0.025:0.5;
    mu = 0.1; rho = 1.05;
    alpha = 1e-4; lambda = 100;
    for cv=1:cv_num
        result_path=strcat(dataset,'/','cv',num2str(cv));
        mkdir(result_path);
        [cv_train_data,cv_train_target,cv_train_par,cv_test_data,cv_test_target,cv_test_par] = generateCVSet(data,partial_labels',target',randorder,cv,cv_num);
        [W, LD,obj, time] = PMFS_FLC(cv_train_data', cv_train_par', alpha,lambda, mu, rho);
        [dumb,idx] = sort(sum(W(1:dim,:).*W(1:dim,:),2),'descend');
        HL=[]; RL=[];OE=[];CV=[];AP=[];
        for feaIdx = 1:length(selectratio)
            feaNum = round(dim*selectratio(feaIdx));
            f = idx(1:feaNum);
            new_train = cv_train_data(:,f);
            new_test = cv_test_data(:,f);
            Pre_LD = classifier(new_train,LD,cv_train_par,new_test);
            % eavluation
            RL(feaIdx) = Ranking_loss(Pre_LD',cv_test_target');       % Ranking Loss
            AP(feaIdx) = Average_precision(Pre_LD',cv_test_target');  % Average Precision
            OE(feaIdx) = One_error(Pre_LD',cv_test_target');          % One Error
            CV(feaIdx) = coverage(Pre_LD',cv_test_target');           % Coverage
            bin_Pre_LD = binaryzation(softmax(Pre_LD')',0.1);
            bin_test_target = binaryzation(softmax(cv_test_target')',0.1);
            HL(feaIdx) =  Hamming_loss(bin_Pre_LD',bin_test_target');
        end
        savepath=strcat(result_path,'/','result.mat');
        save(savepath,'HL','RL','OE','CV','AP');   
    end
end
  
    
    