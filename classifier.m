function [Pre_LD] = classifier(train_data,numerical,train_p_target,test_data)
numerical = (softmax(numerical))';
% setup parameters for classifier model
tol  = 1e-10;   %Tolerance during the iteration
epsi =0.1;      %Instances whose distance computed is more than epsi should be penalized
ker  = 'rbf';   %Type of kernel function
beta1=1;        %Penalty parameter
beta2=50;       %Penalty parameter
par = 1*mean(pdist(train_data));
% training classifier model
[Beta,b] = plmsvr(train_data,numerical,train_p_target,ker,beta1,beta2,epsi,par,tol);
Pre_LD = PL_LEAF_predict(train_data,test_data,ker,Beta,b,par);
end

