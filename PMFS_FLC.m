function [W, LD,obj, time] = PMFS_FLC(X, Y, alpha,lambda, mu, rho)
% ADMM method to solve problem min_{||W'*X+b1'-Y||_1 + alpha*tr(W'*X*L*X'*W)+lambda||W||_2,1
% X: d*n data matrix, each column is a data point
% Y: c*n partial-label matrix, Y(i,j)=1 if xi is labeled to j, and Y(i,j)=0 otherwise
% alpha:parameter of manifold regularization
% lambda:parameter of sparse regularization
% mu, rho: parameters in the ADMM optimization method
% W: d*c embedding matrix
% DL:c*n label distribution of train_data
% obj: objective values in the iterations

%% parameters setting
MaxIter = 50;
minimumLossMargin = 1e-5;
[dim, num_sample] = size(X);
num_class = size(Y,1);
obj = [];

%% similarity matrix of feature space 
options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'HeatKernel';
options.t = 2;%sum(sum(EuDist2(X)))/(num_sample-1);
S = constructW(X', options);

%% similarity matrix of label space 
options = [];
options.NeighborMode = 'KNN';
options.k = 0;
options.WeightMode = 'Cosine';
C = constructW(Y', options) ; 

%% Feature-Label Collaboration
A = S.*C;
L = diag(sum(A, 2))- A; %laplacian matrix   

%% initialization
trainFeature = [X;ones(1, num_sample)];
Sigma = zeros(num_class,num_sample);
E = rand(num_class,num_sample);
W = rand(dim+1,num_class);
XLXT = trainFeature*L*trainFeature';
XXT = trainFeature*trainFeature';

err = W'*trainFeature - Y;
oldloss = sum(sum(abs(err))) + alpha*trace(W'*trainFeature*L*trainFeature'*W) + lambda*(sum(sqrt(sum(W.*W,2)+eps)));
obj = [obj;oldloss];

iter = 1;
tic;
while iter <= MaxIter
    inmu = 1/mu;    
    %update W
    u = 0.5./sqrt(sum(W.*W, 2) + eps);
    U = diag(u);
    A = XXT + 2*alpha*inmu*XLXT + 2*lambda*inmu*U;
    B = trainFeature*(Y + E - inmu*Sigma)'; 
    W = A\B;
    
    %update E
    G = E-(W'*trainFeature - Y + inmu*Sigma);
    E = softthres(G, inmu);
    
    %update parameters
    Sigma = Sigma + mu*(W'*trainFeature-Y-E);
    mu = min(10^10,rho*mu);
    
    err = W'*trainFeature - Y;
    totalloss = sum(sum(abs(err))) + alpha*trace(W'*trainFeature*L*trainFeature'*W) + lambda*(sum(sqrt(sum(W.*W,2)+eps)));
    obj = [obj;totalloss];
     if (abs((oldloss - totalloss)/oldloss) <= minimumLossMargin) && (iter >= 5)
         break;
     elseif totalloss <=0
         break;
     else
           oldloss = totalloss;
     end
     iter = iter + 1;
end
time=toc;
LD = W'*trainFeature;
end

function W = softthres(W_t,lambda)
    W = max(W_t - lambda,0) - max(-W_t - lambda,0);  
end

