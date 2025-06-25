function [ train_x,train_y,train_par,test_x,test_y,test_par ] = generateCVSet( X,partial_label,target,kk,index,totalCV )
    assert(index <= 10);
    assert(totalCV <= 10);
    m = size(X,1);
    slice = ceil(m/totalCV);
    test_x = X(kk((index - 1) * slice + 1: min( index * slice , m ) ) ,:);
    test_y = target(kk((index - 1) * slice + 1: min( index * slice , m ) ) ,:);
    test_par = partial_label(kk((index - 1) * slice + 1: min( index * slice , m ) ) ,:);
    
    
    train_x = X(setdiff(kk,kk((index - 1) * slice + 1: min( index * slice , m ) )),:);
    train_y = target(setdiff(kk,kk((index - 1) * slice + 1: min( index * slice , m ) )),:);
    train_par = partial_label(setdiff(kk,kk((index - 1) * slice + 1: min( index * slice , m ) )),:);
end