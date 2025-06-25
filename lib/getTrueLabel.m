function XU = getTrueLabel(P, Y, tuneThreshold)
    if tuneThreshold == 1   
        nRow = size(P,1);
       for i = 1:nRow
          sum_row = sum(P(i,:));
          fscore(i,:) = P(i,:)/(sum_row+eps);
       end
        [tau,  ~] = TuneThreshold( fscore, Y,1, 1);
        XU        = Predict(P,tau);
    else
        XU = double(P>=0.5);
    end
end
