
% Execution time vector
t_ex = [47.08, 0.058, 0.054, 0.051, 33.027, 0.255, 0.057, 0.045];

% Training performance scores
acc_train = [100 79.68 83.29 36.86 99.98 64.06 96.21 95.96]/100;
prec_train = [100 83.10 84.72 38.57 99.98 94.80 96.91 96.69]/100;
rec_train = [100 79.68 83.29 36.86 99.98 94.06 96.21 95.96]/100;

% Testing performance scores
acc_test = [100 78.20 78.10 67.79 99.83 97.11 98.83 98.53]/100;
prec_test = [100 64.09 68.22 41.61 99.68 86.42 93.44 92.49]/100;
rec_test = [100 79.07 83.02 36.82 99.92 93.93 96.19 95.90]/100;

score = (0.4*min(t_ex)./t_ex + 0.2/3*(acc_train/max(acc_train) + prec_train/max(prec_train) ... 
    + rec_train/max(rec_train)) + 0.4/3*(acc_test/max(acc_test) + prec_test/max(prec_test) ...
    + rec_test/max(rec_test)))


