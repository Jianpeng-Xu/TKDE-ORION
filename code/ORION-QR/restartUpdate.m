function [F1_train, F1_test] = restartUpdate(datafileName, fileSize, trainSize, testSize, mu, lambda, psi, beta, InitialW0)
% This function is used to evaluate the ORION-QR on soil moisture data, handling missing
% values

% save V and w0
w0Cell = cell(0);
VCell = cell(0);

%% load data
load(datafileName);

%% define prior knowledges
% add the bias
[n,d] = size(X);
X = [ones(n,1), X];
Y = y;
T = fileSize-1;

% task relationships:
S = zeros(T, T);
for i = 1:T
    for j = 1:T
        if i==j+1 || i==j-1
            S(i,j) = 1;
        elseif i == j
            S(i,j) = -1;
        end
    end
end
S(1,1) = 0;
S(T, T) = 0;

%% define the extreme value
% assume the distribution of y is normal distribution
Y_timeseries = [];
for i = 1:trainSize
    if i == 1
        Y_timeseries = y_matrix(:,i);
    else
        Y_timeseries = [Y_timeseries;y_matrix(end,i)];
    end
end
mean_y = mean(Y_timeseries);
std_y = std(Y_timeseries);
% define the extreme value as mu+1.5*sigma
extremeHighThreshold = mean_y + 1.64*std_y;


%% construct training and validation dataset
X_train = X(1:trainSize*fileSize , :);
Y_train = Y(1:trainSize*fileSize);
Y_train_extremeLabel = (Y_train >=extremeHighThreshold); 
% if sum(Y_train_extremeLabel)==0
%     fprintf('No extreme value for this dataset, return\n');
%     return; 
% end
time_train = time(1:trainSize*fileSize,:);
ensembleMean_train = ensembledMean(1:trainSize*fileSize);

X_test = X(trainSize*fileSize+1:trainSize*fileSize + testSize*fileSize,:);
Y_test = Y(trainSize*fileSize+1:trainSize*fileSize + testSize*fileSize);
Y_test_extremeLabel = (Y_test >= extremeHighThreshold);
time_test = time(trainSize*fileSize+1: trainSize*fileSize + testSize*fileSize,:);
ensembleMean_test = ensembledMean(trainSize*fileSize+1:trainSize*fileSize + testSize*fileSize);

Y_test_hat_extremeLabel = zeros(size(Y_test_extremeLabel));
Y_train_hat_extremeLabel = zeros(size(Y_train_extremeLabel));

%% online testing
[n,d] = size(X_train);

if InitialW0 == 0
    % w0 = [0 ones(1,d-1).* (1/(d-1))]';
    w0 = [0 zeros(1,d-1).* (1/(d-1))]';
else
    % take the first InitialW0 files to train a batch w0
    num = trainSize * InitialW0;
    index = [1:num*T];
    x = X_train(index, :);
    y = Y_train(index);
    lambda = 1;
    % w = inv(x'*x + lambda*eye(d))*x'*y;
    w0 = (x'*x + lambda*eye(d))\x'*y;
end

V = zeros(T, d);

% record
recordCount = 1;
w0Cell{1} = w0;
Vcell{1} = V;


Y_hat = [];

positionRem = 1; % the position + 1 for Rem of w0 and V
err_train = zeros(trainSize - T, 1);
w0Rem = w0;
VRem = V;

for j = 1:trainSize - T
    % resume the parameter without missing value
    w0 = w0Rem;
    V = VRem;
    % construct x and y for training with missing values
    for t = positionRem : positionRem + T - 1
        index = [2+(t-1)*fileSize:t*fileSize];
        x = X(index, :);
        y = Y(index, :);
        L = (ones(T, 1)==1);
        % construct missing values
        if t ~= positionRem
            L(T - (t - positionRem-1): end) = 0;
            y(T - (t - positionRem-1): end)=NaN;
        end
        [w0, V]=ORION_QR(w0, V, L, x, y, mu, lambda, psi, beta, S);

        if t == positionRem
            w0Rem = w0;
            VRem = V;
        end
    end
    positionRem = positionRem+1;
    Wt = repmat(w0', [T,1]) + V;
    % Yhat = sum(X.*Wt,2);
    index = [2+(j + T-1)*fileSize:(j + T)*fileSize];
    y_hat = sum(X_train(index, :) .* Wt, 2);
    err_train(j) =  mean(abs(y_hat - Y_train(index, :)));
    Y_train_hat_extremeLabel(index) = (y_hat >= extremeHighThreshold);
    % record w0 and V
    recordCount = recordCount + 1;
    w0Cell{recordCount} = w0;
    VCell{recordCount} = V;
    
    index = [1+(j + T-1)*fileSize:(j + T)*fileSize];
    Y_hat = [Y_hat; Y_train(1+(j + T-1)*fileSize); y_hat];
    
end

w0Rem = w0;
VRem = V;
positionRem = trainSize - T + 1; % the position + 1 for Rem of w0 and V
err_test = zeros(size(Y_test));
for j = 1:testSize
    % resume the parameter without missing value
    w0 = w0Rem;
    V = VRem;
    % construct x and y for training with missing values
    for t = positionRem : positionRem + T - 1
        index = [2+(t-1)*fileSize:t*fileSize];
        x = X(index, :);
        y = Y(index, :);
        L = ones(T, 1);
        % construct missing values
        if t ~= positionRem
            L(T - (t - positionRem-1): end) = 0;
            y(T - (t - positionRem-1): end)=NaN;
        end
        [w0, V]=ORION_QR(w0, V, L, x, y, mu, lambda, psi, beta,S);
        if t == positionRem
            w0Rem = w0;
            VRem = V;
        end
    end
    positionRem = positionRem+1;
    index = [2+(j-1)*fileSize:(j)*fileSize];
    x = X_test(index, :);
    y = Y_test(index);
    Wt = repmat(w0',[T, 1]) + V;
    y_hat = sum(x.*Wt,2);
    err_test(index) = (abs(y_hat - y));
    Y_test_hat_extremeLabel(index) = (y_hat >= extremeHighThreshold);
    % record w0 and V
    recordCount = recordCount + 1;
    w0Cell{recordCount} = w0;
    VCell{recordCount} = V;
    
    index = [1+(j-1)*fileSize:(j)*fileSize];
    Y_hat = [Y_hat; Y_test(1+(j-1)*fileSize); y_hat];
end

firstRowIndexTest = 1:fileSize:fileSize*testSize;
Y_test_hat_extremeLabel(firstRowIndexTest) = [];
Y_test_extremeLabel(firstRowIndexTest) = [];
err_test(firstRowIndexTest) = [];

% averageErr_train_exteme = mean(err_train(Y_train_extremeLabel==1));
F1_train = Fmeasure(Y_train_extremeLabel, Y_train_hat_extremeLabel);

% averageErr_test_extreme = mean(err_test(Y_test_extremeLabel==1));
F1_test = Fmeasure(Y_test_extremeLabel, Y_test_hat_extremeLabel);


end