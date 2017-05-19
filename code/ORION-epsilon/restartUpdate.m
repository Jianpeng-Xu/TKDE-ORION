function [averageErr_train, averageErr_test] = restartUpdate(datafileName, fileSize, trainSize, testSize, mu, lambda, beta, epsilon, InitialW0)
% This function is used to evaluate ORION on soil moisture data, handling missing observations
% if InitialW0 is 1, then initialize W0 using ridge regression for each task. Otherwise, set it to 0, or 1/T

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

%% construct training and validation dataset
X_train = X(1:trainSize*fileSize , :);
Y_train = Y(1:trainSize*fileSize);
time_train = time(1:trainSize*fileSize,:);
ensembleMean_train = ensembledMean(1:trainSize*fileSize);
X_test = X(trainSize*fileSize+1:trainSize*fileSize + testSize*fileSize,:);
Y_test = Y(trainSize*fileSize+1:trainSize*fileSize + testSize*fileSize);
time_test = time(trainSize*fileSize+1: trainSize*fileSize + testSize*fileSize,:);
ensembleMean_test = ensembledMean(trainSize*fileSize+1:trainSize*fileSize + testSize*fileSize);

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
        [w0, V]=ORION(w0, V, L, x, y, mu, lambda, beta, epsilon, S);
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
    % record w0 and V
    recordCount = recordCount + 1;
    w0Cell{recordCount} = w0;
    VCell{recordCount} = V;
    
    index = [1+(j + T-1)*fileSize:(j + T)*fileSize];
    Y_hat = [Y_hat; Y_train(1+(j + T-1)*fileSize); y_hat];
    
end
% err_train = err_train / T;
averageErr_train = mean(err_train);

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
        [w0, V]=ORION(w0, V, L, x, y, mu, lambda, beta, epsilon, S);
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
    % record w0 and V
    recordCount = recordCount + 1;
    w0Cell{recordCount} = w0;
    VCell{recordCount} = V;
    
    index = [1+(j-1)*fileSize:(j)*fileSize];
    Y_hat = [Y_hat; Y_test(1+(j-1)*fileSize); y_hat];
end
averageErr_test = mean(err_test);

end