function mainORION(datafileName, outputName)

trainSize = 23; % 23 files for training
testSize = 10; % 10 files for testing
fileSize = 9; % number of tasks + 1, because the first line of the file is not a task
epsilon = 0.0001;
InitialW0 = 0;
lambda = 10;
mu = 0.1;
beta = 0.01;

[err_train, err_test] = restartUpdate(datafileName, fileSize, trainSize, testSize, mu, lambda, beta, epsilon, InitialW0);

save(outputName, 'epsilon', 'lambda', 'mu', 'beta', 'err_train', 'err_test');