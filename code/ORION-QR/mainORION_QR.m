function mainORION_QR(datafileName, outputName)

trainSize = 23; % 23 files for training
testSize = 10; % 10 files for testing
fileSize = 9; % number of tasks + 1, because the first line of the file is not a task

InitialW0 = 0;
lambda = 0.1;
psi = 0.01;
mu = 1;
beta = 0.01;

[F1_train, F1_test] = restartUpdate(datafileName, fileSize, trainSize, testSize, mu, lambda, beta, psi, InitialW0);

save(outputName, 'lambda', 'psi', 'mu', 'beta', 'F1_train', 'F1_test');