function [w0, V]=ORION_QR(w0, V, L, X, Y, mu, lambda, psi, beta, S)
% This function is to solve ORION_QR

% Input: w0 and V are the parameters in previous round
%        L is the available task set, 0 for missing, 1 for not missing
%        X, Y are the input feature space and output for each task
%        S is the adjacency matrix of the tasks 
%        If task i is missing, then Y(i) = NaN (L will indicate which Y is NaN anyway)
% We are going to use CVX to solve the quadratic programming problem. 
% Make sure CVX is installed property on your machine 


[T, d] = size(V);
Lap = eye(T) - S;

[T,d] = size(V);
index = find(L);
x = X(index, :);
y = Y(index);
Olength = length(index);
% if mu < 2
%     mu = 2;
% end
LapNew = Lap + mu*eye(T);
% [Lapsqrt, p] =  chol(LapNew);
[Lapsqrt, p] =  chol(LapNew);

if p ~= 0
    error('not convex when mu = %d\n', mu);
    return;
end
tau = 0.95;
cvx_begin quiet
    variable Vnew(T,d);
    variable w0new(d);
    variable p(T);
    variable q(T);
    minimize(tau * ones(1,T) * p + (1-tau) * ones(1,T) * q + (psi/2) * sum_square(vec(Vnew' * Lapsqrt')) + (lambda/2) * sum_square_abs(w0new - w0)+ (beta/2)*sum_square(vec(Vnew - V)));
    subject to
        y - sum(x.*(Vnew(index,:) + ones(Olength,1)*w0new'),2) == p(index) - q(index);
        p>=0;
        q>=0;
cvx_end

V = Vnew;
w0 = w0new;
end
