function [W,y] = ICA_natural_grad(x)

[N,T] = size(x);
x = x - repmat(mean(x')', 1, T);
mu = 0.04;
itmax = 3000;
Tol = 1E-8 * N;

% initialization
W = rand(N,N) - 0.5;
% Orthogonalization such that W * Cov(x) * W' = I.
W = inv(sqrtm(W* x*x'/T * W')) * W;

for iter = 1:itmax
    fprintf('.');
    if ~mod(iter,50)
        fprintf('\n');
    end
    y = W * x;
    
    % update W: linear ICA with marginal score function estimated from data...
    y_psi = [];
    for i = 1:N
        tem = estim_beta_pham(y(i,:));
        y_psi =[y_psi; tem(1,:)];
    end
    
    G = y_psi * y'/T;
    yy = y*y'/T;
    I_N = eye(N);
    H = G - diag(diag(G)) + I_N - diag(diag(yy));
    W = (I_N + mu*H) * W;
    delta_H = norm(H);
    if delta_H < Tol 
        return;
    end
end
