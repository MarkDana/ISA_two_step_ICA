function W = natural_gra_Mask(x,Mask)

[N,T] = size(x);
mu = 0.03;
itmax = 2500;
Tol = 1E-5 * sum(sum(Mask~=0));

% [icasig, AA, W] = fastica(x, 'approach', 'symm', 'g', 'tanh');

% initilization of W
WW = eye(N,N);
for i = 1:N
    Ind_i = find(Mask(i,:)~=0);
    WW(i,Ind_i) = -.5*(x(i,:) * x(Ind_i,:)') * pdinv(x(Ind_i,:) * x(Ind_i,:)');
end
W = .5*(WW+WW');

for iter = 1:itmax
    fprintf('.');
    if ~mod(iter,100)
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
%      H = G - diag(diag(G)) + I_N - diag(diag(yy));
%     H = G - diag(diag(G));
%     W = (I_N + mu*H) * W;
    Grad_W = y_psi*x'/T + inv(W') ;
    H = Grad_W .*Mask;
    W = W + mu * H;
    delta_H = sum(abs(H)),
    if delta_H < Tol
delta_H/N,
        return;
    end
end
