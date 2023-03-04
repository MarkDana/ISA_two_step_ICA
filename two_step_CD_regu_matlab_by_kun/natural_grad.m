function W = natural_gra(x,W)

[N,T] = size(x);
mu = 0.03;
itmax = 1500;
Tol = 1E-4 * N;

% [icasig, AA, W] = fastica(x, 'approach', 'symm', 'g', 'tanh');

for iter = 1:itmax
    fprintf('.');
    if ~mod(iter,10)
        fprintf('\n'); 
        h1 = subplot(1,2,1), cla(h1), hold on, plot(x(1,:), x(2,:), '.'); AA = inv(W); plot([3*AA(1,1); -3*AA(1,1)], [3*AA(2,1); -3*AA(2,1) ], 'r');
        plot([3*AA(1,2); -3*AA(1,2)], [3*AA(2,2); -3*AA(2,2) ], 'r');
        subplot(2,2,2), plot(y(1,:));  hold off; subplot(2,2,4), plot(y(2,:)); hold off; 
        pause(0.5);
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
delta_H/N,
        return;
    end
end
