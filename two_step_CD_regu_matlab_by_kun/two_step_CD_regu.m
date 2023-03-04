function [B,W_m,y_m] = two_step_CD_regu(X)
% function [B,W_m] = two_step_CD(X)
% Two-step method for linear causal discovery that allows cycles and
% confoudners
% Input: 
%   Data matrix X (variable number * sample size).
% Output: 
%   B: the causal influence matrix X = BX + E;
%   W_m: the ICA de-mixing matrix.

[N,T] = size(X);
X = X - mean(X')'*ones(1,T);
% % To avoid instability
std_X = std(X');
X = 1/mean(std_X) * X;

% estimate the mask
Mask = zeros(N,N);
for i=1:N
    if T<4*N % sample size too small, so preselect the features
        tmp1 = X([1:i-1 i+1:N],:);
        [tmp2, Ind_t] = sort(abs(corr( tmp1', X(i,:)' )), 'descend');
        X_sel = tmp1(Ind_t(1:floor(N/4)),:); % pre-select N/4 features
        [beta_alt, beta_new_nt, beta2_alt, beta2_new_nt] = betaAlasso_grad_2step(X_sel, X(i,:), 0.65^2*var(X(i,:)), log(T)/2); % 0.7^2
        beta2_al = zeros(N-1,1);
        beta2_al(Ind_t(1:floor(N/4))) = beta2_alt;
    else
        [beta_al, beta_new_n, beta2_al, beta2_new_n] = betaAlasso_grad_2step(X([1:i-1 i+1:N],:), X(i,:), 0.35^2*var(X(i,:)), log(T)/6); % 0.7^2
    end
    Mask(i,1:i-1) = abs(beta2_al(1:i-1)) >0.01;
    Mask(i,i+1:N) = abs(beta2_al(i:N-1)) >0.01;
end
Mask = Mask + Mask';
Mask = (Mask~=0);

% perform constrained_ICA
regu = 0; % 0.05
[y_m, W_m, WW_m, Score] = sparseica_W_adasize_Alasso_mask_regu(log(T)/4, Mask, X, regu);
B = eye(N) - W_m;
B = B .* (abs(B) > 0.02);

% figure, subplot(1,3,1), imagesc(WW_m); colorbar; subplot(1,3,2), imagesc(W_m); colorbar; subplot(1,3,3), imagesc(eye(N) - W_m); colorbar;