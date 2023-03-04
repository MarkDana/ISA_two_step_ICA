function [y, W] = W_adasize_groupwise(lambda, Mask, WW, x_all)
% ICA with SCAD penalized entries of the de-mixing matrix
% WW: initialization;
% x_all: structure contains all demixing matrices

[N,T] = size(x_all{1});
Num_Sub = length(x_all);
mu = 2E-3; % 1E-6 1E-3

% initialization
for ii = 1:Num_Sub
    W{ii} = WW;
    
    z{ii} = zeros(N,N);
    eta{ii} = mu * ones(size(W{ii}));
    W_old{ii} = W{ii} + eye(N);
    grad_new{ii} = W_old{ii};
    y_psi{ii} = [];
    
end

% xx = x - mean(x')'*ones(1,T);
% % % To avoid instability
% std_xx = std(xx');
% xx = diag(1./std_xx) * x;

% % Refine = 0;
Num_edges = sum(sum(Mask));

% learning rate

beta = 0; % 1
save_intermediate = 1;
m = 50; % for approximate the derivative of |.|; % 60
% a = 3.7;
itmax = 4000; % 10000
iter_M = 200;
delta_H = 0;
Tol = 1e-4;
w11_back = [];
w12_back = [];
W_backup = zeros(N,N,1);
eta_backup = zeros(N,N,1);
z_backup = zeros(N,N,1);
grad_backup = zeros(N,N,1);

% % % initiliazation
% % fprintf('Initialization....\n')
% % if ~exist('WW','var')
% %     WW = diag(1./std(xx'));
% % end
% % WW = natural_grad_Adasize_Mask(xx, Mask,WW);
% % save WW_temp.mat WW;
% % % load WW_temp.mat; WW = WW * diag(std(x'));
% % omega1 = 1./abs(WW(Mask~=0));
% % % to avoid instability
% % Upper = 3 * mean(mean(omega1));
% % omega1 = (omega1>Upper)*Upper + omega1.*(omega1<=Upper);
% %
% % omega = zeros(N,N);
% % omega(Mask~=0) = omega1;
% % W = WW;
% %
% % z = zeros(N,N);
% % eta = mu * ones(size(W));
% % W_old = W + eye(N);
% % grad_new = W_old;
% % y_psi = [];

fprintf('Starting penalization...\n');
% load W_group_tmp.mat;
% load W_79_after5000_init.mat
% load W_group_tmp_5000_logTover2.mat;
% load W_group_tmp.mat;
load W_group_tmp_5000_logTover2.mat

for iter = 1:itmax
    %     if iter>iter_M
    %         mu = mu*0.99;
    %     end
    fprintf('.');
    if ~mod(iter,100)
        fprintf('\n');
    end
    
    total_grad = 0;
    for ii = 1:Num_Sub
        y{ii} = W{ii} * x_all{ii};
        
        %     % normalization
        %     W = diag(1./std(y')) * W;
        %     y = diag(1./std(y')) * y;
        
        % sum(sum(Mask)) * sum(sum(abs(W - W_old))),
        % sum(sum(abs(grad_new)))/N^2,
        % if sum(sum(Mask)) * sum(sum(abs(W - W_old)))<Tol
        total_grad = total_grad + sum(sum(abs(grad_new{ii} .* Mask)))/Num_edges;
        
        %     if sum(sum(abs(grad_new .* Mask)))/Num_edges<Tol
        %         if Refine
        %             Mask = abs(W) > 0.02;
        %             Mask = Mask - diag(diag(Mask));
        %             lambda = 0;
        %             Refine = 0;
        %         else
        %             break;
        %         end
        %     end
        W_old{ii} = W{ii};
        
        % update W: linear ICA with marginal score function estimated from data...
        if mod(iter,12) ==1
            if ii == 1 
                iter,
                save W_group_tmp.mat W y;
            end
            for i = 1:N
                tem = estim_beta_pham(y{ii}(i,:));
                y_psi{ii}(i,:) = tem(1,:);
                [y0{ii,i} II]  = sort(y{ii}(i,:), 'ascend'); % [y0{i} II JJ]  = unique(y(i,:));
                y_psi0{ii,i} = y_psi{ii}(i,II);
            end
        else
            for i = 1:N
                %             if length(y0{i})~=length(y(i,:))
                %                 y0{i} = [y0{i} y0{end}+ 1E-3*std(y0{i}) * [1:length(y(i,:))-length(y0{i})] ];
                %                 y_psi0{i} = [y_psi0{i} y_psi0{i}(end) * ones(1, length(y(i,:))-length(y0{i}))];
                %             end
                [tmp, II2] = sort(y{ii}(i,:), 'ascend');
                y_psi{ii}(i,II2) = y_psi0{ii,i};
                %             y_psi(i,:) = lininterp1(y0{i}', y_psi0{i}', y(i,:)')';
            end
        end
        
        dev{ii} = zeros(size(W{ii}));
        for jj = 1:Num_Sub
            if jj~=ii
                dev{ii} = dev{ii} + tanh( m*(W{ii} - W{jj}) );
            end
        end
        grad_new{ii} = y_psi{ii}*x_all{ii}'/T + inv(W{ii}') - lambda/T * dev{ii};
        % %
        % %
        % %
        % %     dev{ii} = omega .* tanh(m*W);
        % % % %     dev = omega .* sign(W);
        % %     %     dev = zeros(size(W));
        % %     %     grad_new = y_psi*x'/T + inv(W') - dev*lambda/T;
        % %     % modified objective function
        % %     grad_new = y_psi*x'/T + inv(W') -4*beta* (diag(diag(y*y'/T)) - eye(N)) * (y*x'/T) - dev*lambda/T;
        if iter==1
            grad_old{ii} = grad_new{ii};
        end
        %     G = y_psi * y'/T - lambda * dev * W';
        %     yy = y*y'/T;
        %     I_N = eye(N);
        %     H = G - diag(diag(G)) + I_N - diag(diag(yy));
        %     W = (I_N + mu*H) * W;
        
        % adaptive size
        [eta{ii}, z{ii}] = adaptive_size(grad_new{ii}, grad_old{ii}, eta{ii}, z{ii});
        delta_W{ii} = eta{ii}.*z{ii};
        %         delta_A = mu2 * grad_new;
        W{ii} = W{ii} + .9* delta_W{ii} .* Mask;
        %     delta_H = norm(H),
        %     if delta_H < Tol
        %         break;
        %     end
        grad_old{ii} = grad_new{ii};
    end
    total_grad,
    if total_grad<Tol
        break;
    end
    if total_grad>1E5
        pause;
    end
    
    
    if save_intermediate
        W_backup(:,:,iter) = W{1};
        z_backup(:,:,iter) = z{1};
        eta_backup(:,:,iter) = eta{1};
        grad_backup(:,:,iter) = grad_new{1};
    end
    %     if sum(sum(abs(delta_W)))<Tol
    %         break;
    %     end
end

% % % re-scaling
% % W = diag(std_xx) * W * diag(1./std_xx);
% % WW = diag(std_xx) * WW * diag(1./std_xx);
% % y = diag(std_xx) * y;


% %% further process the matrix W_m and WW_m so that the corresponding causal system is stable
% W_perm = W;
% Row_switch = [];
%
% for i = 1:N
%     [tmp, II] = max(abs(W_perm(i,:)));
%     if II~=i & ~ismember(i,Row_switch)
%         if abs(W_perm(i,II)) * abs(W_perm(II,i)) > abs(W_perm(i,i)) * abs(W_perm(II,II))
%             tmp_row = W_perm(i,:);
%             W_perm(i,:) = W_perm(II,:);
%             W_perm(II,:) = tmp_row;
%             Row_switch = [Row_switch II];
%         end
%     end
% end
% %%%%%%%%%
% W_perm = W;
% Row_switch = [];
%
% for i = 1:N
%     [tmp, II] = max(abs(W_perm(i,:) .* W_perm(:,i)'));
%     if II~=i & ~ismember(i,Row_switch)
%  %       if abs(W_perm(i,II)) * abs(W_perm(II,i)) > abs(W_perm(i,i)) * abs(W_perm(II,II))
%             tmp_row = W_perm(i,:);
%             W_perm(i,:) = W_perm(II,:);
%             W_perm(II,:) = tmp_row;
%             Row_switch = [Row_switch II];
%  %       end
%     end
% end

% figure, for i=1:4 for j=1:4 subplot(4,4,(i-1)*4 + j), plot(squeeze(W_backup(i,j,:))); end; end
% figure, for i=1:4 for j=1:4 subplot(4,4,(i-1)*4 + j), plot(squeeze(eta_backup(i,j,:))); end; end
% figure, for i=1:4 for j=1:4 subplot(4,4,(i-1)*4 + j), plot(squeeze(z_backup(i,j,:))); end; end
% % Score = omega .* abs(W);