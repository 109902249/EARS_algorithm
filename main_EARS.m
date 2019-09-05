%--------------------------------------------------------------------------
% Corresponding author: Qi Zhang
% Department of Applied Mathematics and Statistics,
% Stony Brook University, Stony Brook, NY 11794-3600
% Email: zhangqi{dot}math@gmail{dot}com
%--------------------------------------------------------------------------
% 1. The EARS algorithm by Qi Zhang and Jiaqiao Hu [1] is implemented for
% solving single-objective box-constrained expensive deterministic
% optimization problems.
% 2. In this implementation, the algorithm samples candidate solutions from 
% a sequence of independent multivariate normal distributions that 
% recursively approximiates the corresponding Boltzmann distributions [2].
% 3. In this implementation, the surrogate model is constructed by the 
% radial basis function (RBF) method [3].
%--------------------------------------------------------------------------
% REFERENCES
% [1] Qi Zhang and Jiaqiao Hu (2019): Enhancing Random Search with 
% Surrogate Models for Continuous Optimization. Proceedings of the 
% IEEE 15th International Conference on Automation Science and Engineering,
% forthcoming.
% [2] Jiaqiao Hu and Ping Hu (2011): Annealing adaptive search,
% cross-entropy, and stochastic approximation in global optimization.
% Naval Research Logistics 58(5):457-477.
% [3] Gutmann HM (2001): A radial basis function method for 
% global optimization. Journal of Global Optimization 19:201-227.
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------
clearvars; close all;
%% PROBLEM SETTING
% All test functions H(x) are in dimension 20 (d=20),
% with box-constrain [-10,10]^d,
% and scaled to have an optimal (max) objective function value -1.
d=20; % dimension of the search region
left_bound=-10; % left bound of the search region
right_bound=10; % right bound of the search region
optimal_objective_value=-1; % optimal objective function value
%--------------------------------------------------------------------------
% The test functions can be chosen from 
% 1. 'sumSquares'
% 2. 'bohachevsky'
% 3. 'cigar'
% 4. 'ackley'
% 5. 'qing'
% 6. 'styblinskiTang'
% 7. 'pinter'
fcn_name='sumSquares';
%--------------------------------------------------------------------------

%% HYPERPARAMETERS
budget=2000; % total number of function evaluations assigned
warm_up=5*d; % the number of function evaluations used for warm up 
% alpha: learning rate for updating the mean parameter function m(theta)
% alpha=1/(k-warm_up+ca)^gamma0
ca=100; gamma0=.51;
% exploration: sampling exploration factor
exploration=0;

%% INITIALIZATION
% initial mean of the sampling distribution
mu_old=left_bound+(right_bound-left_bound)*rand(d,1);
% initial variance of the sampling distribution
var_old=((right_bound-left_bound)/2)^2*ones(d,1);
% calculating the initial mean parameter function value
[eta_x_old,eta_x2_old]=...
    truncated_mean_para_fcn(left_bound,right_bound,mu_old,var_old);

%% RECORDS
cur_best_H=[]; % record all current best objective values found
H=[]; % record all objective values sampled

%% WARM UP PERIOD
% A warm up period is performed in order to get a robust performance.
% Sobol set is used to construct the initial surrogate model.
tic; % count warm up time
fprintf('Warm up begins \n');

sobol_all=sobolset(d);
Lambda=net(sobol_all,warm_up); % Lambda: record all sampled solutions
Lambda=left_bound+(right_bound-left_bound)*Lambda; Lambda=Lambda';
% D: kernel matrix for calculating the weights of the cubic surrogate model
D=zeros(warm_up,warm_up);
for i=1:warm_up-1
    for j=i+1:warm_up
        D(i,j)=norm(Lambda(:,i)-Lambda(:,j))^3;
    end
end
D=D+D';
% weight: the coefficients of the cubic surrogate model
H(1:warm_up)=feval(fcn_name,Lambda);
cur_best_H(1:warm_up)=max(H(1:warm_up));
weight=D\H';
fprintf('Warm up ends \n');
tWarmUp=toc; % count warm up time
fprintf('Warm up takes %8.4f seconds \n',tWarmUp);

%% MAIN LOOP
fprintf('Main loop begins \n');
tic; % count main loop time
k=warm_up; % iteration counter
num_evaluation=warm_up; % budget consumption
while num_evaluation+1<=budget
    %% PROGRESS REPORT
    if mod(k,100)==0
        fprintf('iter: %5d, eval: %5d, cur best: %8.4f, true optimum: %8.4f \n',...
            k,num_evaluation,cur_best_H(k),optimal_objective_value);
    end
    k=k+1;
    
    %% ADAPTIVE HYPERPARAMETERS
    % alpha: learning rate for updating the mean parameter function
    alpha=1/(k-warm_up+ca)^gamma0;
    % t: annealing temperature
    t=.1*abs(cur_best_H(end))/log(k-warm_up+1);
    % numNumInt: number of points for numerical integration
    numNumInt=max(100,floor((k-warm_up)^.5));
    
    %% SAMPLING
    % given the sampling parameter (mu_old,var_old), generate a sample from
    % the truncated independent multivariate normal density.
    distribution_choice=rand();
    if distribution_choice<exploration
        % the exploration part: generate from the uniform distribution
        x_sample=left_bound+(right_bound-left_bound)*rand(d,1);
    else
        x_sample=normt_rnd(mu_old,var_old,left_bound,right_bound);
    end
    Lambda(:,k)=x_sample;
    
    % Record the current best true objective value.
    H(k)=feval(fcn_name,x_sample);
    num_evaluation=num_evaluation+1;
    cur_best_H(k)=max(cur_best_H(k-1),H(k));
    
    %% SURROGATE MODELING
    % given H, find the new surrogate model based on sampled solutions
    % cubic model: Sk(x)=\sum_{i=1}^{k} weight(i)*||x-xi||^3
    temp=zeros(k,k);
    temp(1:k-1,1:k-1)=D;
    temp(end,1:end-1) = vecnorm(Lambda(:,1:k-1)-Lambda(:,k)).^3;
    temp(1:end-1,end) = temp(end,1:end-1)';
    D=temp;
    weight=D\H';
    
    %% SAMPLING PARAMETER UPDATING
    % given the surrogate model (weight),
    % update the sampling parameter (mu, var).
    
    % using numerical integration to find E_g[X] (EX) and E_g[X^2] (EX2)
    % based on the surrogate model.
    % 1. generate numNumInt points from the sampling distribution
    % 2. calculate exp_surrogate_num_int_pts based on these sample points
    x_num_int=...
        normt_rnd(mu_old*ones(1,numNumInt),var_old*ones(1,numNumInt),...
        left_bound,right_bound);
    % exp_surrogate_num_int_pts=[..,exp(\sum_j w || xi-xj ||^3),..]
    % exponential values of surrogate model at numerical integration points
    exp_surrogate_num_int_pts=...
        esnip_fcn(x_num_int,numNumInt,weight,Lambda,...
        t,mu_old,var_old,left_bound,right_bound);
  
    % normalization
    exp_surrogate_num_int_pts=...
        exp_surrogate_num_int_pts/max(exp_surrogate_num_int_pts);
    
    int_exp_surrogate=sum(exp_surrogate_num_int_pts);
    int_exp_surrogate_x=sum(x_num_int.*exp_surrogate_num_int_pts,2);
    int_exp_surrogate_x2=sum(x_num_int.^2.*exp_surrogate_num_int_pts,2);
    
    EX=int_exp_surrogate_x/int_exp_surrogate;
    EX2=int_exp_surrogate_x2/int_exp_surrogate;
    % update the mean parameter function
    eta_x_new=eta_x_old+alpha*(EX-eta_x_old);
    eta_x2_new=eta_x2_old+alpha*(EX2-eta_x2_old);
    
    [mu_new,var_new]=...
        truncated_inv_mean_para_fcn(left_bound,right_bound,...
        mu_old,var_old,eta_x_new,eta_x2_new);

    %% UPDATING
    eta_x_old=eta_x_new;
    eta_x2_old=eta_x2_new;
    
    mu_old=mu_new; 
    var_old=var_new;
    
    %% VISUALIZATION
    if mod(k,10)==0
        plot(cur_best_H,'r-o'); % current best
        hold on
        cur_size=size(cur_best_H);
        optimal_line=optimal_objective_value*ones(cur_size(2),1);
        plot(optimal_line,'k:','LineWidth',5); % true optimal value
        xlabel('Number of function evaluations')
        ylabel('Objective function value')
        title(sprintf('<%s function>   Iteration: %5d  Evaluation: %5d',fcn_name,k,num_evaluation));
        legend('EARS','True optimal value','Location','southeast');
        ylim([cur_best_H(1)*1.1 -0.8]);
        grid on
        drawnow;
    end
end

%% FINAL REPORT
fprintf('iter: %5d, eval: %5d, cur best: %8.4f, true optimum: %8.4f \n',...
    k,num_evaluation,cur_best_H(k),optimal_objective_value);
fprintf('Main loop ends \n');
tMainLoop=toc; % count main loop time
fprintf('Main loop takes %8.4f seconds \n',tMainLoop);