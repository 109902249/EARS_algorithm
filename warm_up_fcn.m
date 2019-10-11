function [H,Lambda,D,best_H]=warm_up_fcn(d,warm_up,left_bound,right_bound,fcn_name)
%--------------------------------------------------------------------------
% 'warm_up_fcn'
% performs the warm up period by choosing the Sobol sequence as the initial
% sample points
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% H           : objective function values
% Lambda      : sampled solutions
% D           : distance matrix of sampled solutions
% best_H      : best objective values found
%
% Input arguments
% ---------------
% d           : dimension of the search region
% warm_up     : number of function evaluations used for warm up 
% left_bound  : left bound of the search region
% right_bound : right bound of the search region
% fcn_name    : test function
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

% sobol initial points
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

H(1:warm_up)=feval(fcn_name,Lambda); % objective function value
best_H(1:warm_up)=max(H(1:warm_up)); % current best