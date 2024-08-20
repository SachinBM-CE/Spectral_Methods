clc; clear all; close all;

k = 1; % Parameter in u(x)

x_max = 2*pi; x_min = 0;
x_an_step = 0.01;
x_an = x_min:x_an_step:x_max;

% Function handle for u(x) and its derivative

% anSoln_handle = @(X, K) exp(K*sin(X));
% du_exact_handle = @(X, K) K*cos(X) .* anSoln_handle(X, K);

% anSoln_handle = @(X, K) (cos(K*X));
% du_exact_handle = @(X, K) (-K*sin(K*X));

anSoln_handle = @(X,K) (K*X);
du_exact_handle = @(X,K) (0.*anSoln_handle(X,K)+1);

% Threshold for minimum error
threshold = 1e-5;

N = 2; % Starting with a small N
min_error_rel = inf;
min_error_Linf = inf;
min_error_L2 = inf;

function [D, x] = Odd_Diff_Matrix(N, x_max)
    % Odd method
    j = 0:1:N;
    x = (x_max/(N+1))*j;
    [I, J] = ndgrid(j, j); % grid of indices
    D = ((-1).^(I + J) ./ 2) .* (sin((I - J) .* pi ./ (N + 1))).^(-1);
    D(1:(size(D,1)+1):end) = 0; % Set diagonal elements to zero
end

function [D, x] = Even_Diff_Matrix(N, x_max)
    % Even method
    j = 0:1:N-1;
    x = (x_max/N)*j;
    [I, J] = ndgrid(j, j); % grid of indices
    [XI, XJ] = ndgrid(x, x); % coordinates of grid
    D = ((-1).^(I + J) ./ 2) .* cot((XI - XJ)/2);
    D(1:(size(D,1)+1):end) = 0; % Set diagonal elements to zero
end

% Arrays to store N and min_error for plotting evolution
N_evolution = [];
min_error_rel_evolution = [];
min_error_Linf_evolution = [];
min_error_L2_evolution = [];

while min_error_rel >= threshold

    N = N + 2; % Increment N by 2 since it should be an even number

    % [D, x] = Odd_Diff_Matrix(N, x_max);
    [D, x] = Even_Diff_Matrix(N, x_max);

    % Compute derivative using Fourier differentiation matrix
    u = anSoln_handle(x,k);
    du_approx = D * u';

    du_real = du_exact_handle(x,k);

    % Compute relative pointwise error
    error_rel = abs((du_real' - du_approx) ./ du_real');
    error_Linf = norm((du_real' - du_approx), Inf);
    error_L2 = norm((du_real' - du_approx), 2);

    % Measure minimum relative pointwise error
    min_error_rel = min(error_rel);
    min_error_Linf = min(error_Linf);
    min_error_L2 = min(error_L2);

    % Store N and min_error for plotting evolution
    N_evolution(end+1) = N;
    min_error_rel_evolution(end+1) = min_error_rel;
    min_error_Linf_evolution(end+1) = min_error_Linf;
    min_error_L2_evolution(end+1) = min_error_L2;

    % Plot

    subplot(2,2,1)
    plot(x_an, du_exact_handle(x_an,k), 'b', x, du_approx, 'r-');
    xlabel('x');
    ylabel('Derivative');
    title(['Comparison between Exact and Approximate Derivative for N = ', num2str(N), ', k = ', num2str(k)]);
    legend('Exact', 'Approximate');
    drawnow;

    subplot(2,2,2)
    semilogy(N_evolution, min_error_rel_evolution, 'r-');
    hold on;
    xlabel('N');
    ylabel('Minimum Relative Error');
    title(['Evolution of Minimum Relative Error with N for k = ', num2str(k)]);
    drawnow;

    subplot(2,2,3)
    semilogy(N_evolution, min_error_Linf_evolution, 'b-');
    hold on;
    xlabel('N');
    ylabel('Minimum L_{\infty} Error');
    title(['Evolution of Minimum L_{\infty} Error with N for k = ', num2str(k)]);
    drawnow;

    subplot(2,2,4)
    semilogy(N_evolution, min_error_L2_evolution, 'k-');
    hold on;
    xlabel('N');
    ylabel('Minimum L2 Error');
    title(['Evolution of Minimum L2 Error with N for k = ', num2str(k)]);
    drawnow;
end

fprintf('Minimum N value needed for error < 10^-5: %d\n', N);