clc; clear all; close all;

k = 6; % Parameter in u(x)

x_max = 2*pi; x_min = 0; 
x_an_step = 0.01;
x_an = x_min:x_an_step:x_max;

% Function handle for u(x) and its derivative

anSoln_handle = @(X, K) exp(K*sin(X));
du_exact_handle = @(X, K) K*cos(X) .* anSoln_handle(X, K);

% Threshold for minimum error
threshold = 1e-5;

N = 2; % Start with a small N
min_error = inf;

% Arrays to store N and min_error for plotting evolution
N_evolution = [];
min_error_evolution = [];

while min_error >= threshold

    N = N + 2; % Increment N by 2 since it should be an even number

    % odd method
    j = 0:1:N;
    x = (x_max/(N+1))*j;
    [I, J] = ndgrid(j, j); % grid of indices
    D = ((-1).^(I + J) ./ 2) .* (sin((I - J) .* pi ./ (N + 1))).^(-1);
    D(1:(size(D,1)+1):end) = 0; % Set diagonal elements to zero

    % Compute derivative using Fourier differentiation matrix
    u = anSoln_handle(x, k);
    du_approx = D * u';
    du_real = du_exact_handle(x,k);

    % Compute relative pointwise error
    error = abs((du_real' - du_approx) ./ du_real');

    % Measure minimum relative pointwise error
    min_error = min(error);

    % Store N and min_error for plotting evolution
    N_evolution(end+1) = N;
    min_error_evolution(end+1) = min_error;

    % Plot
    subplot(2,1,1)
    plot(x_an, du_exact_handle(x_an,k), 'b', x, du_approx, 'r-');
    xlabel('x');
    ylabel('Derivative');
    title(['Exact vs Approximate Derivative for N = ', num2str(N), ', k = ', num2str(k)]);
    legend('Exact', 'Approximate');
    drawnow;

    subplot(2,1,2)
    semilogy(N_evolution, min_error_evolution, 'r-');
    hold on;
    xlabel('N');
    ylabel('Minimum Relative Error');
    title(['Minimum Relative Error vs N for k = ', num2str(k)]);
    drawnow;
end

fprintf('Minimum N value needed for error < 10^-5: %d\n', N);