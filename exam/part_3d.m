clc; clear; close all;

% Scalar hyperbolic problem parameters
T = 5; % Final time
dt = 1e-6; % Time step

% Function to compute init solution
u_init = @(x) cos(5*pi*x);

% Function to create Chebyshev differentiation matrix
function D = chebyshev_diff_matrix(N)
    theta = pi * (0:N) / N;
    x = cos(theta)';
    c = [2; ones(N-1,1); 2] .* (-1).^(0:N)';
    X = repmat(x, 1, N+1);
    dX = X - X';
    D = (c * (1./c)') ./ (dX + (eye(N+1))); % off-diagonal entries
    D = diag(sum(D,2)) - D; % diagonal entries
end

% Function to advance the solution in time using RK-4 method
function u = runge4(u,dt,x,F,D)
    u1 = u + 0.5*dt*F;
    u2 = u + 0.5*dt*(-x*D*u1);
    u3 = u + (dt*(-x*D*u2));
    u = (-u + u1 + (2*u2) + u3 + (0.5*dt*(-x*D*u3)))/3;
end

% Grid sizes
N_values = [16, 32, 64, 128, 256, 512];

% Initialize array to store error
errors_Linf_cheby = zeros(size(N_values));

% Loop over different grid sizes
for i = 1:length(N_values)
    
    N = N_values(i);
    fprintf('Current grid size N = %d\n', N);

    theta = pi * (0:N) / N;
    x = cos(theta);
    
    u0 = u_init(x);
    u_cheby = u0';
    
    % Compute numerical solution using RK-4 with second order spatial derivative
    D = chebyshev_diff_matrix(N);
    
    for t = dt:((N^2)*dt):T
        u_cheby = runge4(u_cheby, dt, x, (-x*D*u_cheby), D);
    end

    u_real = u_init(x)';

    % Compute L∞-error
    errors_Linf_cheby(i) = norm(u_cheby - u_real, inf);

    fprintf('   Inifinity Norm (Chebyshev) = %d\n', errors_Linf_cheby(i));

    figure(N)
    plot(-1:0.01:1, u_init(-1:0.01:1), 'b', x', u_cheby, '--r');
    legend('exact', 'Chebyshev');
    xlabel('Solution');
    ylabel('Space Coordinate');
    title(sprintf('Solution Comparison for N = %d', N));

    % Save the plot with the required filename
    saveas(gcf, sprintf('3d_%d.png', N));

end

% Calculate the rate of convergence
logN = log(N_values);
logErrors = log(errors_Linf_cheby);

% Perform linear regression on the log-log data
p = polyfit(logN, logErrors, 1);
rate_of_convergence = p(1);

% Display the rate of convergence
fprintf('Rate of convergence: %f\n', rate_of_convergence);

% Plotting the error convergence
figure;
loglog(N_values, errors_Linf_cheby, '-o');
xlabel('N');
ylabel('L∞-error');
title('Convergence of Chebyshev Collocation Method');
grid on;

% Display the fitted line for reference
hold on;
loglog(N_values, exp(p(2)) * N_values.^rate_of_convergence, '--');
legend('Error', sprintf('Fit: O(N^{%.2f})', rate_of_convergence));
hold off;

% Save the convergence plot
saveas(gcf, '3d_convergence.png');