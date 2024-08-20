clc; clear;

function [u_final, x] = burgers_fourier_galerkin(N, Tmax, CFL)
    % Parameters
    c = 4.0;
    nu = 0.1;
    L = 2 * pi;
    k_c = N/2;

    % Discretization
    x = linspace(0, L, N+1);
    x = x(1:end-1);  % Periodic domain, drop the last point
    dx = L / N;
    k = [0:N/2-1, -N/2:-1]';  % Wavenumbers

    % Initial condition (phi at t=1)
    phi = @(a, b) sum(exp(-(a' - (2*(-k_c:k_c)+1) * pi).^2 / (4 * nu * b)), 2);
    u0 = @(x) c - 2 * nu * (gradient(phi(x - c * 0, 1), dx) ./ phi(x - c * 0, 1));
    u = u0(x);

    % Fourier transform of initial condition
    u_hat = fft(u);

    % Time stepping
    t = 0;
    while t < Tmax
        % Calculate maximum time step
        umax = max(abs(ifft(u_hat)));
        kmax = N /  2;
        dt = CFL ./ (umax * kmax + nu * (kmax)^2);
        if t + dt > Tmax
            dt = Tmax - t;  % Adjust final time step
        end

        % RK4 method
        u_hat = RK4_step(u_hat, k, dt, nu);

        % Update time
        t = t + dt;
    end

    % Final solution
    u_final = ifft(u_hat);

end

function u_hat_new = RK4_step(u_hat, k, dt, nu)
    N = length(u_hat);

    % Define nonlinear term in Fourier space
    nonlinear_term = @(u_hat) -1j * k .* fft(ifft(u_hat).^2) / 2;

    % RK4 steps
    k1 = dt * (nonlinear_term(u_hat) - nu * (k.^2 .* u_hat));
    k2 = dt * (nonlinear_term(u_hat + k1/2) - nu * (k.^2 .* (u_hat + k1/2)));
    k3 = dt * (nonlinear_term(u_hat + k2/2) - nu * (k.^2 .* (u_hat + k2/2)));
    k4 = dt * (nonlinear_term(u_hat + k3) - nu * (k.^2 .* (u_hat + k3)));

    % Update solution
    u_hat_new = u_hat + (k1 + 2*k2 + 2*k3 + k4) / 6;
end

Ns = [32, 48, 64, 96, 128, 192, 256];
CFL_values = 0.1:0.1:1;

for i = 1:length(Ns)
    N = Ns(i);
    for CFL = CFL_values
        try
            burgers_fourier_galerkin(N, pi/4, CFL);
            disp(['N = ', num2str(N), ', CFL = ', num2str(CFL), ' is stable']);
        catch
            disp(['N = ', num2str(N), ', CFL = ', num2str(CFL), ' is unstable']);
            break;
        end
    end
end

errors = zeros(length(Ns), 1);

for i = 1:length(Ns)

    N = Ns(i);

    % Parameters
    c = 4.0;
    nu = 0.1;
    L = 2 * pi;
    k_c = N/2;

    % Discretization
    x = linspace(0, L, N+1);
    x = x(1:end-1);  % Periodic domain, drop the last point
    dx = L / N;
    k = [0:N/2-1, -N/2:-1]';  % Wavenumbers

    [u_final, x] = burgers_fourier_galerkin(N, pi/4, 0.4);

    % Exact solution at t = pi/4
    t_exact = pi/4;
    phi_exact = @(a, b) sum(exp(-(a' - (2*(-10:10)+1) * pi).^2 / (4 * nu * b)), 2);
    u_exact = @(x) c - 2 * nu * (gradient(phi_exact(x - c * t_exact, 1+t_exact), dx) ./ phi_exact(x - c * t_exact, 1+t_exact));
    u_exact_values = u_exact(x)';

    % L-infinity error
    errors(i) = max(abs(real(u_final) - u_exact_values'));
    
end

% Plot error vs N
figure;
loglog(Ns, errors, 'bo-', 'LineWidth', 2);
title('L^\infty Error vs N');
xlabel('N');
ylabel('L^\infty Error');
grid on;

% Calculate the rate of convergence
logN = log(Ns);
logErrors = log(errors);

% Perform linear regression on the log-log data
p = polyfit(logN, logErrors, 1);
rate_of_convergence = p(1);

% Display the rate of convergence
fprintf('Rate of convergence: %f\n', rate_of_convergence);

% Plotting the error convergence
figure;
loglog(Ns, errors, '-o');
xlabel('N');
ylabel('L∞-error');
title('Convergence of Chebyshev Collocation Method');
grid on;

% Display the fitted line for reference
hold on;
loglog(Ns, exp(p(2)) * Ns.^rate_of_convergence, '--');
legend('Error', sprintf('Fit: O(N^{%.2f})', rate_of_convergence));
hold off;

% Save the convergence plot
saveas(gcf, '2c_conv.png');