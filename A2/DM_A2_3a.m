clc; clear;

% Scalar hyperbolic problem parameters
T = pi; % Final time
dt = 1e-4; % Time step

% Grid sizes
N_values = [8, 16, 32, 64, 128, 256, 512, 1024];

% Function to compute init solution
u_init = @(x) exp(sin(x));

% Function to compute exact solution
u_exact = @(x, t) exp(sin(x - 2*pi*t));

% Function for second order difference approximation
function A = second_diff_matrix_odd(N)

    num_points = 0:1:N;
    x = (2*pi/(N+1))*num_points;
    del_x = x(2)-x(1);
    d_m1 = -1/(2*del_x);
    d_p1 = 1/(2*del_x);

    % Create the main diagonal and the off-diagonals
    main_diag = zeros(N+1, 1);
    off_diag_m1 = d_m1 * ones(N+1, 1);
    off_diag_p1 = d_p1 * ones(N+1, 1);
    
    % Apply periodic boundary conditions
    % off_diag_m1(1) = d_m1; % Wrap-around for the first element to the last
    % off_diag_p1(N+1) = d_p1;   % Wrap-around for the last element to the first
    
    % Create the sparse matrix with periodic boundary conditions
    A = spdiags([off_diag_m1, main_diag, off_diag_p1], [-1, 0, 1], N+1, N+1);

    % Periodic boundary conditions
    A(1, end) = d_m1;
    A(end, 1) = d_p1;
end

% Function for fourth order difference approximation
function A = fourth_diff_matrix_odd(N)
    % Define the number of points and corresponding x values
    num_points = 0:1:N;
    x = (2*pi/(N+1)) * num_points;
    del_x = x(2) - x(1); % Uniform grid spacing

    % Coefficients for fourth-order central difference
    d_m2 = 1/(12*del_x);
    d_m1 = -8/(12*del_x);
    d_p1 = 8/(12*del_x);
    d_p2 = -1/(12*del_x);

    % Create the main diagonal and the off-diagonals
    main_diag = zeros(N+1, 1);
    off_diag_m2 = d_m2 * ones(N+1, 1);
    off_diag_m1 = d_m1 * ones(N+1, 1);
    off_diag_p1 = d_p1 * ones(N+1, 1);
    off_diag_p2 = d_p2 * ones(N+1, 1);

    % Create the sparse matrix with periodic boundary conditions
    A = spdiags([off_diag_m2, off_diag_m1, main_diag, off_diag_p1, off_diag_p2], [-2, -1, 0, 1, 2], N+1, N+1);

    % Periodic boundary conditions
    A(1, end-1) = d_m2;
    A(1, end) = d_m1;
    A(2, end) = d_m2;
    A(end, 1) = d_p1;
    A(end, 2) = d_p2;
    A(end-1, 1) = d_p2;
end

% Function for infinite difference approximation
function D = infinite_diff_matrix_odd(N)
    num_points = 0:1:N;
    [i, j] = ndgrid(num_points, num_points);  
    D = ((-1).^(i + j) ./ 2) .* (sin((i - j) .* pi ./ (N+1))).^(-1);  
    D(1:(N+2):end) = 0;
end

% Function to advance the solution in time using 4th order Runge-Kutta method
function u = runge4(u,dt,F,A)
    u1 = u + 0.5*dt*F;
    u2 = u + 0.5*dt*(-2*pi*A*u1);
    u3 = u + (dt*(-2*pi*A*u2));
    u = (-u + u1 + (2*u2) + u3 + (0.5*dt*(-2*pi*A*u3)))/3;
end

% Initialize arrays to store errors
Linf_2nd_order = zeros(size(N_values));
Linf_4th_order = zeros(size(N_values));
Linf_infinite_order = zeros(size(N_values));
dx = zeros(size(N_values));
cfl = zeros(size(N_values));

% Loop over different grid sizes
for i = 1:length(N_values)
    
    N = N_values(i);
    fprintf('Current grid size N = %d\n', N);
    
    num_points = 0:1:N;
    x = ((2*pi)/(N+1))*num_points;
    dx(i) = x(N) - x(N-1);
    cfl(i) = dt/dx(i);
    
    u0 = u_init(x);
    u_second_order = u0';
    u_fourth_order = u0';
    u_infinite_order = u0';
    
    % Compute numerical solution using 4th order Runge-Kutta method with second order spatial derivative
    A2 = second_diff_matrix_odd(N);
    A4 = fourth_diff_matrix_odd(N);
    D = infinite_diff_matrix_odd(N);
    for t = dt:dt:T
        if mod(t,1) == 0
            fprintf('t = %.0f seconds ... \n', t);
        end
        u_second_order = runge4(u_second_order, dt, (-2*pi*A2*u_second_order), A2);
        u_fourth_order = runge4(u_fourth_order, dt, (-2*pi*A4*u_fourth_order), A4);
        u_infinite_order = runge4(u_infinite_order, dt, (-2*pi*D*u_infinite_order), D);
    end

    u_real = u_exact(x, T)';

    % Compute L∞-error for each approximation
    Linf_2nd_order(i) = norm(u_second_order - u_real, inf);
    Linf_4th_order(i) = norm(u_fourth_order - u_real, inf);
    Linf_infinite_order(i) = norm(u_infinite_order - u_real, inf);
    
    fprintf('   Inifinity Norm (2nd Order)      = %d\n', Linf_2nd_order(i));
    fprintf('   Inifinity Norm (4th Order)      = %d\n', Linf_4th_order(i));
    fprintf('   Inifinity Norm (Infinite Order) = %d\n', Linf_infinite_order(i));

end

plot(x', u_real, 'b', ...
    x', u_second_order, '.m', ...
    x', u_fourth_order, '-xg', ...
    x', u_infinite_order, '*y');
legend('exact', '2nd order', '4th order', 'infinite order');
xlabel('Solution');
ylabel('Space Coordinate');

% % Plot L∞-error for different grid sizes
figure;
loglog(N_values, Linf_2nd_order, '-o', 'DisplayName', '2nd Order');
hold on;
loglog(N_values, Linf_4th_order, '-o', 'DisplayName', '4th Order');
loglog(N_values, Linf_infinite_order, '-o', 'DisplayName', 'Infinite Order');
xlabel('Grid Size (N)');
ylabel('L_\infty Error');
title('L_\infty Error vs. Grid Size');
legend('Location', 'best');
hold off;
