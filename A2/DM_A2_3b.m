% Compare performance of 2nd order and infinite order schemes for long time integration
clc; clear; close all;

% Scalar hyperbolic problem parameters
T = 200; % Final time
dt = 1e-3; % Time step

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
    
    % Create the sparse matrix with periodic boundary conditions
    A = spdiags([off_diag_m1, main_diag, off_diag_p1], [-1, 0, 1], N+1, N+1);

    % Periodic boundary conditions
    A(1, end) = d_m1;
    A(end, 1) = d_p1;
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

function plot_sols(x1, u1, leg1, x2, u2, leg2, x3, u3, leg3, time)
    figure
    plot(x1', u1, 'b', x2', u2, '+m', x3, u3, '*r')
    legend(leg1, leg2, leg3);
    title(['Solutions at t = ', num2str(time)]);
    ylabel('Solution');
    xlabel('Space Coordinate');
end

T_values = [0, pi, 100, 200];
dx = zeros(size(T_values));
cfl = zeros(size(T_values));

N_exact = 500;
N_2nd_order = 200; % Grid size for second order scheme
N_inf_order = 10; % Grid size for infinite order scheme

num_points_exact = 0:1:N_exact;
num_points_2nd = 0:1:N_2nd_order;
num_points_inf = 0:1:N_inf_order;

x_exact = ((2*pi)/(N_exact+1)) * num_points_exact;
x_2nd_order = ((2*pi)/(N_2nd_order+1)) * num_points_2nd;
x_inf_order = ((2*pi)/(N_inf_order+1)) * num_points_inf;

u_real_0 = u_exact(x_exact, 0);
% u_real_100 = u_exact(x_exact, 100);
% u_real_200 = u_exact(x_exact, 200);

u0_2nd = u_init(x_2nd_order);
u_2nd_order = u0_2nd';

u0_inf = u_init(x_inf_order);
u_inf_order = u0_inf';

% Compute numerical solution using 4th order Runge-Kutta method with second order spatial derivative
A2 = second_diff_matrix_odd(N_2nd_order);
D = infinite_diff_matrix_odd(N_inf_order);

plot_sols(x_exact, u_exact(x_exact,0), 'exact', ...
            x_2nd_order, u_2nd_order, '2nd', ...
            x_inf_order, u_inf_order, 'inf', 0);

% Loop over different grid sizes
for t = dt:dt:T
    
    if mod(t,100) == 0
        fprintf('t = %.0f seconds ... \n', t);
        plot_sols(x_exact, u_exact(x_exact,t), 'exact', ...
            x_2nd_order, u_2nd_order, '2nd', ...
            x_inf_order, u_inf_order, 'inf', t);
    end
    
    u_2nd_order = runge4(u_2nd_order, dt, (-2*pi*A2*u_2nd_order), A2);
    u_inf_order = runge4(u_inf_order, dt, (-2*pi*D*u_inf_order), D);

end