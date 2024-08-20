function burgers_solver
    % Parameters
    c = 4.0;
    nu = 0.1;
    L = 2 * pi;
    T_final = pi / 4;
    CFL_values = 0.1:0.1:5.0;
    N_values = [16, 32, 48, 64, 96, 128, 192, 256];

    % Results storage
    results = zeros(length(N_values), 3);

    for i = 1:length(N_values)
        N = N_values(i);
        x = linspace(0, L, N+1);
        x = x(1:end-1); % periodic domain
        u0 = sin(x); % example initial condition
        uhat0 = fft(u0);
        k = [0:N/2-1 0 -N/2+1:-1]' * (2*pi/L); % wave numbers

        kmax = N / 2;
        stable_CFL = 0.1;

        for CFL = CFL_values
            dt = compute_dt(CFL, u0, kmax, nu);
            u_approx = time_evolution(uhat0, N, dt, T_final, nu, k);
            u_exact = exact_solution(x, T_final, c, nu);
            error = compute_error(u_approx, u_exact);

            if error < 1e-3
                stable_CFL = CFL;
            else
                break;
            end
        end

        results(i, :) = [N, stable_CFL, error];
        fprintf('N = %d, Stable CFL = %.1f, Error = %.5e\n', N, stable_CFL, error);
    end

    % Output the results
    disp('Results (N, Stable CFL, Error):');
    disp(results);
end

function dt = compute_dt(CFL, u, kmax, nu)
    umax = max(abs(u));
    dt = CFL / (umax * kmax + nu * kmax^2);
end

function u = time_evolution(uhat0, N, dt, T, nu, k)
    uhat = uhat0;
    t = 0;

    while t < T
        if t + dt > T
            dt = T - t;
        end
        uhat = runge_kutta_step(uhat, dt, N, nu, k);
        t = t + dt;
    end

    u = real(ifft(uhat));
end

function uhat_new = runge_kutta_step(uhat, dt, N, nu, k)
    k1 = compute_rhs(uhat, N, nu, k);
    k2 = compute_rhs(uhat + 0.5 * dt * k1, N, nu, k);
    k3 = compute_rhs(uhat + 0.5 * dt * k2, N, nu, k);
    k4 = compute_rhs(uhat + dt * k3, N, nu, k);
    uhat_new = uhat + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
end

function rhs = compute_rhs(uhat, N, nu, k)
    u = real(ifft(uhat));
    ux = real(ifft(1i * k .* uhat));
    uux = fft(u .* ux);
    rhs = -uux + nu * (1i * k).^2 .* uhat;
end

function u_exact = exact_solution(x, t, c, nu)
    a = x - c * t;
    b = t + 1;
    phi_xb = arrayfun(@(ax) phi(ax, b, nu), a);
    dphi_dx = arrayfun(@(ax) dphi(ax, b, nu), a);
    u_exact = c - 2 * nu * dphi_dx ./ phi_xb;
end

function result = phi(a, b, nu)
    k = -10:10;
    result = sum(exp(-(a - (2*k + 1) * pi).^2 / (4 * nu * b)));
end

function result = dphi(a, b, nu)
    k = -10:10;
    result = -2 * sum(exp(-(a - (2*k + 1) * pi).^2 / (4 * nu * b)) .* (a - (2*k + 1) * pi) / (4 * nu * b));
end

function error = compute_error(u_approx, u_exact)
    error = max(abs(u_approx - u_exact));
end
