function [PL_1D, PL_2D, x_hat] = calculateBayesianPL(gmm, TIR)
    L = length(gmm);
    x_hat = zeros(2, 1);
    for l = 1:L
        x_hat = x_hat + gmm(l).w * gmm(l).mu;
    end
    PL_1D = zeros(2, 1);
    % --- The Q-function (tail probability of standard normal distribution)
    % Use erfc for better numerical precision with small probabilities
    q_func = @(z) 0.5 * erfc(z / sqrt(2));
    
    % --- Loop over the two orthogonal axes (i=1 and i=2)
    for i = 1:2
        objective_func = @(PL) ...
            sum(arrayfun(@(l) gmm(l).w * ( ...
                (1 - q_func((x_hat(i) - PL - gmm(l).mu(i)) / sqrt(gmm(l).Sigma(i,i)))) + ...
                q_func((x_hat(i) + PL - gmm(l).mu(i)) / sqrt(gmm(l).Sigma(i,i))) ...
            ), 1:L)) - TIR;
    
        % Solve for PL_1D,i using bisection search
        PL_1D(i) = bisection_search(objective_func);
    end
    
    % --- Calculate 2D Protection Level using eq. (89)
    PL_2D = norm(PL_1D);
    end
    
    % --- Nested Bisection Search Function ---
    function root = bisection_search(func)
        % Standard bisection method to find the root of func(x) = 0.
        a = 0;          % Lower bound for PL (must be non-negative)
        b = 10;         % Initial guess for upper bound
        tol = 1e-9;     % Tolerance for the result
        max_iter = 1000;% Maximum number of iterations
        
        % --- Find a valid upper bound `b` such that func(b) < 0
        % The function is monotonically decreasing, and func(0) > 0 for TIR < 1.
        iter_b = 0;
        while func(b) > 0
            b = b * 2;
            iter_b = iter_b + 1;
            if iter_b > 50 % Prevent infinite loop if something is wrong
                error('Could not find an upper bound for the bisection search.');
            end
        end
        
        % --- Bisection loop
        for k = 1:max_iter
            c = a + (b-a)/2; % Midpoint
            f_c = func(c);
            
            % Check for convergence
            if abs(f_c) < tol || (b-a)/2 < tol
                root = c;
                return;
            end
            
            % Update bounds
            if sign(f_c) == sign(func(a))
                a = c;
            else
                b = c;
            end
        end
        
        % If max_iter is reached, return the best estimate
        root = c;
    end