function test_adaptive_branching()
% test_adaptive_branching - Tests Path 9: Adaptive set splitting on ACAS Xu
%
% Can a user verify an NSBT by splitting uncertain sets whenever
% they encounter an advisory boundary, branching per unambiguous piece,
% and continuing forward reachability?

    addpath(genpath('/home/lnam/VSCode/CORA'));
    base = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/AcasXu/networks/';
    net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], false, 'BSSC');
    opts.nn.poly_method = "singh";
    rho_mean = 19791.091; rho_range = 60621;

    fprintf('=================================================================\n');
    fprintf('Path 9: Adaptive Set Splitting & Branching on ACAS Xu (%s)\n', datestr(now, 31));
    fprintf('=================================================================\n\n');

    % Initial set spanning the decision boundary between adv 5 and adv 2
    % Center: rho = 14000 ft, theta = 0, psi = 0, v_own = 0.3, v_int = 0.3
    % Radius: rho_rad = 200 ft
    c0 = [(14000 - rho_mean)/rho_range; 0; 0; 0.3; 0.3];
    r0 = [200/rho_range; 0; 0; 0; 0];
    
    dt = 1.0; % 1 second update
    v_own_fps = 0.3 * 1100; % ft/s (approx scaled)
    v_int_fps = 0.3 * 1100;
    turn_rates = [0, 1.5, -1.5, 3.0, -3.0] * (pi/180); % rad/s for 5 advisories

    % Initial queue of sets at t=0: struct with center c, radius r, time t
    queue = struct('c', c0, 'r', r0, 'adv', 0);
    max_steps = 5;
    max_splits = 20;

    for t = 0:max_steps-1
        fprintf('--- Step t = %d: %d set(s) in queue ---\n', t, length(queue));
        next_queue = struct('c', {}, 'r', {}, 'adv', {});
        t_step_start = tic;
        
        total_evals = 0;
        splits_this_step = 0;
        
        while ~isempty(queue)
            curr = queue(1);
            queue(1) = [];
            
            % Evaluate network over set curr
            S = zonotope(curr.c, diag(curr.r));
            out = interval(net.evaluate(S, opts));
            [dec, winner] = dominates(out);
            total_evals = total_evals + 1;
            
            if dec
                % Unambiguous! Advance forward under winner advisory
                curr.adv = winner;
                % Discrete plant update (simplified 1D range closure for test)
                % rho_dot = -v_own - v_int * cos(theta)
                omega = turn_rates(winner);
                c_next = curr.c;
                r_next = curr.r;
                
                % Delta range in normalised units: closure speed ~ 400 ft/s
                d_rho = (-400 * dt) / rho_range;
                c_next(1) = c_next(1) + d_rho;
                % Uncertainty grows slightly due to plant dynamics
                r_next(1) = r_next(1) * 1.05;
                
                next_queue(end+1) = struct('c', c_next, 'r', r_next, 'adv', winner);
            else
                % Ambiguous! Split along dimension of highest uncertainty (rho)
                if splits_this_step >= max_splits
                    fprintf('  [WARN] Hit max_splits (%d) at step t=%d!\n', max_splits, t);
                    % Cannot split further
                    next_queue(end+1) = curr;
                    continue;
                end
                
                splits_this_step = splits_this_step + 1;
                % Bisect along range
                c_left = curr.c; c_right = curr.c;
                r_half = curr.r;
                r_half(1) = curr.r(1) / 2;
                c_left(1) = curr.c(1) - r_half(1);
                c_right(1) = curr.c(1) + r_half(1);
                
                queue(end+1) = struct('c', c_left, 'r', r_half, 'adv', 0);
                queue(end+1) = struct('c', c_right, 'r', r_half, 'adv', 0);
            end
        end
        
        step_time = toc(t_step_start);
        adv_dist = [next_queue.adv];
        fprintf('  Step %d finished in %.3f s | evals: %d | splits: %d | active sets: %d\n', ...
                t, step_time, total_evals, splits_this_step, length(next_queue));
        fprintf('  Advisories chosen: ');
        for a = 1:5
            cnt = sum(adv_dist == a);
            if cnt > 0, fprintf('adv %d: %d  ', a, cnt); end
        end
        fprintf('\n\n');
        queue = next_queue;
    end
    fprintf('Path 9 adaptive branching finished successfully through %d steps!\n', max_steps);
end

function [tf, k] = dominates(out)
    n = length(out); tf = false; k = 0;
    for i = 1:n
        o = [1:i-1, i+1:n];
        if all(supremum(out(o)) <= infimum(out(i))), tf = true; k = i; return; end
    end
end
