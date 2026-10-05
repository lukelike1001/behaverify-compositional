function test_acas_ambiguity_multidim()
% test_acas_ambiguity_multidim - Resolves the 46.6% vs 11.9% disagreement.
%
% Claude tested 1D uncertainty (range only, 135 points) and got 11.9% at 100 ft.
% What happens with:
%   (1) 1D uncertainty (reproducing Claude exactly)
%   (2) Multi-dimensional uncertainty (25m range + bearing/speed uncertainty)
%   (3) Full grid sweep across state space

    addpath(genpath('/home/lnam/VSCode/CORA'));
    base = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/AcasXu/networks/';
    net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], false, 'BSSC');
    opts.nn.poly_method = "singh";
    rho_mean = 19791.091; rho_range = 60621;

    fprintf('=================================================================\n');
    fprintf('ACAS Xu Ambiguity Audit: 1D vs Multi-Dim Uncertainty (%s)\n', datestr(now, 31));
    fprintf('=================================================================\n\n');

    % Experiment 1: Claude's 1D range uncertainty sweep (135 encounters)
    fprintf('--- Exp 1: 1D Range Uncertainty (Claude configuration) ---\n');
    fprintf('%10s %12s %12s %10s\n', 'radius ft', 'decidable', 'ambiguous', 'rate');
    for radius_ft = [0, 25, 50, 82, 100, 250, 500]
        dec = 0; amb = 0;
        for rho_ft = 5000:2500:40000
            for theta = [-0.5, 0, 0.5]
                for psi = [-0.5, 0, 0.5]
                    c = [(rho_ft - rho_mean)/rho_range; theta; psi; 0.3; 0.3];
                    r = [radius_ft / rho_range; 0; 0; 0; 0];
                    S = zonotope(c, diag(r));
                    out = interval(net.evaluate(S, opts));
                    if dominates(out), dec = dec + 1; else, amb = amb + 1; end
                end
            end
        end
        fprintf('%10d %12d %12d %9.1f%%\n', radius_ft, dec, amb, 100 * amb / (dec + amb));
    end

    % Experiment 2: Multi-dim uncertainty at 25 meters (82.02 ft)
    % A real continuous reach set carries uncertainty in theta, psi, and velocities.
    fprintf('\n--- Exp 2: Multi-Dim Uncertainty (25m range + bearing/speed uncertainty) ---\n');
    r_range_25m = (25 * 3.28084) / rho_range; % 25m in feet normalised (~82 ft)
    
    test_configs = {
        '1D only (rho +/- 25m)', [r_range_25m; 0; 0; 0; 0];
        '2D (rho +/- 25m, theta +/- 0.02 rad)', [r_range_25m; 0.02; 0; 0; 0];
        '3D (rho +/- 25m, theta/psi +/- 0.02 rad)', [r_range_25m; 0.02; 0.02; 0; 0];
        '5D (rho +/- 25m, angles +/- 0.02, v +/- 0.01)', [r_range_25m; 0.02; 0.02; 0.01; 0.01];
        '5D (rho +/- 25m, angles +/- 0.05, v +/- 0.02)', [r_range_25m; 0.05; 0.05; 0.02; 0.02];
    };

    fprintf('%46s %12s %12s %10s\n', 'configuration', 'decidable', 'ambiguous', 'rate');
    for i = 1:size(test_configs, 1)
        name = test_configs{i, 1};
        r_vec = test_configs{i, 2};
        dec = 0; amb = 0;
        for rho_ft = 5000:2500:40000
            for theta = [-0.5, 0, 0.5]
                for psi = [-0.5, 0, 0.5]
                    c = [(rho_ft - rho_mean)/rho_range; theta; psi; 0.3; 0.3];
                    S = zonotope(c, diag(r_vec));
                    out = interval(net.evaluate(S, opts));
                    if dominates(out), dec = dec + 1; else, amb = amb + 1; end
                end
            end
        end
        fprintf('%46s %12d %12d %9.1f%%\n', name, dec, amb, 100 * amb / (dec + amb));
    end
end

function tf = dominates(out)
    n = length(out); tf = false;
    for k = 1:n
        o = [1:k-1, k+1:n];
        if all(supremum(out(o)) <= infimum(out(k))), tf = true; return; end
    end
end
