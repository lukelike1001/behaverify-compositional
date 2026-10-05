function acas_ambiguity()
% acas_ambiguity - the kill condition, posed where the ambiguity is GENUINE.
%
% My earlier grid world measurement was confounded and is withdrawn: those
% networks are trained only on 2,401 integer lattice points, so evaluating over
% a radius-one set integrates where the network is undefined by construction.
% Luke's correction, and the project already knew it - the 2026-03-31 result
% found those nets perfect on integer points while PGD found counterexamples
% BETWEEN the squares.
%
% ACAS Xu is different and that is the whole point. Its inputs are genuinely
% continuous physical quantities - range, bearings, speeds. A reachable set
% straddling an advisory boundary is the system doing its job, not the network
% being out of distribution. So the question is fair here:
%
%   when a reachable set spans an advisory boundary, can CORA's own domination
%   test still decide which advisory the network selects?
%
% The test is CORA's, taken verbatim from their VCAS benchmark: one class must
% dominate all others across the whole set, or the computation stops.

    addpath(genpath('/home/lnam/VSCode/CORA'));
    base = ['/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/' ...
            '2025_NEUS/examples/AcasXu/networks/'];
    % ACAS ONNX declares a 4-D input {1,1,1,5}, so the data format must be given
    net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], ...
                                        false, 'BSSC');
    opts.nn.poly_method = "singh";
    fprintf('ACAS Xu | ACASXU_run2a_1_1 | 5 advisories | %s\n', datestr(now, 31));
    fprintf('inputs normalised as NeuS does: (raw - mean) / range\n\n');

    % NeuS normalisation, from the paper: 50000 ft -> (50000-19791.091)/60621
    rho_mean = 19791.091; rho_range = 60621;
    % Sweep encounters across the range band where advisories actually change,
    % with an uncertainty radius in FEET that a reachable set would carry.
    fprintf('%10s %12s %12s %10s\n', 'radius ft', 'decidable', 'ambiguous', 'rate');
    for radius_ft = [0, 25, 50, 100, 250, 500]
        decidable = 0; ambiguous = 0;
        for rho_ft = 5000:2500:40000
            for theta = [-0.5 0 0.5]
                for psi = [-0.5 0 0.5]
                    c = [(rho_ft - rho_mean) / rho_range; theta; psi; 0.3; 0.3];
                    r = radius_ft / rho_range;
                    if r == 0
                        S = zonotope(c, zeros(5, 1));
                    else
                        S = zonotope(c, [r; 0; 0; 0; 0]);
                    end
                    out = interval(net.evaluate(S, opts));
                    if dominates(out)
                        decidable = decidable + 1;
                    else
                        ambiguous = ambiguous + 1;
                    end
                end
            end
        end
        total = decidable + ambiguous;
        fprintf('%10d %12d %12d %9.1f%%\n', radius_ft, decidable, ambiguous, ...
                100 * ambiguous / total);
    end
    fprintf(['\nEvery ambiguous row is a reachable set on which CORA''s own\n' ...
             'VCAS loop prints "Unable to determine maximum!" and stops.\n' ...
             'Unlike the grid world attempt, these inputs are continuous\n' ...
             'physical quantities and the network is in distribution.\n']);
end

function tf = dominates(out)
% CORA's VCAS test, verbatim: does one advisory dominate all others?
    n = length(out); tf = false;
    for k = 1:n
        o = [1:k-1, k+1:n];
        if all(supremum(out(o)) <= infimum(out(k))), tf = true; return; end
    end
end
