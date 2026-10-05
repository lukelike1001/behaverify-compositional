function gridworld_cora()
% gridworld_cora - can CORA express the PUBLISHED grid world NSBT?
%
% Luke's inversion of three weeks of failed attempts: instead of inventing a
% benchmark CORA cannot do, test CORA on the benchmarks that already exist and
% are published. Grid world is from NeuS 2025 and is reused in 2026_TBA.
%
% THE MECHANISM UNDER TEST, taken verbatim from CORA'S OWN ARCH-COMP VCAS
% benchmark (examples/contDynamics/neurNetContrSys/example_neuralNet_reach_07_VCAS.m,
% lines 199-215). That file is the closest thing CORA has to a classification-
% driven mode-switching benchmark, and this is how its REACHABILITY section
% decides which class the network selected:
%
%     logits = interval(logits);
%     adv_next = -1;
%     for k = 1:9
%         k_ = [1:k-1, k+1:9];
%         if all(supremum(logits(k_)) <= infimum(logits(k)))
%             adv_next = k; break;
%         end
%     end
%     if adv_next == -1
%         disp("Unable to determine maximum!")
%         break;
%     end
%
% It requires ONE class to dominate ALL others across the WHOLE reachable set.
% If no class dominates it abandons the computation. There is a single Ri and a
% single adv_cur; nothing carries "class A on this subset, class B on that one".
%
% So the question this script asks is narrow and checkable: on the published
% grid world network, over sets of states a real drone actually occupies, how
% often is the argmax unambiguous? Every ambiguous set is one CORA's own method
% cannot continue past.

    addpath(genpath('/home/lnam/VSCode/CORA'));
    net = neuralNetwork.readONNXNetwork(['/home/lnam/VSCode/behaverify-compositional/' ...
        'REPRODUCIBILITY/2026_TBA/examples/grid_world/networks/1000__6_18_0__0200_1.onnx']);
    opts.nn.poly_method = "singh";
    fprintf('grid world | 1000__6_18_0__0200_1.onnx | 7x7, 5 actions\n');
    display(net);

    % A point query first: the network must work at all before set queries mean
    % anything. Drone at (3,3), goal at (5,5).
    y = net.evaluate([3; 3; 5; 5]);
    [~, k] = max(y);
    fprintf('point (3,3)->(5,5): argmax = class %d, logits %s\n\n', k, mat2str(y', 3));

    % Now the real question. A reachable set is never a single lattice point;
    % it is a REGION. Sweep sets of increasing width and ask CORA's own
    % domination test whether the argmax is decidable over each.
    fprintf('%8s %10s %10s %12s\n', 'radius', 'decidable', 'ambiguous', 'rate');
    for r = [0.0, 0.1, 0.25, 0.5, 1.0]
        decidable = 0; ambiguous = 0;
        for dx = 1:5
            for dy = 1:5
                for gx = [1 4 6]
                    for gy = [1 4 6]
                        c = [dx; dy; gx; gy];
                        if r == 0
                            S = zonotope(c, zeros(4,1));
                        else
                            S = zonotope(c, r * [eye(2); zeros(2,2)]);
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
        end
        total = decidable + ambiguous;
        fprintf('%8.2f %10d %10d %11.1f%%\n', r, decidable, ambiguous, ...
                100 * ambiguous / total);
    end
    fprintf(['\nEvery ambiguous row is a set on which CORA''s own VCAS method\n' ...
             'prints "Unable to determine maximum!" and stops.\n']);
end

function tf = dominates(out)
% CORA's VCAS test, verbatim: does one class dominate all others?
    n = length(out);
    tf = false;
    for k = 1:n
        others = [1:k-1, k+1:n];
        if all(supremum(out(others)) <= infimum(out(k)))
            tf = true; return
        end
    end
end
