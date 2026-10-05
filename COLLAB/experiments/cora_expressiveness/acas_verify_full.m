function acas_verify_full()
% CORA's nn.verify against its own VCAS domination test, on the same regions.
%
% The VCAS reachability loop decides an advisory only when one class dominates
% the whole output set, and otherwise prints "Unable to determine maximum!" and
% stops. I spent most of a night treating that as evidence that CORA cannot
% branch on an ambiguous classification.
%
% It is not. nn/@neuralNetwork/verify.m is CORA's automated NN verification
% entry point - A*y <= b with adaptive INPUT splitting, from VNN-COMP'24 - and
% argmax is exactly that specification: advisory k wins iff y_j - y_k <= 0 for
% every other j. Refining the input sidesteps the preimage problem entirely.
%
% This measures both on identical regions so the comparison is direct.

    addpath(genpath('/home/lnam/VSCode/CORA'));
    base = ['/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/' ...
            '2025_NEUS/examples/AcasXu/networks/'];
    net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], ...
                                        false, 'BSSC');
    rho_mean = 19791.091; rho_range = 60621;
    opts.nn.poly_method = "singh";
    fprintf('ACAS Xu ACASXU_run2a_1_1 | VCAS domination test vs nn.verify | %s\n', ...
            datestr(now, 31));
    fprintf('uncertainty in range only; bearings and speeds held at points\n\n');
    fprintf('%9s %10s %14s %12s %9s\n', 'rho ft', 'radius ft', ...
            'VCAS loop', 'nn.verify', 'sec');
    agree = 0; rescued = 0; total = 0;
    for rho_ft = [8000 12000 20000 28000 36000]
        for radius_ft = [500 2000 5000 10000]
            c = [(rho_ft - rho_mean) / rho_range; 0; 0; 0.3; 0.3];
            r = [radius_ft / rho_range; 0; 0; 0; 0];
            iv = interval(net.evaluate(zonotope(c, diag(r)), opts));
            [dec, w] = dominates(iv);
            y = net.evaluate(c); [~, k] = max(y);
            o = [1:k-1, k+1:5]; A = zeros(4, 5); b = zeros(4, 1);
            for i = 1:4, A(i, o(i)) = 1; A(i, k) = -1; end
            t0 = tic;
            try
                res = net.verify(c, r, A, b, true, struct(), 120, false);
                verdict = res.str;
            catch err
                verdict = ['ERR:' err.identifier];
            end
            secs = toc(t0);
            total = total + 1;
            if dec, vc = sprintf('adv %d', w); else, vc = 'AMBIGUOUS'; end
            if dec && strcmp(verdict, 'VERIFIED'), agree = agree + 1; end
            if ~dec && strcmp(verdict, 'VERIFIED'), rescued = rescued + 1; end
            fprintf('%9d %10d %14s %12s %9.2f\n', rho_ft, radius_ft, vc, ...
                    verdict, secs);
        end
    end
    fprintf('\n%d of %d regions: VCAS loop gave up, nn.verify PROVED anyway\n', ...
            rescued, total);
    fprintf('%d of %d: both decided\n', agree, total);
    fprintf(['\nAny nonzero rescue count means CORA resolves classification by\n' ...
             'refining the INPUT, and "CORA cannot branch on an ambiguous\n' ...
             'argmax" is false.\n']);
end

function [tf, k] = dominates(iv)
    tf = false; k = 0;
    for i = 1:length(iv)
        o = [1:i-1, i+1:length(iv)];
        if all(supremum(iv(o)) <= infimum(iv(i))), tf = true; k = i; return; end
    end
end
