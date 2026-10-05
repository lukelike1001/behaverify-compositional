function acas_verify()
% Does CORA's OWN nn.verify decide the advisory where its VCAS loop cannot?
%
% I spent hours on the VCAS domination test and the cone trick without looking
% at nn/@neuralNetwork/verify.m, which is CORA's automated NN verification
% entry point - "rewritten with efficient splitting", "constraint zonotope
% splitting", from VNN-COMP'24. It proves A*y <= b over an input box and
% SPLITS THE INPUT ADAPTIVELY when the bound is too loose.
%
% Argmax is exactly that specification: advisory k wins iff y_j - y_k <= 0 for
% every j != k. So if nn.verify decides this on regions where the VCAS
% domination test gives up, then CORA can resolve classification after all, by
% refining the INPUT rather than inverting the network - which sidesteps the
% preimage problem entirely, and the "CORA cannot branch" line is dead.
%
% I am trying to make this SUCCEED.

    addpath(genpath('/home/lnam/VSCode/CORA'));
    base = ['/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/' ...
            '2025_NEUS/examples/AcasXu/networks/'];
    net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], ...
                                        false, 'BSSC');
    rho_mean = 19791.091; rho_range = 60621;
    fprintf('ACAS Xu | nn.verify vs the VCAS domination test | %s\n\n', ...
            datestr(now, 31));
    fprintf('%9s %9s %14s %12s %10s\n', 'rho ft', 'radius ft', ...
            'VCAS test', 'nn.verify', 'sec');

    for rho_ft = [10000 20000 30000]
        for radius_ft = [100 500 2000]
            c = [(rho_ft - rho_mean) / rho_range; 0; 0; 0.3; 0.3];
            r = [radius_ft / rho_range; 0; 0; 0; 0];

            % the VCAS domination test, for comparison
            S = zonotope(c, diag(r));
            opts.nn.poly_method = "singh";
            iv = interval(net.evaluate(S, opts));
            [dec, winner] = dominates(iv);

            % the advisory a point query picks, which is what we ask verify to prove
            y = net.evaluate(c);
            [~, k] = max(y);
            A = zeros(4, 5); b = zeros(4, 1);
            others = [1:k-1, k+1:5];
            for i = 1:4
                A(i, others(i)) = 1; A(i, k) = -1;   % y_other - y_k <= 0
            end

            t0 = tic;
            try
                res = net.verify(c, r, A, b, true, struct(), 120, false);
                secs = toc(t0);
                if isempty(res)
                    verdict = 'unknown';
                elseif isstruct(res)
                    verdict = res.str;
                elseif res
                    verdict = 'PROVED';
                else
                    verdict = 'false';
                end
            catch err
                secs = toc(t0);
                verdict = ['ERR:' err.identifier];
            end
            fprintf('%9d %9d %14s %12s %10.2f\n', rho_ft, radius_ft, ...
                    ternary(dec, sprintf('adv %d', winner), 'AMBIGUOUS'), ...
                    verdict, secs);
        end
    end
    fprintf(['\nIf nn.verify PROVES rows the VCAS test calls AMBIGUOUS, then\n' ...
             'CORA resolves classification by refining the INPUT, no preimage\n' ...
             'needed, and the "CORA cannot branch" claim is dead.\n']);
end

function [tf, k] = dominates(iv)
    tf = false; k = 0;
    for i = 1:length(iv)
        o = [1:i-1, i+1:length(iv)];
        if all(supremum(iv(o)) <= infimum(iv(i))), tf = true; k = i; return; end
    end
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
