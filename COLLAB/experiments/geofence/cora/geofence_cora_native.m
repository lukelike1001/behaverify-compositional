function geofence_cora_native(steps, v0bound, mode)
% geofence_cora_native - the geofence in CORA, with CORA doing the control.
%
% WHY THIS FILE EXISTS. The first CORA script (geofence_cora.m) was refuted.
% It propagated CORA's constrained zonotopes correctly, but at every step it
% called interval(S) to get an axis-aligned box and then looked our CROWN
% contracts up on that box. So the correlation CORA preserves in the set was
% thrown away at the one moment it mattered, and the network file was never
% touched. What that script measured was the loss of a box-lookup contract
% abstraction, not CORA's ability on this system.
%
% This one gives CORA its own machinery. cruise.onnx is imported with
% neuralNetwork.readONNXNetwork and evaluated directly on the reachable set
% with CORA's neural network reachability (Kochdumper et al., the same code
% path CORA runs on the ARCH-COMP AINNCS benchmarks). No CROWN contracts
% appear anywhere on the CORA side.
%
% Three modes, so the ablation is explicit rather than asserted:
%   'box'   interval enclosure to the network - what the refuted script did,
%           minus the contract grid. Isolates the cost of the box alone.
%   'zono'  zonotope enclosure to the network. Exact guard cuts are kept in
%           the propagation; the network sees a zonotope, not a box.
%   'poly'  polynomial zonotope through the network, so the control stays
%           dependent on the state within each step - the tightest thing
%           CORA offers short of a native hybridAutomaton reach().
%
% The horizon runs to completion in every mode and the peak is reported, so
% the answer does not depend on where an early exit happens to fire.

    if nargin < 1, steps = 20; end
    if nargin < 2, v0bound = 0.8; end
    if nargin < 3, mode = 'zono'; end

    dt = 0.1; fence = 1.0; brake = -1.0; limit = 1.5;
    x0lo = -2.0; x0hi = 0.9;
    A = [1 dt; 0 1];
    B = [0; dt];

    here = fileparts(mfilename('fullpath'));
    net = neuralNetwork.readONNXNetwork(fullfile(here, '..', 'cruise.onnx'));
    opts.nn.poly_method = "singh";

    fprintf(['CORA native: %d steps, |v0| <= %.2f, mode %s, ' ...
             'network %s (CORA evaluates it)\n'], ...
            steps, v0bound, mode, 'cruise.onnx');

    sets = {conZonotope(zonotope([(x0lo+x0hi)/2; 0], ...
                                 [(x0hi-x0lo)/2 0; 0 v0bound]))};
    worst = x0hi;
    peak = 1;
    firstBad = 0;
    t0 = tic;
    for k = 1:steps
        next = {};
        for i = 1:numel(sets)
            pieces = split_at(sets{i}, fence);
            for p = 1:numel(pieces)
                next{end+1} = step_once(pieces{p}, net, opts, mode, ...
                                        A, B, fence, brake);           %#ok<AGROW>
            end
        end
        sets = next;
        peak = max(peak, numel(sets));
        for i = 1:numel(sets)
            sup = supremum(interval(sets{i}));
            worst = max(worst, sup(1));
        end
        if worst > limit && firstBad == 0
            firstBad = k;
        end
        if numel(sets) > 20000
            fprintf('  GAVE UP at step %d: %d sets\n', k, numel(sets));
            return
        end
    end
    report(worst, limit, firstBad, peak, steps, toc(t0), mode);
end


function S = step_once(S, net, opts, mode, A, B, fence, brake)
% One tick of the tree, then one tick of the plant, on a single set.
    lo = infimum(interval(S));
    if lo(1) >= fence - 1e-12                     % Brake branch: a = -1
        S = A*S + B*brake;
        return
    end
    switch mode                                   % NeuralCruise branch
        case 'box'
            U = net.evaluate(interval(S), opts);
            S = A*S + B*zonotope(interval(U));
        case 'zono'
            U = net.evaluate(zonotope(S), opts);
            S = A*S + B*zonotope(interval(U));
        case 'poly'
            P = polyZonotope(zonotope(S));
            U = net.evaluate(P, opts);
            S = conZonotope(zonotope(A*P + B*U));
        otherwise
            error('unknown mode %s', mode);
    end
    if ~isa(S, 'conZonotope')
        S = conZonotope(S);
    end
end


function pieces = split_at(S, fence)
% Exact halfspace intersection at the guard. Constrained zonotopes are closed
% under it, so the x-v correlation survives the cut.
    iv = interval(S);
    lo = infimum(iv); hi = supremum(iv);
    if hi(1) <= fence || lo(1) >= fence
        pieces = {S}; return
    end
    left  = S & polytope([1 0], fence);
    right = S & polytope([-1 0], -fence);
    pieces = {};
    if ~representsa_(left, 'emptySet', 1e-12),  pieces{end+1} = left;  end
    if ~representsa_(right, 'emptySet', 1e-12), pieces{end+1} = right; end
    if isempty(pieces), pieces = {S}; end
end


function report(worst, limit, firstBad, peak, steps, secs, mode)
% Say plainly which way the pilot's headline goes.
    if firstBad > 0
        fprintf(['  VIOLATED (first at step %d): peak x = %.4f, limit %.1f, ' ...
                 'true max 1.2622\n'], firstBad, worst, limit);
        fprintf('  %d sets peak, %.1fs, mode %s\n', peak, secs, mode);
        fprintf(['  => CORA fails WITH its own network reachability. The ' ...
                 'refuted claim is rebuilt on a fair baseline.\n']);
    else
        fprintf('  SAFE through %d steps: peak x = %.4f (limit %.1f)\n', ...
                steps, worst, limit);
        fprintf('  %d sets peak, %.1fs, mode %s\n', peak, secs, mode);
        fprintf(['  => CORA PROVES it once it evaluates the network itself. ' ...
                 'The headline is WITHDRAWN.\n']);
    end
end
