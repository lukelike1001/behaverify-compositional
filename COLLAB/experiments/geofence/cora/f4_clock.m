function f4_clock()
% f4_clock - Gemini's falsifier F4: can CORA verify a node-status property?
%
% The claim under test is that a reachability tool cannot express "the Brake
% node is never active for more than N consecutive ticks" (claim #93, proved
% compositionally with a boundary between 16 and 19 against a true maximum of
% 18). Gemini's F4 says that is false: augment the plant with a clock that
% runs while braking and the property becomes an ordinary bounded-time
% location invariant, which hybrid systems verification does as a matter of
% routine.
%
% If this script returns a bound near 18 ticks, the expressiveness claim is
% dead and we reframe today rather than in a fortnight. The script is written
% to make that outcome EASY to reach, not hard, because the point is to find
% out we are wrong.
%
% State: [x; v; tau] where tau counts ticks spent in the Brake branch.

    addpath(genpath('/home/lnam/VSCode/CORA'));
    here = fileparts(mfilename('fullpath'));
    net = neuralNetwork.readONNXNetwork(fullfile(here, '..', 'cruise.onnx'));
    opts.nn.poly_method = "singh";

    dt = 0.1; fence = 1.0; brake = -1.0;
    steps = 30;
    A = [1 dt 0; 0 1 0; 0 0 1];          % x, v, tau (tau held, set per branch)

    fprintf('F4: can a clock-augmented reach computation bound Brake ticks?\n');
    fprintf('    true maximum consecutive Brake ticks = 18 (measured)\n');
    fprintf('    compositional boundary = between 16 and 19\n\n');

    % start from the operating envelope, clock at zero
    S = conZonotope(zonotope([-0.55; 0; 0], [1.45 0 0; 0 0.8 0; 0 0 0]));
    sets = {S};
    peak = 0;
    t0 = tic;
    for k = 1:steps
        next = {};
        for i = 1:numel(sets)
            for p = split_at(sets{i}, fence)
                next{end+1} = step(p{1}, net, opts, A, dt, fence, brake); %#ok<AGROW>
            end
        end
        % Merge, which is what a reachability tool does to stop fragmentation
        % and which the first run of this script omitted - it then spent forty
        % minutes without finishing. Merging is sound (it over-approximates)
        % and makes the comparison fair rather than flattering to us.
        sets = merge_sets(next, 24);
        for i = 1:numel(sets)
            sup = supremum(interval(sets{i}));
            peak = max(peak, sup(3));
        end
        fprintf('  step %2d: %3d sets, clock bound %6.2f, %5.1fs\n', ...
                k, numel(sets), peak, toc(t0));
    end
    fprintf('  CORA bound on consecutive Brake ticks: %.2f\n', peak);
    fprintf('  true maximum: 18\n');
    if peak < 19
        fprintf('  => F4 CONFIRMED. CORA states and bounds the property.\n');
        fprintf('     The node-status expressiveness claim is DEAD.\n');
    else
        fprintf('  => CORA''s bound is %.0fx the truth - sound but useless.\n', peak/18);
        fprintf('     The claim survives, as a PRECISION result not an\n');
        fprintf('     expressiveness one.\n');
    end
end

function out = merge_sets(sets, cap)
% Enclose sets in their interval hulls once there are too many, keeping the
% ones with a live clock separate from the ones without so the bound is not
% destroyed by merging a braking set into a cruising one.
    if numel(sets) <= cap, out = sets; return; end
    live = {}; dead = {};
    for i = 1:numel(sets)
        sup = supremum(interval(sets{i}));
        if sup(3) > 1e-9, live{end+1} = sets{i}; else, dead{end+1} = sets{i}; end
    end
    out = {};
    if ~isempty(live), out{end+1} = conZonotope(hull_of(live)); end
    if ~isempty(dead), out{end+1} = conZonotope(hull_of(dead)); end
end

function Z = hull_of(group)
% Interval hull of a group of sets, as a zonotope.
    iv = interval(group{1});
    for i = 2:numel(group)
        iv = iv | interval(group{i});
    end
    Z = zonotope(iv);
end


function out = split_at(S, fence)
    iv = interval(S); lo = infimum(iv); hi = supremum(iv);
    if hi(1) <= fence, out = {{S, false}}; return; end
    if lo(1) >= fence, out = {{S, true}};  return; end
    L = S & polytope([1 0 0], fence);
    R = S & polytope([-1 0 0], -fence);
    out = {};
    if ~representsa_(L, 'emptySet', 1e-12), out{end+1} = {L, false}; end
    if ~representsa_(R, 'emptySet', 1e-12), out{end+1} = {R, true};  end
    if isempty(out), out = {{S, false}}; end
end

function S = step(piece, net, opts, A, dt, fence, brake)
% One tick. tau increments while the Brake branch is taken and resets otherwise.
    S = piece{1}; braking = piece{2};
    if braking
        S = A*S + [0; dt*brake; 1];          % clock ticks up
    else
        U = net.evaluate(zonotope(project(S, 1:2)), opts);
        iv = interval(U);
        S = A*S + [0; dt*(infimum(iv)+supremum(iv))/2; 0] ...
              + zonotope([0;0;0], [0; dt*(supremum(iv)-infimum(iv))/2; 0]);
        S = reset_clock(S);                  % clock resets outside the fence
    end
    if ~isa(S, 'conZonotope'), S = conZonotope(S); end
end

function S = reset_clock(S)
% Project the clock to zero: Cruise means the Brake run has ended.
    M = [1 0 0; 0 1 0; 0 0 0];
    S = M*S;
end
