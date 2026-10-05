function cone_split2()
% KILL 2, retried with the right primitive.
%
% My first attempt converted the network's output set to a polytope and died
% allocating 81.7 GB. That was a bad choice of representation, not a CORA
% limit: zonotope-to-polytope conversion is exponential in the generator count.
% Constrained zonotopes ARE closed under halfspace intersection and that is
% what the geofence work used throughout, so the cone cut belongs there.
%
% THE QUESTION, and I am trying to make the answer YES. If intersecting the
% output set with each argmax cone yields usable per-advisory branches, then
% CORA can branch on an ambiguous classification, and the kill condition dies
% along with four earlier expressiveness claims.
%
% Two things have to work, and they are different:
%   (a) can we tell which advisories are possible?   -- cheap, expected to work
%   (b) can we recover the INPUT subset for each?    -- the actual requirement,
%       because reachability must continue FROM something

    addpath(genpath('/home/lnam/VSCode/CORA'));
    base = ['/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/' ...
            '2025_NEUS/examples/AcasXu/networks/'];
    net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], ...
                                        false, 'BSSC');
    opts.nn.poly_method = "singh";
    rho_mean = 19791.091; rho_range = 60621;

    % An encounter where the advisory is genuinely undecided: 500 ft of range
    % uncertainty, which my sweep found ambiguous 21.5 percent of the time.
    c = [(20000 - rho_mean) / rho_range; 0; 0; 0.3; 0.3];
    S = zonotope(c, [500 / rho_range; 0; 0; 0; 0]);
    out = net.evaluate(S, opts);
    iv = interval(out);
    fprintf('ACAS cone split | rho 20000 +/- 500 ft | %s\n\n', datestr(now, 31));
    for k = 1:5
        fprintf('  advisory %d: [%8.4f, %8.4f]\n', k, infimum(iv(k)), supremum(iv(k)));
    end
    fprintf('  CORA VCAS test decidable: %d\n\n', dominates(iv));

    % (a) which advisories survive the cone cut, done in conZonotope space
    C = conZonotope(zonotope(out));
    fprintf('(a) cone intersection with constrained zonotopes:\n');
    live = [];
    for k = 1:5
        others = [1:k-1, k+1:5];
        piece = C;
        for j = others
            h = zeros(1, 5); h(j) = 1; h(k) = -1;     % y_j - y_k <= 0
            piece = piece & polytope(h, 0);
            if representsa_(piece, 'emptySet', 1e-9), break; end
        end
        if ~representsa_(piece, 'emptySet', 1e-9)
            live(end+1) = k;  %#ok<AGROW>
            fprintf('    advisory %d: POSSIBLE\n', k);
        else
            fprintf('    advisory %d: excluded\n', k);
        end
    end
    fprintf('  %d of 5 advisories possible\n\n', numel(live));

    % (b) the requirement that actually matters: the INPUT subset per advisory
    fprintf('(b) recovering the input subset for each advisory:\n');
    fprintf('    the cut above lives in OUTPUT space (5 logits). Continuing the\n');
    fprintf('    reachability needs the INPUT states that produce each advisory,\n');
    fprintf('    i.e. X_k = X & f^-1(cone_k). Checking what CORA offers:\n');
    has_back = exist(['/home/lnam/VSCode/CORA/contDynamics/@neuralNetwork/' ...
                      'reachBackward.m'], 'file');
    fprintf('    neuralNetwork/reachBackward.m exists: %d\n', has_back > 0);
    fprintf('    linearSys/reachBackward.m exists:     %d\n', ...
            exist('/home/lnam/VSCode/CORA/contDynamics/@linearSys/reachBackward.m', 'file') > 0);
    fprintf(['\n    If only the linear one exists, the cone cut tells us which\n' ...
             '    advisories are POSSIBLE but not WHERE, and branching would have\n' ...
             '    to propagate the full input set under every live advisory -\n' ...
             '    which is unsound in the sense that it violates mutual\n' ...
             '    exclusivity, and explodes as 5^T.\n']);
end

function tf = dominates(iv)
    tf = false;
    for k = 1:length(iv)
        o = [1:k-1, k+1:length(iv)];
        if all(supremum(iv(o)) <= infimum(iv(k))), tf = true; return; end
    end
end
