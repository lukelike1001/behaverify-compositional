function cone_split()
% KILL CONDITION 2: can CORA be made to BRANCH on an ambiguous argmax?
%
% CORA's own VCAS reachability aborts when no class dominates the whole set
% ("Unable to determine maximum!"). The obvious rebuttal a reviewer will make
% is: intersect the OUTPUT set with each argmax cone y_i >= y_j and carry one
% branch per class. This script tries to do exactly that, because if it works
% in an evening then tonight's finding is "generation, not capability" for the
% fifth time and I want to know before building on it.
%
% I am trying to make this SUCCEED. The measurement is only worth something if
% the attempt was honest.

    addpath(genpath('/home/lnam/VSCode/CORA'));
    net = neuralNetwork.readONNXNetwork(['/home/lnam/VSCode/behaverify-compositional/' ...
        'REPRODUCIBILITY/2026_TBA/examples/grid_world/networks/1000__6_18_0__0200_1.onnx']);
    opts.nn.poly_method = "singh";
    fprintf('cone-split attempt | grid world 1000__6_18_0__0200_1 | 5 classes\n\n');

    % A set CORA's own method cannot decide: one grid cell around (3,3).
    S = zonotope([3;3;5;5], [eye(2); zeros(2,2)]);
    out = net.evaluate(S, opts);
    iv = interval(out);
    fprintf('output interval per class:\n');
    for k = 1:5
        fprintf('  class %d: [%7.3f, %7.3f]\n', k, infimum(iv(k)), supremum(iv(k)));
    end
    fprintf('CORA VCAS test decidable: %d\n\n', dominates(iv));

    % THE ATTEMPT. For each class k, intersect the output set with the cone
    % {y : y_k >= y_j for all j != k}. If the pieces are non-empty we know
    % which classes are POSSIBLE; the question is whether we can map each piece
    % back to the INPUT subset that produced it, because that is what branching
    % actually requires.
    fprintf('cone intersection on the OUTPUT set:\n');
    possible = [];
    for k = 1:5
        others = [1:k-1, k+1:5];
        A = zeros(4, 5); b = zeros(4,1);
        for i = 1:4
            A(i, others(i)) = 1; A(i, k) = -1;      % y_j - y_k <= 0
        end
        try
            piece = polytope(out) & polytope(A, b);
            empty = representsa_(piece, 'emptySet', 1e-9);
        catch err
            fprintf('  class %d: FAILED - %s\n', k, err.message);
            continue
        end
        if ~empty
            possible(end+1) = k; %#ok<AGROW>
            fprintf('  class %d: possible\n', k);
        else
            fprintf('  class %d: excluded\n', k);
        end
    end
    fprintf('\n%d of 5 classes possible over this one cell.\n', numel(possible));
    fprintf(['\nTHE PART THAT DECIDES IT: branching needs the INPUT subset for\n' ...
             'each class, not the output piece. CORA''s evaluate() is one-way -\n' ...
             'it maps an input set to an output set and does not retain the\n' ...
             'preimage. Without the preimage there is nothing to continue the\n' ...
             'reachability FROM.\n']);
end

function tf = dominates(iv)
    tf = false;
    for k = 1:length(iv)
        o = [1:k-1, k+1:length(iv)];
        if all(supremum(iv(o)) <= infimum(iv(k))), tf = true; return; end
    end
end
