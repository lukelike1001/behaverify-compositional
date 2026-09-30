function geofence_cora(steps, v0bound, gridfile)
% geofence_cora - the geofence NSBT in CORA, as the strongest baseline.
%
% Our own zonotope baseline (COLLAB/experiments/geofence/zonotope_box.py)
% fails at 1.5298 with a branch hull and 1.5861 with guard splitting, against
% a limit of 1.5, while the true closed loop reaches only 1.2622. But that
% baseline re-fits an AXIS-ALIGNED zonotope to each piece at the guard,
% because a plain zonotope is not closed under halfspace intersection.
%
% CORA's constrained zonotopes ARE closed under it. That is the one capability
% our baseline lacked and the only remaining way reachability could win here,
% so this is the experiment that decides whether the pilot's headline stands.
%
% Same dynamics, same dt, same CROWN contracts (read from gridfile) as the
% compositional encoding, so the only difference is the method.

    if nargin < 1, steps = 20; end
    if nargin < 2, v0bound = 0.8; end
    if nargin < 3, gridfile = '/tmp/cora_contracts.csv'; end

    dt = 0.1; fence = 1.0; brake = -1.0; limit = 1.5;
    x0lo = -2.0; x0hi = 0.9;

    A = [1 dt; 0 1];          % x' = x + v dt
    B = [0; dt];              % v' = v + a dt

    cells = readmatrix(gridfile);            % xlo xhi vlo vhi alo ahi
    fprintf('CORA geofence: %d steps, |v0| <= %.2f, %d contracts\n', ...
            steps, v0bound, size(cells,1));

    % initial set as a constrained zonotope
    c0 = [(x0lo+x0hi)/2; 0];
    G0 = [(x0hi-x0lo)/2 0; 0 v0bound];
    sets = {conZonotope(zonotope(c0, G0))};

    worst = x0hi;
    for k = 1:steps
        next = {};
        for i = 1:numel(sets)
            pieces = split_at(sets{i}, fence);
            for p = 1:numel(pieces)
                S = pieces{p};
                iv = interval(S);
                lo = infimum(iv); hi = supremum(iv);
                if lo(1) >= fence - 1e-12
                    alo = brake; ahi = brake;
                else
                    [alo, ahi] = crown_hull(cells, lo(1), hi(1), lo(2), hi(2));
                end
                U = zonotope((alo+ahi)/2, (ahi-alo)/2);
                next{end+1} = A*S + B*U;                             %#ok<AGROW>
            end
        end
        sets = next;
        for i = 1:numel(sets)
            sup = supremum(interval(sets{i}));
            worst = max(worst, sup(1));
        end
        if worst > limit
            fprintf('  VIOLATED at step %d: max x = %.4f (limit %.1f), %d sets\n', ...
                    k, worst, limit, numel(sets));
            fprintf('  => CORA also fails; the pilot headline stands.\n');
            return
        end
        if numel(sets) > 20000
            fprintf('  GAVE UP at step %d: %d sets (fragmentation)\n', k, numel(sets));
            return
        end
    end
    fprintf('  SAFE through %d steps: max x = %.4f (limit %.1f), %d sets\n', ...
            steps, worst, limit, numel(sets));
    fprintf('  => CORA PROVES it. The benchmark does not discriminate and the\n');
    fprintf('     pilot headline must be WITHDRAWN.\n');
end

function pieces = split_at(S, fence)
% Intersect with each side of the guard. Constrained zonotopes are closed
% under this, so no correlation is lost - the thing our own baseline could
% not do.
    iv = interval(S);
    lo = infimum(iv); hi = supremum(iv);
    if hi(1) <= fence || lo(1) >= fence
        pieces = {S}; return
    end
    left  = S & polytope([1 0], fence);      % x <= fence
    right = S & polytope([-1 0], -fence);    % x >= fence
    pieces = {};
    if ~representsa_(left, 'emptySet', 1e-12),  pieces{end+1} = left;  end
    if ~representsa_(right, 'emptySet', 1e-12), pieces{end+1} = right; end
    if isempty(pieces), pieces = {S}; end
end

function [alo, ahi] = crown_hull(cells, xlo, xhi, vlo, vhi)
% Union of the CROWN contracts of every cell the set overlaps - identical to
% the rule the compositional encoding and our own baselines use.
    m = (cells(:,2) >= xlo) & (cells(:,1) <= xhi) & ...
        (cells(:,4) >= vlo) & (cells(:,3) <= vhi);
    if any(m)
        alo = min(cells(m,5)); ahi = max(cells(m,6));
    else
        alo = -1.0; ahi = 1.0;
    end
end
