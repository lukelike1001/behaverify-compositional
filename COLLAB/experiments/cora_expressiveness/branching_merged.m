function branching_merged()
% Is the O(B^T) explosion CORA's, or ours for not merging?
%
% Path 9 of the capability matrix - split the ambiguous set, verify per piece,
% step forward per piece - ran 5 ticks on ACAS Xu and went from 1 set to 84,
% which we recorded as exponential. But test_adaptive_branching.m appends with
% next_queue(end+1) and NEVER MERGES. That is precisely the flaw that killed my
% geofence claim: a baseline with no mitigation, reported as a property of the
% method rather than of the implementation.
%
% Every serious reachability tool merges. The standard rule for a mode-switched
% system is to merge sets that share a discrete mode, because from then on they
% evolve under the same dynamics. ACAS has five advisories, so if merging
% works the count should be bounded by FIVE regardless of horizon.
%
% This runs both loops side by side. If merging bounds it, the exponential is
% ours and the claim dies - the sixth expressiveness or cost claim to do so.

    addpath(genpath('/home/lnam/VSCode/CORA'));
    base = ['/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/' ...
            '2025_NEUS/examples/AcasXu/networks/'];
    net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], ...
                                        false, 'BSSC');
    rho_mean = 19791.091; rho_range = 60621;
    opts.nn.poly_method = "singh";

    fprintf('ACAS Xu branching: unmerged vs merged-by-advisory | %s\n\n', ...
            datestr(now, 31));
    for merging = [false true]
        q = {struct('c', [(20000 - rho_mean) / rho_range; 0; 0; 0.3; 0.3], ...
                    'r', [2000 / rho_range; 0; 0; 0; 0], 'adv', 0)};
        label = 'unmerged'; if merging, label = 'merged  '; end
        fprintf('%s:', label);
        for t = 1:8
            nxt = {};
            for i = 1:numel(q)
                cur = q{i};
                iv = interval(net.evaluate(zonotope(cur.c, diag(cur.r)), opts));
                [dec, w] = dominates(iv);
                if dec
                    nxt{end+1} = step_forward(cur, w, rho_range); %#ok<AGROW>
                else
                    % split the widest dimension and retry both halves
                    [~, d] = max(cur.r);
                    hr = cur.r; hr(d) = hr(d) / 2;
                    l = cur; l.r = hr; l.c(d) = cur.c(d) - hr(d);
                    r = cur; r.r = hr; r.c(d) = cur.c(d) + hr(d);
                    nxt{end+1} = l; nxt{end+1} = r; %#ok<AGROW>
                end
            end
            if merging
                nxt = merge_by_advisory(nxt);
            end
            q = nxt;
            fprintf(' %d', numel(q));
            if numel(q) > 2000, fprintf(' (gave up)'); break; end
        end
        fprintf('   <- sets after each of 8 ticks\n');
    end
    fprintf(['\nIf the merged row stays bounded near the number of advisories,\n' ...
             'the explosion was ours for not merging, not CORA''s.\n']);
end

function s = step_forward(cur, adv, rho_range)
% A deliberately crude plant step: the advisory nudges range, uncertainty grows.
    s = cur; s.adv = adv;
    s.c(1) = s.c(1) - 200 / rho_range;
    s.r(1) = s.r(1) * 1.15;
end

function out = merge_by_advisory(sets)
% Merge every set sharing an advisory into one interval hull. Sound (it
% over-approximates) and it is what a mode-switched reachability tool does.
    out = {};
    advs = unique(cellfun(@(s) s.adv, sets));
    for a = advs(:)'
        grp = sets(cellfun(@(s) s.adv == a, sets));
        if isempty(grp), continue; end
        lo = grp{1}.c - grp{1}.r; hi = grp{1}.c + grp{1}.r;
        for i = 2:numel(grp)
            lo = min(lo, grp{i}.c - grp{i}.r);
            hi = max(hi, grp{i}.c + grp{i}.r);
        end
        out{end+1} = struct('c', (lo + hi) / 2, 'r', (hi - lo) / 2, 'adv', a); %#ok<AGROW>
    end
end

function [tf, k] = dominates(iv)
    tf = false; k = 0;
    for i = 1:length(iv)
        o = [1:i-1, i+1:length(iv)];
        if all(supremum(iv(o)) <= infimum(iv(i))), tf = true; k = i; return; end
    end
end
