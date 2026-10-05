addpath(genpath('/home/lnam/VSCode/CORA'));

fprintf('Testing discrDynamics (@partition, @markovchain)...\n');
% Create a 2D partition: [0, 7] x [0, 7], 7 segments in each dim
P = partition([0 7; 0 7], [7; 7]);
disp(P);

% Try creating a markovchain
T = sparse(49, 49);
mc = markovchain(P, T);
disp(mc);

% Check methods of markovchain
m = methods(mc);
fprintf('markovchain methods:\n');
disp(m);
