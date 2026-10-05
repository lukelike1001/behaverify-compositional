addpath(genpath('/home/lnam/VSCode/CORA'));
base_gw = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2026_TBA/examples/grid_world/networks/';
net_gw = neuralNetwork.readONNXNetwork([base_gw '1000__6_18_0__0200_1.onnx']);
sys_gw = net_gw.nonlinearSysDT();
disp(sys_gw);

% Try running reach on sys_gw
params.R0 = zonotope([3; 3; 5; 5], 0.1*eye(4));
params.U = zonotope(zeros(sys_gw.nrOfInputs, 1));
params.tFinal = 1;
options.timeStep = 1;
options.taylorTerms = 2;
options.zonotopeOrder = 5;

try
    fprintf('Attempting reach on sys_gw...\n');
    R = reach(sys_gw, params, options);
    disp('Reach succeeded!');
    disp(R);
catch err
    disp(getReport(err, 'extended'));
end
