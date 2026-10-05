addpath(genpath('/home/lnam/VSCode/CORA'));
base = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/AcasXu/networks/';
net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], false, 'BSSC');

fprintf('Testing net.nonlinearSysDT()...\n');
try
    sys = net.nonlinearSysDT();
    disp(sys);
    
    % Try reach on this sys
    params.R0 = zonotope(zeros(5,1), 0.1*eye(5));
    params.U = zonotope(zeros(sys.nrOfInputs, 1));
    params.tFinal = 1;
    options.timeStep = 1;
    options.taylorTerms = 2;
    options.zonotopeOrder = 10;
    
    fprintf('Running reach on sys...\n');
    R = reach(sys, params, options);
    disp('Reach succeeded!');
catch err
    disp(getReport(err, 'extended'));
end
