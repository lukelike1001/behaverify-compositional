addpath(genpath('/home/lnam/VSCode/CORA'));
base_acas = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/AcasXu/networks/';
net = neuralNetwork.readONNXNetwork([base_acas 'ACASXU_run2a_1_1_batch_2000.onnx'], false, 'BSSC');

% Create a dummy plant for 5 states and 5 inputs
A_plant = eye(5);
B_plant = eye(5);
plant = linearSys('plant', A_plant, B_plant);

fprintf('Testing neurNetContrSys inside hybridAutomaton location...\n');
try
    nnc = neurNetContrSys(plant, net, 1.0);
    disp('Created neurNetContrSys:');
    disp(nnc);
    
    % Try putting nnc inside a location
    inv = fullspace(5);
    trans = transition();
    loc1 = location('loc1', inv, trans, nnc);
    disp('Created location with neurNetContrSys:');
    disp(loc1);
    
    HA = hybridAutomaton('testHA', loc1);
    disp('Created hybridAutomaton:');
    disp(HA);
    
    % Now try to run reach
    params.R0 = zonotope(zeros(5,1), 0.1*eye(5));
    params.tStart = 0;
    params.tFinal = 2;
    params.startLoc = 1;
    params.finalLoc = 0;
    params.Uloc = {zonotope(zeros(0,1))};
    params.uloc = {zeros(0,1)};
    params.Wloc = {zonotope(zeros(5,1))};
    params.Vloc = {zonotope(zeros(5,1))};
    
    options.timeStepLoc = {1.0};
    options.verbose = false;
    options.linAlg = 'standard';
    
    fprintf('Attempting reach on hybridAutomaton with neurNetContrSys...\n');
    R = reach(HA, params, options);
    disp('Reach succeeded!');
catch err
    disp(getReport(err, 'extended'));
end
