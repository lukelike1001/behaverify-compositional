addpath(genpath('/home/lnam/VSCode/CORA'));

fprintf('--- ACAS Xu layers ---\n');
base_acas = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/AcasXu/networks/';
net_acas = neuralNetwork.readONNXNetwork([base_acas 'ACASXU_run2a_1_1_batch_2000.onnx'], false, 'BSSC');
for i = 1:length(net_acas.layers)
    fprintf('Layer %d: %s\n', i, class(net_acas.layers{i}));
end

fprintf('\n--- Grid World layers ---\n');
base_gw = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2026_TBA/examples/grid_world/networks/';
net_gw = neuralNetwork.readONNXNetwork([base_gw '1000__6_18_0__0200_1.onnx']);
for i = 1:length(net_gw.layers)
    fprintf('Layer %d: %s\n', i, class(net_gw.layers{i}));
end

fprintf('\n--- Testing nonlinearSysDT on Grid World ---\n');
try
    sys_gw = net_gw.nonlinearSysDT();
    fprintf('Grid World converted to nonlinearSysDT successfully!\n');
catch err
    fprintf('Grid World nonlinearSysDT failed: %s: %s\n', err.identifier, err.message);
end
