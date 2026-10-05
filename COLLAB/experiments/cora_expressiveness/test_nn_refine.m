addpath(genpath('/home/lnam/VSCode/CORA'));
base_acas = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/AcasXu/networks/';
net = neuralNetwork.readONNXNetwork([base_acas 'ACASXU_run2a_1_1_batch_2000.onnx'], false, 'BSSC');

fprintf('Testing nn.refine on ACAS Xu network...\n');
disp('Initial layer orders:');
ref_layers = net.getRefinableLayers();
for i = 1:min(5, length(ref_layers))
    fprintf('  Layer %d: order = %d\n', i, max(ref_layers{i}.order));
end

% Call refine
fprintf('\nCalling net.refine(3, "layer", "approx_error", [], true)...\n');
try
    net.refine(3, 'layer', 'approx_error', [], true);
    disp('Refinement completed!');
    disp('Updated layer orders:');
    for i = 1:min(5, length(ref_layers))
        fprintf('  Layer %d: order = %d\n', i, max(ref_layers{i}.order));
    end
catch err
    disp(getReport(err, 'extended'));
end
