addpath(genpath('/home/lnam/VSCode/CORA'));
base = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/AcasXu/networks/';
net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], false, 'BSSC');
c = [(10000 - 19791.091) / 60621; 0; 0; 0.3; 0.3];
r = [100 / 60621; 0; 0; 0; 0];
y = net.evaluate(c);
[~, k] = max(y);
A = zeros(4, 5); b = zeros(4, 1);
others = [1:k-1, k+1:5];
for i = 1:4
    A(i, others(i)) = 1; A(i, k) = -1;
end
[res, x_, y_] = net.verify(c, r, A, b, true, struct(), 120, false);
disp('res:');
disp(res);
disp('x_:');
disp(x_);
disp('y_:');
disp(y_);
