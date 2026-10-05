addpath(genpath('/home/lnam/VSCode/CORA'));
base = '/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/AcasXu/networks/';
net = neuralNetwork.readONNXNetwork([base 'ACASXU_run2a_1_1_batch_2000.onnx'], false, 'BSSC');
rho_mean = 19791.091; rho_range = 60621;

% Search for transition point along rho with theta=0, psi=0, v_own=0.3, v_int=0.3
prev_k = -1;
trans_rho = -1;
for rho = 10000:100:25000
    c = [(rho - rho_mean)/rho_range; 0; 0; 0.3; 0.3];
    y = net.evaluate(c);
    [~, k] = max(y);
    if prev_k ~= -1 && k ~= prev_k
        fprintf('Transition from adv %d to adv %d between rho=%d and rho=%d\n', prev_k, k, rho-100, rho);
        trans_rho = rho - 50;
        k1 = prev_k;
        k2 = k;
        break;
    end
    prev_k = k;
end

if trans_rho ~= -1
    fprintf('\nTesting box centered at transition rho=%d with radius 100 ft (spans boundary):\n', trans_rho);
    c = [(trans_rho - rho_mean)/rho_range; 0; 0; 0.3; 0.3];
    r = [100/rho_range; 0; 0; 0; 0];
    
    % Test spec 1: does k1 win everywhere?
    A1 = zeros(4, 5); b1 = zeros(4, 1);
    others1 = [1:k1-1, k1+1:5];
    for i = 1:4, A1(i, others1(i)) = 1; A1(i, k1) = -1; end
    res1 = net.verify(c, r, A1, b1, true, struct(), 30, false);
    fprintf('Spec: adv %d wins everywhere -> res: %s\n', k1, res1.str);
    
    % Test spec 2: does k2 win everywhere?
    A2 = zeros(4, 5); b2 = zeros(4, 1);
    others2 = [1:k2-1, k2+1:5];
    for i = 1:4, A2(i, others2(i)) = 1; A2(i, k2) = -1; end
    res2 = net.verify(c, r, A2, b2, true, struct(), 30, false);
    fprintf('Spec: adv %d wins everywhere -> res: %s\n', k2, res2.str);
end
