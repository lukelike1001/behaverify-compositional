function [dim1, dim2, numDimsY] = prepareFlattenArgs(X, ONNXAxis, numDimsX)
% Prepares arguments for implementing the ONNX Flatten operator

%   Copyright 2020 The MathWorks, Inc.    

% ONNXAxis is the number of dimensions that go on the left in ONNX, so here
% it is the number of dimensions that go on the right.
if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end
if ONNXAxis == 0
    dim2 = 1;
else
    dim2 = prod(size(X, numDimsX+1-ONNXAxis:numDimsX));     % numel on the right
end
dim1 = numel(X)/dim2;                                   % numel on the left
numDimsY = 2;
end
