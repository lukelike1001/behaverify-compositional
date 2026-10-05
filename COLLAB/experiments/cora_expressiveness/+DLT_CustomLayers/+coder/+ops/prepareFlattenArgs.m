function [dim1, dim2, numDimsY] = prepareFlattenArgs(X_, ONNXAxis_, numDimsX_)
% Prepares arguments for implementing the ONNX Flatten operator
%#codegen

%   Copyright 2024 The MathWorks, Inc.    

% ONNXAxis is the number of dimensions that go on the left in ONNX, so here
% it is the number of dimensions that go on the right.
X = DLT_CustomLayers.coder.ops.extractIfDlarray(X_);
ONNXAxis = DLT_CustomLayers.coder.ops.extractIfDlarray(ONNXAxis_);
numDimsX = DLT_CustomLayers.coder.ops.extractIfDlarray(numDimsX_);


if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end
if ONNXAxis == 0
    dim2 = 1;
else
    dim2 = prod(size(X, numDimsX + 1 - ONNXAxis:numDimsX));     % numel on the right
end
dim1 = numel(X)/dim2;                                   % numel on the left
numDimsY = 2;
end
