classdef FlattenLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end


    methods(Static, Hidden)
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'DLT_CustomLayers.coder.FlattenLayer1000';
        end
    end


    methods
        function this = FlattenLayer1000(name)
            this.Name = name;
            this.OutputNames = {'Operation_1_Flatten'};
        end

        function [Operation_1_Flatten] = predict(this, input_Sub)
            if isdlarray(input_Sub)
                input_Sub = stripdims(input_Sub);
            end
            input_SubNumDims = 4;
            input_Sub = DLT_CustomLayers.ops.permuteInputVar(input_Sub, [4 1 2 3], 4);

            [Operation_1_Flatten, Operation_1_FlattenNumDims] = FlattenGraph1000(this, input_Sub, input_SubNumDims, false);
            Operation_1_Flatten = DLT_CustomLayers.ops.permuteOutputVar(Operation_1_Flatten, [2 1], 2);

            Operation_1_Flatten = dlarray(single(Operation_1_Flatten), 'CB');
        end

        function [Operation_1_Flatten] = forward(this, input_Sub)
            if isdlarray(input_Sub)
                input_Sub = stripdims(input_Sub);
            end
            input_SubNumDims = 4;
            input_Sub = DLT_CustomLayers.ops.permuteInputVar(input_Sub, [4 1 2 3], 4);

            [Operation_1_Flatten, Operation_1_FlattenNumDims] = FlattenGraph1000(this, input_Sub, input_SubNumDims, true);
            Operation_1_Flatten = DLT_CustomLayers.ops.permuteOutputVar(Operation_1_Flatten, [2 1], 2);

            Operation_1_Flatten = dlarray(single(Operation_1_Flatten), 'CB');
        end

        function [Operation_1_Flatten, Operation_1_FlattenNumDims1001] = FlattenGraph1000(this, input_Sub, input_SubNumDims, Training)

            % Execute the operators:
            % Flatten:
            [dim1, dim2, Operation_1_FlattenNumDims] = DLT_CustomLayers.ops.prepareFlattenArgs(input_Sub, 1, input_SubNumDims);
            Operation_1_Flatten = reshape(input_Sub, dim1, dim2);

            % Set graph output arguments
            Operation_1_FlattenNumDims1001 = Operation_1_FlattenNumDims;

        end

    end

end