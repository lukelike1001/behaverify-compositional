classdef FlattenLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = DLT_CustomLayers.coder.FlattenLayer1000(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = DLT_CustomLayers.FlattenLayer1000(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
        end
    end

    methods
        function this = FlattenLayer1000(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'Operation_1_Flatten'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = DLT_CustomLayers.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
        end

        function [Operation_1_Flatten] = predict(this, input_Sub__)
            if isdlarray(input_Sub__)
                input_Sub_ = stripdims(input_Sub__);
            else
                input_Sub_ = input_Sub__;
            end
            input_SubNumDims = 4;
            input_Sub = DLT_CustomLayers.coder.ops.permuteInputVar(input_Sub_, [4 1 2 3], 4);

            [Operation_1_Flatten__, Operation_1_FlattenNumDims__] = FlattenGraph1000(this, input_Sub, input_SubNumDims, false);
            Operation_1_Flatten_ = DLT_CustomLayers.coder.ops.permuteOutputVar(Operation_1_Flatten__, [2 1], 2);

            Operation_1_Flatten = dlarray(single(Operation_1_Flatten_), 'CB');
        end

        function [Operation_1_Flatten, Operation_1_FlattenNumDims1001] = FlattenGraph1000(this, input_Sub, input_SubNumDims, Training)

            % Execute the operators:
            % Flatten:
            [dim11000, dim21001, Operation_1_FlattenNumDims] = DLT_CustomLayers.coder.ops.prepareFlattenArgs(input_Sub, 1, coder.const(input_SubNumDims));
            Operation_1_Flatten = reshape(input_Sub, dim11000, dim21001);

            % Set graph output arguments
            Operation_1_FlattenNumDims1001 = coder.const(Operation_1_FlattenNumDims);

        end

    end

end