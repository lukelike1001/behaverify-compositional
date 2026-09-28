"""
Networks of increasing hidden width, identical in every other respect.

Tests the claim that general SMT suffers exponential ReLU phase-splitting: each
extra hidden neuron adds one boolean split per BMC step, so if that argument is
right, verification time should blow up with width.
"""
import sys, pathlib
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

def build(width, path, rng):
    w1 = rng.uniform(-0.6, 0.6, (width, 2)).astype(np.float32)
    b1 = rng.uniform(-0.3, 0.3, width).astype(np.float32)
    w2 = rng.uniform(-0.6, 0.6, (2, width)).astype(np.float32)
    b2 = rng.uniform(-0.3, 0.3, 2).astype(np.float32)
    for arr in (w1, b1, w2, b2):        # no exact zeros: fixed/fixed_direct crash on them
        arr[arr == 0.0] = 0.01
    g = helper.make_graph(
        [helper.make_node('Gemm', ['X', 'w1', 'b1'], ['h'], transB=1),
         helper.make_node('Relu', ['h'], ['hr']),
         helper.make_node('Gemm', ['hr', 'w2', 'b2'], ['y'], transB=1)],
        f'net{width}',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(w1, 'w1'), numpy_helper.from_array(b1, 'b1'),
         numpy_helper.from_array(w2, 'w2'), numpy_helper.from_array(b2, 'b2')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.checker.check_model(m); onnx.save(m, path)

TREE = """configuration {{
    neural
}}
enumerations {{'left', 'right'}}
constants {{horizon := 20}}
variables {{
    variable {{ env a VAR REAL assign{{result{{0.0}}}}}}
    variable {{ env b VAR REAL assign{{result{{0.0}}}}}}
    variable {{ env ticks VAR [0, 20] assign{{result{{0}}}}}}
    variable {{ bl net NEURAL classification {{'left', 'right'}}
        inputs {{ a, b }}
        config {{ real }}
        source {{ './net{w}.onnx' }}
    }}
    variable {{ bl seen VAR BOOLEAN assign{{result{{False}}}}}}
}}
environment_update {{
    variable_statement {{ a assign{{result{{(add, a, 3.0)}}}} }}
    variable_statement {{ b assign{{result{{(sub, b, 3.0)}}}} }}
    variable_statement {{ ticks assign{{result{{(if, (lt, ticks, horizon), (add, ticks, 1), horizon)}}}} }}
}}
checks {{}}
environment_checks {{}}
actions {{
    action {{ Look arguments {{}} local_variables {{}} read_variables {{net}} write_variables {{seen}}
        initial_values {{}}
        update {{ variable_statement {{ seen assign{{case{{(eq, net, 'left')}}result{{True}}result{{False}}}} }}
                 return_statement {{result{{success}}}} }} }}
    action {{ Idle arguments {{}} local_variables {{}} read_variables {{}} write_variables {{seen}}
        initial_values {{}}
        update {{ variable_statement {{ seen assign{{result{{seen}}}} }} return_statement {{result{{success}}}} }} }}
}}
sub_trees {{}}
tree {{ composite {{ Root sequence children {{ Look {{}} Idle {{}} }}}} }}
tick_prerequisite {{ (lt, ticks, horizon) }}
specifications {{
    INVARSPEC {{ (implies, (gt, a, 30.0), (eq, net, 'left')) }}
}}
"""

out = pathlib.Path(sys.argv[1]); rng = np.random.default_rng(0)
for w in [int(x) for x in sys.argv[2:]]:
    build(w, out / f"net{w}.onnx", rng)
    (out / f"w{w}.tree").write_text(TREE.format(w=w))
    print(f"  width {w:>4}: net{w}.onnx + w{w}.tree")
