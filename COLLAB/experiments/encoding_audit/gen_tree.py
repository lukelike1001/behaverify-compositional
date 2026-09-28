"""
Minimal NSBT parameterised by (network, encoding, input domain).

Isolates one question: which of BehaVerify's five neural encodings accept which
ONNX operations, and can any of them take REAL-valued inputs? Everything else
about the tree is held constant.
"""
import sys, pathlib

TREE = """configuration {{
    neural
}}
enumerations {{}}
constants {{lo := 0, hi := 4}}
variables {{
    variable {{ env a VAR {domain} assign{{result{{{init}}}}}}}
    variable {{ env b VAR {domain} assign{{result{{{init}}}}}}}
    variable {{ bl net NEURAL regression {out_domain}
        inputs {{ a, b }}
        num_outputs {{ 1 }}
        config {{ {config} }}
        source {{ '{onnx}' }}
    }}
    variable {{ bl seen VAR BOOLEAN assign{{result{{False}}}}}}
}}
environment_update {{}}
checks {{}}
environment_checks {{}}
actions {{
    action {{ Look
        arguments {{}} local_variables {{}} read_variables {{net}} write_variables {{seen}}
        initial_values {{}}
        update {{
            variable_statement {{ seen assign{{case{{(gt, (index, net, 0), 0)}}result{{True}}result{{False}}}} }}
            return_statement {{result{{success}}}}
        }}
    }}
    action {{ Idle
        arguments {{}} local_variables {{}} read_variables {{}} write_variables {{seen}}
        initial_values {{}}
        update {{
            variable_statement {{ seen assign{{result{{seen}}}} }}
            return_statement {{result{{success}}}}
        }}
    }}
}}
sub_trees {{}}
tree {{
    composite {{ Root sequence children {{
        Look {{}}
        Idle {{}}
    }}}}
}}
tick_prerequisite {{ True }}
specifications {{
    INVARSPEC {{ (or, seen, (not, seen)) }}
}}
"""

# float needs total_size; fixed/fixed_direct need total_size AND decimal_size
CONFIG = {"table": "table", "real": "real", "float": "float 32",
          "fixed": "fixed 32 16", "fixed_direct": "fixed_direct 32 16"}

def write(path, onnx_name, encoding, real_inputs):
    domain = "REAL" if real_inputs else "[lo, hi]"
    path.write_text(TREE.format(
        domain=domain, init="0.0" if real_inputs else "0",
        out_domain="REAL" if real_inputs else "INT",
        config=CONFIG[encoding], onnx=onnx_name))

if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    for net in ("net_gemm_relu", "net_tanh", "net_sigmoid"):
        for enc in ("table", "float", "fixed", "fixed_direct", "real"):
            for real in (False, True):
                tag = f"{net}__{enc}__{'real' if real else 'int'}"
                # source resolves relative to the .tree file's own directory
                write(out / f"{tag}.tree", f"./{net}.onnx", enc, real)
    print("wrote", len(list(out.glob('*__*.tree'))), "trees")
