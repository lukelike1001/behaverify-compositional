#!/usr/bin/env python3
"""
KILL 1 AUDIT: Measuring how Table, Fixed, and Float encodings scale
as the Neural Network INPUT SPACE grows (network architecture held constant).
Templating directly from the author's published small grid trees.
"""
import os
import sys
import time
import subprocess
import shutil

NUXMV = "/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv"
DSL_TO_NUXMV = "/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2024_ANSR/src/dsl_to_nuxmv.py"
METAMODEL = "/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2024_ANSR/metamodel/behaverify.tx"
BASE_DIR = "/home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2024_ANSR/examples/small"

SCRATCH_DIR = "/home/lnam/.gemini/antigravity-cli/brain/34099bb9-e53e-43e4-8c3b-54c2b2621e62/scratch/kill1_audit"
os.makedirs(SCRATCH_DIR, exist_ok=True)
shutil.copy(os.path.join(BASE_DIR, "9_10_1__128_1.onnx"), os.path.join(SCRATCH_DIR, "9_10_1__128_1.onnx"))

with open(os.path.join(BASE_DIR, "table.tree")) as f:
    TABLE_TEMPLATE = f.read()

with open(os.path.join(BASE_DIR, "fixed.tree")) as f:
    FIXED_TEMPLATE = f.read()

with open(os.path.join(BASE_DIR, "float.tree")) as f:
    FLOAT_TEMPLATE = f.read()

def generate_tree(mode, max_val, out_path):
    if mode == "table":
        template = TABLE_TEMPLATE
    elif mode == "fixed":
        template = FIXED_TEMPLATE
    elif mode == "float":
        template = FLOAT_TEMPLATE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Replace constants line: min_val := 0, max_val := 9
    content = template.replace("min_val := 0, max_val := 9", f"min_val := 0, max_val := {max_val}")
    with open(out_path, "w") as f:
        f.write(content)

def run_test(mode, max_val, timeout_sec=120):
    grid_dim = max_val + 1
    input_points = grid_dim ** 4
    tag = f"{mode}_M{max_val}_pts{input_points}"
    tree_path = os.path.join(SCRATCH_DIR, f"{tag}.tree")
    smv_path = os.path.join(SCRATCH_DIR, f"{tag}.smv")
    generate_tree(mode, max_val, tree_path)

    # 1. Translate to SMV
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["python3", DSL_TO_NUXMV, METAMODEL, tree_path, smv_path],
            cwd=SCRATCH_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec
        )
        t_trans = time.time() - t0
        if proc.returncode != 0:
            err_line = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "error"
            return {"mode": mode, "grid_dim": grid_dim, "input_points": input_points, "status": f"trans_err: {err_line}"}
    except subprocess.TimeoutExpired:
        return {"mode": mode, "grid_dim": grid_dim, "input_points": input_points, "status": "trans_timeout"}

    smv_size_kb = os.path.getsize(smv_path) / 1024.0
    with open(smv_path) as f:
        smv_lines = sum(1 for _ in f)

    # 2. Run nuXmv on invariant check only (as in NeuS Table 3)
    # command: time; go; check_invar; time; quit
    cmd_content = "time\ngo\ncheck_invar\ntime\nquit\n"
    cmd_file = os.path.join(SCRATCH_DIR, f"{tag}_cmd.txt")
    with open(cmd_file, "w") as f:
        f.write(cmd_content)

    t0 = time.time()
    try:
        p = subprocess.Popen(
            ["/usr/bin/time", "-v", NUXMV, "-source", cmd_file, smv_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = p.communicate(timeout=timeout_sec)
        t_nuxmv = time.time() - t0
        
        max_rss = None
        for line in err.splitlines():
            if "Maximum resident set size" in line:
                max_rss = int(line.split(":")[-1].strip())
        
        verdict = "UNKNOWN"
        if "is true" in out:
            verdict = "TRUE"
        elif "is false" in out:
            verdict = "FALSE"

        return {
            "mode": mode,
            "grid_dim": grid_dim,
            "input_points": input_points,
            "t_trans": round(t_trans, 3),
            "smv_lines": smv_lines,
            "smv_kb": round(smv_size_kb, 1),
            "t_nuxmv": round(t_nuxmv, 3),
            "max_rss_mb": round(max_rss / 1024.0, 1) if max_rss else None,
            "verdict": verdict,
            "status": "OK"
        }
    except subprocess.TimeoutExpired:
        p.kill()
        return {
            "mode": mode,
            "grid_dim": grid_dim,
            "input_points": input_points,
            "t_trans": round(t_trans, 3),
            "smv_lines": smv_lines,
            "smv_kb": round(smv_size_kb, 1),
            "t_nuxmv": f"> {timeout_sec}s",
            "status": "TIMEOUT"
        }

if __name__ == "__main__":
    print(f"{'Mode':<8} {'Grid':<6} {'InputPts':<10} {'Trans(s)':<10} {'SMV Lines':<10} {'SMV KB':<10} {'nuXmv(s)':<10} {'RSS(MB)':<10} {'Verdict':<8}")
    print("-" * 88)
    
    # We test:
    # max_val = 2 -> 3x3 grid (81 points)
    # max_val = 4 -> 5x5 grid (625 points)
    # max_val = 6 -> 7x7 grid (2,401 points)
    # max_val = 9 -> 10x10 grid (10,000 points - the author's exact small grid)
    # max_val = 14 -> 15x15 grid (50,625 points)
    for max_val in [2, 4, 6, 9, 14]:
        for mode in ["table", "fixed"]:
            res = run_test(mode, max_val, timeout_sec=180)
            if res.get("status") == "OK":
                print(f"{res['mode']:<8} {res['grid_dim']:<6} {res['input_points']:<10} {res['t_trans']:<10} {res['smv_lines']:<10} {res['smv_kb']:<10} {res['t_nuxmv']:<10} {res['max_rss_mb']:<10} {res['verdict']:<8}")
            else:
                print(f"{res['mode']:<8} {res['grid_dim']:<6} {res['input_points']:<10} {res.get('t_trans', '-'):<10} {res.get('smv_lines', '-'):<10} {res.get('smv_kb', '-'):<10} {res.get('t_nuxmv', '-'):<10} {'-':<10} {res.get('status', 'FAIL'):<8}")
            sys.stdout.flush()
