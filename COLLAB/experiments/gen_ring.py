"""
Generate an N-robot ring behaviour tree, matching examples/MultiAgent/FiveInRing.tree.

This is a REALISTIC scaling axis, unlike the synthetic binary tree: N robots on a
ring of N+1 positions, each running an identical 6-node subtree, coordinating
through a shared position vector. State space is (N+1)^N, and the tree grows
linearly in N. It is exactly the shape where per-agent assume-guarantee
reasoning would be expected to pay off, which is why it is the right benchmark
for testing whether monolithic BehaVerify has any headroom left.
"""
import sys, pathlib

def ring(n):
    npos = n + 1
    goal = lambda i: (i + 3) % npos
    others = lambda i: [j for j in range(n) if j != i]
    L = []
    L.append("configuration {}")
    L.append("enumerations {'Move', 'Stay'}")
    L.append(f"constants {{num_pos := {npos}}}")
    L.append("\nvariables {")
    for i in range(n):
        L.append(f"    variable {{ bl act{i} VAR {{'Move', 'Stay'}} assign{{result{{'Stay'}}}}}}")
    for i in range(n):
        L.append(f"    variable {{ env x_pos{i} VAR [0, {npos-1}] assign{{result{{{i}}}}}}}")
    L.append("}\n")
    L.append("environment_update {")
    for i in range(n):
        L.append(f"    variable_statement {{ x_pos{i}\n        assign {{\n"
                 f"            case {{(eq, act{i}, 'Move')}} result {{(mod, (add, x_pos{i}, 1), num_pos)}}\n"
                 f"            result {{x_pos{i}}}\n        }}\n    }}")
    L.append("}\n")
    L.append("checks {}\n")
    L.append("environment_checks {")
    allpos = ", ".join(f"x_pos{j}" for j in range(n))
    for i in range(n):
        L.append(f"    environment_check {{ AtGoal{i}\n        arguments {{}} read_variables {{x_pos{i}}}\n"
                 f"        condition {{(eq, x_pos{i}, {goal(i)})}}}}")
    for i in range(n):
        clauses = "".join(
            f"\n            (eq, x_pos{j}, (mod, (add, x_pos{i}, 1), num_pos))," for j in others(i)).rstrip(",")
        L.append(f"    environment_check {{ PathClear{i}\n        arguments {{}} read_variables {{{allpos}}}\n"
                 f"        condition {{(not, (or,{clauses}))}}}}")
    L.append("}\n")
    L.append("actions {")
    for i in range(n):
        for nm, val in (("Move", "Move"), ("SetStayGoal", "Stay"), ("SetStayWait", "Stay")):
            L.append(f"    action {{ {nm}{i}\n        arguments {{}} local_variables {{}} read_variables {{}} "
                     f"write_variables {{act{i}}}\n        initial_values {{}}\n        update {{\n"
                     f"            variable_statement {{act{i} assign{{result{{'{val}'}}}}}}\n"
                     f"            return_statement {{result{{success}}}}\n        }}\n    }}")
    L.append("}\n")
    L.append("sub_trees {}\n")
    L.append("tree {\n    composite { Root parallel policy success_on_all children {")
    for i in range(n):
        L.append(f"        composite {{ Robot{i} selector children {{\n"
                 f"            composite {{ AtGoalSeq{i} sequence children {{\n                AtGoal{i} {{}}\n"
                 f"                SetStayGoal{i} {{}}\n            }}}}\n"
                 f"            composite {{ TryMove{i} sequence children {{\n                PathClear{i} {{}}\n"
                 f"                Move{i} {{}}\n            }}}}\n            SetStayWait{i} {{}}\n        }}}}")
    L.append("    }}\n}\n")
    L.append("tick_prerequisite {(True)}\n")
    pairs = "".join(f"\n        (neq, x_pos{i} at -1, x_pos{j} at -1),"
                    for i in range(n) for j in range(i+1, n)).rstrip(",")
    L.append("specifications {")
    L.append(f"    INVARSPEC {{(and,{pairs})}}")
    for i in range(n):
        L.append(f"    CTLSPEC {{(always_globally, (always_finally, (eq, x_pos{i} at -1, {goal(i)})))}}")
    L.append("}")
    return "\n".join(L) + "\n"

out = pathlib.Path(sys.argv[1])
for n in [int(a) for a in sys.argv[2:]]:
    p = out / f"ring_{n}.tree"; p.write_text(ring(n))
    print(f"  N={n:<3} positions={n+1:<3} state space={(n+1)**n:>18,}  nodes={1+6*n:<4} {p.name}")
