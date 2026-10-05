# The benchmark — draft v0.3

**Name:** `divert` (working). "Stormy NAV" is retired: the weather variables were
decorative, and the 6-D dimension lever only mattered for a reachability contest
we are no longer entering.

**The headline is node-status properties over continuous state.** Not a
reachability comparison. CORA will compute the reachable set of this plant
easily and that is fine — it cannot *state* the properties we verify.

Every claim below names the measurement that falsifies it. That is what the
geofence lacked.

---

## 1. Plant — 4-D planar vehicle

```
x'  = x  + vx*dt              vx' = vx + (ax - cd*vx)*dt
y'  = y  + vy*dt              vy' = vy + (ay - cd*vy)*dt      dt = 0.1
```

Continuous, discrete time. Grounded, not invented: this is the double
integrator of the run-time-assurance tutorial (Hobbs, Mote, Abate, Coogan,
Feron, *IEEE Control Systems Magazine* 2023, doi:10.1109/MCS.2023.3234380),
whose Fig. 3 is a vehicle approaching a wall and whose viability kernel our
geofence envelope already matched to four decimals (claim #100). The planar
version is two copies of it.

An obstacle `O = [x1,x2] x [y1,y2]` and a goal region `G`.

## 2. Two networks

```
net_cruise  : (x,y,vx,vy) -> (ax,ay)    reach G while avoiding O
net_divert  : (x,y,vx,vy) -> (ax,ay)    return to the start region
```

Two networks is not decoration. NeuS 2025 diagnoses its own worst case as *"the
5 NNs... each of these networks also affects the same variable may create
unexpected complexity in nuXmv"*, and claim #108 gives the mechanism: each
freed network becomes one `VAR` plus N `INVAR`s, so two of them writing the
same downstream variable is precisely the nondeterminism that defeats the
fastforwarding `DEFINE` encoding.

## 3. Tree

```
Root  selector
├── Recovery  sequence
│   ├── HazardMonitor  parallel policy success_on_one with_partial_memory
│   │     ├── OutOfBounds?   |x| >= xb  or  |y| >= yb
│   │     ├── TooFast?       vx^2 + vy^2 >= s_max
│   │     └── Stuck?         progress counter exceeded
│   └── DivertProcedure  sequence with_partial_memory
│         ├── Brake    ax,ay := -k*(vx,vy)
│         └── Return   ax,ay := net_divert(x,y,vx,vy)
└── NeuralCruise       ax,ay := net_cruise(x,y,vx,vy)
```

`HazardMonitor` has memory, so the hazard **latches**. `DivertProcedure` has
memory, so it resumes rather than restarting each tick.

## 4. Properties — the headline

```
S1  safety      the vehicle never enters O                        [control]
N1  recovery    Recovery is never active more than N ticks
N2  no preempt  once DivertProcedure is active it stays active until done
N3  latching    a hazard true at tick k is still true at k+1
N4  liveness-ish  Brake completes within M ticks of Recovery starting
```

S1 is the **control**: CORA should handle it well, and including a property we
expect to lose on is how the comparison stays honest.

N1–N4 are over node activation. **N1 is already demonstrated** on the geofence
(claim #93): refuted at N=12 and N=16, proved at N=19 and N=22, against a true
maximum of 18 — tight to about one tick and non-vacuous. This benchmark turns
one such result into a family.

**The trap to avoid, which we already hit once.** `geofence_latched.tree`
generated `INVARSPEC (Brake.active -> x_stage_1 >= 0.7)`, which is a tautology
of the guard and proves nothing. A node-status property only has content when
discharging it requires the *closed loop* — N1 needs the network to actually
bring the state back. **Every property above must be checked for guard-tautology
before it is used.**

## 5. Falsifiers — measure these before building any comparison

| claim | falsified if |
|---|---|
| monolithic cannot run this | a sensible integer quantisation of (x,y,vx,vy) stays under ~1.6M cases at 19.6 KB/case (claim #114) and the table verifies S1. **Compute the grid size needed for the property to still hold.** |
| N1–N4 are not guard tautologies | any of them can be discharged from the guard alone, without the plant |
| N1–N4 need a tree | any can be rewritten over `(x,y,vx,vy,t)` alone, making it a CORA reach-avoid query |
| two networks matter | merging both into one network leaves the verification cost unchanged |
| memory matters | replacing `with_partial_memory` by no memory leaves all properties true and costs the same |
| the result is real | the TRUE system violates any of S1, N1–N4 — **simulate first**, as the geofence envelope had to be rebuilt after the first guess gave max x = 2.55 |

## 6. Horizon

Compositional dies between 40 and 60 steps on a 2-D system (0.1 / 7.7 / 54.9 /
246 s at 10 / 20 / 30 / 40, no proof at 60). **Design the mission to complete
inside 40 ticks** and state the horizon explicitly. If it cannot, that is a
finding about BMC's limits — which is §6.1.1 of the dissertation, measured.

## 7. Build order

1. Plant, obstacle, two networks. Simulate densely. **Confirm the true system
   satisfies S1 and N1–N4**, and derive the operating envelope rather than
   asserting it.
2. Run every falsifier in §5.
3. Only then: contracts, Z3, and a serious CORA attempt.

## 8. Both outcomes are publishable

The benchmark definition plus a tool-capability matrix is the contribution;
the compositional result is the first row of the results table. ARCH-COMP
AINNCS has twelve benchmarks and none has discrete execution structure, and the
category is run out of this group's own repository.
