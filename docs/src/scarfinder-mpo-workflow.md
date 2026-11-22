# Constrained ScarFinder with layer MPOs

This page describes the numerical primitives for a ScarFinder search on a
general χ = 2 iMPS with a constraint folded into every gate (the staggered PXP
chain with the blockade), and the exact meaning of what they record. Selection
rules (primitive period, representative, orbit classes, scores, phase labels)
are not part of the package. A complete example is
`examples/pxp_scarfinder_point.jl`.

## One iteration

```julia
plan = ScarFinderPlan(layers, 0.05; n=2, gate_map=G -> P3 * G, constraint=Blockade())
r = scarfinder_step!(ψ, plan, 2; nstep=20, maxdim=16, target=e, tol=1e-6,
                     repair=true, max_project=3, leak_tol=1e-10)
```

1. Real-time evolution for `nstep` Trotter steps (`plan.mpos`, half steps
   merged, bond cap `maxdim`). The evolved state `b` is retained.
2. Every bond is truncated to χ Schmidt values: `a₀`.
3. With `repair=true`: while the leakage exceeds `leak_tol` and fewer than
   `max_project` passes were made, the full blockade projector is applied and
   the state truncated back to χ. Passes are counted after the initial SVD.
4. `loss = −log F_site(b, a)` on the repaired candidate `a`, before energy
   correction; `loss_svd` is the same for `a₀`.
5. The energy correction starts from the residual of `a`, with imaginary-time
   gates mapped by the **same** `gate_map` (P3 folded), and restarts every
   rejected trial from the last accepted state.

`constraint=Blockade()` without `repair=true` only adds diagnostics, as in
earlier versions.

The repair is a bounded heuristic, not an optimal constrained truncation.
In a reviewed example the first pass lowered the loss from 0.01285235 to
0.01195834; later passes lowered the leakage further but raised the loss
slightly; in eight random examples only three reached `1e-10` within three
passes. The outcome is reported in `project_steps`, `project_status`
(`:off`, `:clean`, `:converged`, `:budget`, `:stalled`, `:annihilated`),
`project_converged` and `leakage_repaired`.

The energy correction reports `converged`, `energy_error` (whole state, after
correction), `energy_initial_error`, `energy_fix_status` (`:converged`,
`:maxstep`, `:stalled`, `:not_run`), `energy_fix_steps` and
`energy_fix_rejected`; with `projected_energy=true` also the energy of the legal
component (`energy_projected_error`, `energy_projected_status`).

### Why corrections fail on leaky candidates

A folded gate tends to the projector, not the identity, as `dτ → 0`. On a
candidate whose repair did not reach `leak_tol`, every trial also moves the
energy by the projection. In the measured fine stages, a correction that missed
`tol` typically got close, after which the residuals of the halved steps
approached the displacement of the projector alone. That limit was farther
from the target than the accepted state, so no shorter step could be accepted
and the remaining trials were spent without progress. These corrections took
about 84 % of all imaginary-time trials of a 30 + 400 run. They are reported
as `:maxstep` and keep the last accepted state. The opt-in
`max_rejections = k` ends such a correction after `k` consecutive rejections
(`:stalled`). In two measured seeds, `k = 6` left all 400 fine-stage residuals
bit-identical and cut the trials by 70 %. It changes the trial count, so it is
not the default.

All static inputs are validated before the state changes: supports and local
dimension, at most one gate per cell bond in a layer, commutation with periodic
translates, `gate_map(I)` commuting with each term, commutation of the mapped
gates (sampled imaginary times), the constraint, and the energy, repair and
truncation parameters.

## Recording trajectories

`scarfinder!(ψ, plan, χ, N; refine=false, observe=f, …)` calls `f(ψ, k, r)`
for `k = 0` (initial state, `r = nothing`) and after each iteration with the
returned candidate and its named tuple. `ψ` is the live state; copy it to keep
it. Use `refine=false` whenever the caller selects its own representative: the
default `refine=true` adds a minimum-entropy scan that can replace `ψ`.
Refinement samples are neither observed nor in `history`.

`evolve!(ψ, mpos, steps; observe=f, every=k)` calls `f(φ, s, stats)` at
`s = 0`, every `k` complete Trotter steps, and at the last step. `φ` is the
state at time `s·dt`, identical to `evolve!(copy, mpos, s)` from the same input.
At a merged half-step boundary the pending half step is applied to a snapshot;
the main trajectory is unchanged. Splitting a window into single-step calls
instead changes the integrator: at χ = 2, ten calls of `evolve!(…, 1)` differed
from one ten-step window by `1.36e-5` in `n_A`.
[`release_recorder`](@ref) records physical time, both bond entropies,
retained dimensions, discarded weight, `num_saturated` (requested dimension
exceeded the cap), `num_at_cap` (retained dimension equal to the cap), site
occupations and the per-site return fidelity. How to turn these into a score is
a project decision.

## Reproducibility and symmetry

- `rand_iMPS(rng, T, n, d, dim)` uses the same uniform `rand` distribution as
  `rand_iMPS(T, n, d, dim)`. Complex Gaussian seeds (`randn`) belong to a
  different distribution and are drawn directly. Derive one RNG per task from
  `(grid point, seed id)`.
- A `ScarFinderPlan` holds no mutable workspace and can be shared by tasks. In
  the pinned environment (Julia 1.12.6, single BLAS thread), four tasks gave
  results bit-identical to serial runs. That covers this environment only; across
  BLAS builds, compare fidelities, spectra, energies and statuses within
  tolerances.
- `translate(ψ, k)` shifts site tensors with their right-bond spectra and
  returns independent arrays. On the staggered chain a one-site translation maps
  `H(μ, δ)` to `H(μ, −δ)`, so at `δ ≠ 0` it is not a symmetry of a grid point.
  Complex conjugation of a real `H` reverses time.

## Migration notes

- `per_site_fidelity` returns the amplitude `|⟨a|b⟩|^(1/n)`. A squared per-site
  overlap is its square, `F_old = F_new²`: a closure tolerance `1 − F_old < ε`
  becomes `1 − F_new < 1 − √(1 − ε) ≈ ε/2`, and a `−log` distance halves.
- `constraint_leakage` is the per-site negative log of the legal amplitude, not
  the maximum double occupancy `max⟨n_i n_{i+1}⟩`; thresholds do not carry over.
- Entropy (`ent_S`, natural log, per bond), "retained dimension reached the cap"
  and "requested dimension exceeded the cap" are separate measurements.
