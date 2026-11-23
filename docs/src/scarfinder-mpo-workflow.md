# Constrained ScarFinder with layer MPOs

This page describes the numerical primitives for a ScarFinder search on a
general χ = 2 iMPS in the staggered PXP chain. A complete example is
`examples/pxp_scarfinder_point.jl`.

## One iteration

```julia
U = trotter_mpos(layers, 0.05; n=2, gate_map=G -> P3 * G)
r = scarfinder_step!(ψ, U, 2; nstep=20, maxdim=16, cutoff=1e-12,
                     terms=layers, target=e, n=2)
```

The step runs `evolve!`, truncates each bond with `truncate!` and calls
`energy_fix!` when given a target. The real-time gates carry the blockade
projector. The correction uses ordinary imaginary-time gates built from
`layers`. The next real-time window projects out the small leakage introduced
after the previous window. Measure it separately with
`constraint_leakage(ψ, Blockade())`.

The result is `(loss, energy_error, converged)`. The loss is minus the log of
the per-site fidelity between the evolved and truncated states. The energy
error is the residual of the state after correction.

## Recording trajectories

`scarfinder!(ψ, U, χ, N; refine=false, observe=f, …)` calls `f(ψ, k, r)`
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
