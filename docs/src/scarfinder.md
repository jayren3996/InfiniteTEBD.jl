# ScarFinder

ScarFinder searches for low-entanglement, weakly thermalizing trajectories
directly on a variational manifold. One iteration leaves the manifold and
projects back onto it:

```julia
evolve!(ψ, U, nstep; maxdim)        # evolve off the manifold
r = project!(ψ, χ; terms, target)   # back to bond dimension χ at energy density target
```

`scarfinder_step!(ψ, U, χ; …)` runs these two stages, and
`scarfinder!(ψ, U, χ, N; …)` runs `N` of them. `U` is anything
[`evolve!`](@ref) accepts:

| `U` | Example |
|-----|---------|
| a sequence of `(G, i, j)` gates (matrices, [`EndpointGate`](@ref)s, [`ControlledGate`](@ref)s) | `trotter_gates(layers, dt; trotter=:second)` |
| a vector of layer MPOs | `trotter_mpos(layers, dt; n=2)` |
| a single local gate `G`, applied at every translation of the cell | `exp(-1im * dt * h)`, a Floquet or projected step `P * U` |

For the iMPS manifold, [`project!`](@ref) is [`truncate!`](@ref) (every bond
keeps its `χ` largest Schmidt values of the canonical form) followed, when
`target` is given, by [`energy_fix!`](@ref) (normalized imaginary-time steps
toward `target`). It returns `(loss, energy_error, converged)`:

- `loss = −log F_site` between the evolved and truncated states, accounting for
  their norms and measured before energy correction;
- `energy_error = E(ψ) − target` after the correction (`nothing` without
  `target`);
- `converged`: whether the correction met `tol` (`true` without `target`).

`scarfinder_step!` uses the bond-dimension projection for dense iMPS and
checks the projection arguments before evolution. For a nonempty built-in MPO
evolution it can reuse the normalized Schmidt basis when the cut is well
separated and the transfer has an isolated dominant fixed point. After
slicing, the gauge is restored on the retained support.
This can change the tensor gauge and roundoff compared with separate calls
to `evolve!` and `project!`. Gate sequences are checked
as each gate is applied, so a later error can leave earlier updates in place.
For a different variational manifold, compose your evolution and projection
in an explicit loop.

## The energy

The energy is given by `terms`, a list of `(h, i, j)` (sites `i ≤ j` of the
chain, repeated with period `n`) or of commuting layers of them, as for
[`energy_fix!`](@ref). The period `n` defaults to the cell length of the first
MPO, or `ψ.n`. Two shortcuts take a local density `h`, which stands for
`(h, i, i + span − 1)` on every site of the cell:

```julia
scarfinder!(ψ, U, h, χ, N; target)               # any U, energy from h
scarfinder!(ψ, h, dt, χ, N; target, trotter)      # U built from h and dt as well
```

In the `(h, dt)` form, `trotter=nothing` applies `exp(−i dt h)` at every
translation of the cell, left to right; `trotter=:second`, `:fourth` or
`:fourth_opt` uses `trotter_gates(uniform_layers(ψ, h), dt; trotter)`, and
`layers=…` replaces the automatic grouping by a commuting decomposition of
your own.

For constrained models the dynamics and the energy differ: the gates carry the
projector, the energy is the bare Hamiltonian. Pass both:

```julia
U = trotter_mpos(layers, dt; n=2, gate_map=G -> P3 * G)   # every gate is P3 * U
scarfinder!(ψ, U, 2, N; terms=layers, target=e, n=2, nstep=20, maxdim=16, refine=false)
```

The correction uses ordinary imaginary-time gates built from `terms`. The next
real-time window projects out the small leakage introduced by it; measure it
with `constraint_leakage(ψ, Blockade())`.

## Keyword arguments

- `nstep=10`: microscopic steps per iteration. One iteration covers physical
  time `Δt ≈ nstep · dt`; `nstep = 1` warns, since one step followed by a hard
  projection is usually too aggressive.
- `maxdim=MAXDIM`: bond dimension the evolution may grow into before the
  projection to `χ`; typically larger than `χ`.
- `cutoff=SVDTOL`: singular-value cutoff of the evolution and the projection.
- `terms`, `target=nothing`, `tol=1e-6`, `alpha=0.1`, `maxstep=50`, `n`: energy
  correction, see [`energy_fix!`](@ref).
- `refine=true`, `refine_step=100`, `refine_U=U` (`refine_dt=dt/10` in the
  `(h, dt)` form): after the `N` iterations, scan `refine_step` further steps
  with `refine_U` and keep the state of lowest maximal bond entropy. Set
  `refine=false` to run exactly `N` iterations.
- `history`: a vector that collects the `N` results.
- `observe`: a function `f(ψ, k, r)`, called with `k = 0` and `r = nothing` on
  the initial state and after each iteration with the candidate and its
  result. `ψ` is the live state; copy it to keep it. Refinement samples are
  neither observed nor in `history`.

## Composing the loop yourself

```julia
gates = trotter_gates(uniform_layers(ψ, h), dt; trotter=:fourth)
terms = [(h, i, i + 2) for i in 1:ψ.n]
for _ in 1:N
    evolve!(ψ, gates, nstep; maxdim)
    r = project!(ψ, χ; terms, target)
    r.converged || break
end
```

[`minimize_on_trajectory!`](@ref) is the refinement scan with your own step
and objective:

```julia
advance!(state) = evolve!(state, gates, 1; maxdim)
objective(state) = maximum(i -> ent_S(state, i), 1:state.n)
selection = minimize_on_trajectory!(objective, advance!, ψ, 50)
```

It considers `ψ` and the states after each of the 50 steps, writes the first
minimizer into `ψ` and returns `selection.step` and `selection.value`.

## Recording a release

`evolve!(ψ, mpos, steps; observe=f, every=k)` calls `f(φ, s, stats)` at
`s = 0`, every `k` complete Trotter steps, and at the last step. `φ` is the
state at time `s·dt`, identical to `evolve!(copy, mpos, s)` from the same
input; at a merged half-step boundary the pending half step is applied to a
snapshot, and the main trajectory is unchanged. Splitting a window into
single-step calls instead changes the integrator: at χ = 2, ten calls of
`evolve!(…, 1)` differed from one ten-step window by `1.36e-5` in `n_A`.

[`release_recorder`](@ref) records physical time, bond entropies, retained
dimensions, discarded weight, `num_saturated` (requested dimension exceeded the
cap), `num_at_cap` (retained dimension equal to the cap), site occupations and
the per-site return fidelity. How to turn these into a score is up to the
caller.

## A complete PXP example

The gate `G` contains a two-site blockade projector that restores the
Rydberg-constraint subspace after each truncation; the energy target is that
of the initial Z2 product state.

```@example
using InfiniteTEBD
using LinearAlgebra

P0 = [0 0; 0 1]
N1 = [1 0; 0 0]
X = [0 1; 1 0]
h_pxp = kron(P0, X, P0)                                  # three-site PXP density
no_double_2 = Matrix{Float64}(I, 4, 4) - kron(N1, N1)    # two-site blockade
dt = 0.01
G = kron(no_double_2, I(2)) * kron(I(2), no_double_2) * exp(-1im * dt * h_pxp)

psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
target = energy_density(psi, h_pxp)

# One ScarFinder iteration represents Δt = 0.05.
scarfinder!(psi, G, h_pxp, 2, 5; nstep=5, target, maxdim=12, refine=false)

(; target_energy=target, final_energy=energy_density(psi, h_pxp),
   maxbond=maximum(length.(psi.λ)))
```

A full grid point of the staggered PXP phase diagram, with seeds, recording
and a release, is `examples/pxp_scarfinder_point.jl`.

## Notes

- `rand_iMPS(rng, T, n, d, dim)` draws from the same uniform distribution as
  `rand_iMPS(T, n, d, dim)`; derive one RNG per task from
  `(grid point, seed id)` for reproducible runs.
- `translate(ψ, k)` shifts the site tensors with their spectra. On the
  staggered chain a one-site translation maps `H(μ, δ)` to `H(μ, −δ)`.
- `per_site_fidelity` returns the amplitude `|⟨a|b⟩|^(1/n)`; a squared per-site
  overlap is its square.
- `constraint_leakage` is the per-site negative log of the legal amplitude, not
  the maximal double occupancy.
- [`energy_span`](@ref) estimates the lowest and highest energy densities
  reachable by imaginary-time evolution with `h`, useful for picking `target`.

## Reference

J. Ren, A. Hallam, L. Ying, and Z. Papić, [ScarFinder: A Detector of Optimal
Scar Trajectories in Quantum Many-Body Dynamics](https://eprints.whiterose.ac.uk/id/eprint/238368/),
PRX Quantum 6, 040332 (2025).
