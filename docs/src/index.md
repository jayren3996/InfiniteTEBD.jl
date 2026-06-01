```@raw html
<div style="text-align: center; margin-bottom: 1.5rem;">
  <img src="assets/logo.svg" alt="InfiniteTEBD.jl" width="160"/>
</div>
```

# InfiniteTEBD.jl

`InfiniteTEBD.jl` implements the time-evolving block decimation (TEBD) algorithm
for infinite matrix-product states (iMPS). A state is specified by a periodic
unit cell and represented directly in the thermodynamic limit, so no
finite-size extrapolation from a long open chain is required. This
representation is appropriate when the observables of interest (entanglement
structure, bulk energy density, and scar trajectories) are properties of the
unit cell rather than of a particular system size.

This site is the narrative manual: it explains the storage conventions and the
main workflows, with runnable examples throughout. The source docstrings,
collected on the [API Reference](api.md) page, are the precise reference for
signatures and keyword defaults.

!!! note "Formerly the unregistered `iTEBD` package"
    `InfiniteTEBD` is the registered republication of the package previously
    distributed, unregistered, as `iTEBD`. The original
    [`iTEBD.jl`](https://github.com/jayren3996/iTEBD.jl) repository is kept as a
    compatibility shim that re-exports `InfiniteTEBD`, so existing `using iTEBD`
    code keeps working — see [Getting Started](getting-started.md).

## Choose a path

| If you want to … | Start with |
|------------------|------------|
| Build your first infinite MPS | [Getting Started](getting-started.md) → [States and Canonical Form](imps.md) |
| Evolve a state in real or imaginary time | [Time Evolution](time-evolution.md) |
| Measure energy, entropy, or overlaps | [Observables](observables.md) |
| Exploit a U(1) or Zₙ symmetry | [Symmetric infinite MPS](symmetries.md) |
| Search for low-entanglement scar trajectories | [ScarFinder Workflow](scarfinder.md) |
| Look up exact signatures and defaults | [API Reference](api.md) |

A typical reading order is Getting Started → States and Canonical Form → Time
Evolution → Observables, then ScarFinder and the API Reference as needed.

## A minimal example

The snippet below builds a random two-site unit cell with bond dimension 4,
brings it to Schmidt-canonical form, and inspects the leading Schmidt spectrum.
For a random initial state the dominant Schmidt value is close to 1 and the
rest of the spectrum is small, which is what `canonical!` exposes by rotating
the gauge into the Schmidt basis on every bond.

```@example
using InfiniteTEBD

psi = rand_iMPS(ComplexF64, 2, 2, 4)
canonical!(psi)

(;
    n_sites = length(psi.Γ),
    n_bonds = length(psi.λ),
    lambda1 = psi.λ[1],
)
```

After `canonical!`, the stored tensors `psi.Γ[i]` are right-canonical and the
entanglement structure on each bond is carried explicitly by `psi.λ[i]`. See
[States and Canonical Form](imps.md) for the full convention and
[Time Evolution](time-evolution.md) for how local gates are applied on top of
this representation.

## Package scope

`InfiniteTEBD.jl` is intentionally direct: the unit cell, the local operators,
and the truncation settings (`maxdim`, `cutoff`) are all specified explicitly.
There is no automatic Hamiltonian builder, no finite-size DMRG, and no
non-Abelian quantum-number bookkeeping. Abelian-symmetric tensors (U(1), Z_N,
and products) are available through the TensorKit-based extension documented on
the [Symmetric infinite MPS](@ref) page. The core stays small enough to read
end to end, which suits exploratory tensor-network work and custom ScarFinder
protocols.
