# Symmetric infinite MPS

`InfiniteTEBD.jl` ships with an optional symmetric backend that lets you exploit
Abelian conservation laws — total `Sz`, particle number `N`, parity, or any
combination of these — without ever leaving the `iMPS` API you already know.

This page assumes you have run the dense examples in
[States and Canonical Form](imps.md) and now want to take advantage of a
symmetry your model has.

Use TensorKit 0.17.3 or newer in the 0.17 series for a stock installation.
TensorKit 0.16.5 can fail while repartitioning fusion trees on Julia 1.10
and 1.12. The documentation examples use the 0.17 series.

!!! tip "No TensorKit experience needed"
    You do not need any prior experience with TensorKit. The sections below
    introduce charges, flux, and arrow conventions from zero.

## What you gain

For the spin-1/2 XXZ chain, total `Sz` is conserved. If you tell `InfiniteTEBD.jl`
this, every internal tensor splits into independent blocks labelled by `Sz`,
so at a fixed bond dimension `χ` only about `1/√χ` of the dense data still
needs to be stored — a memory saving that grows like `√χ`. The numerical
answer is unchanged; the runtime and memory both shrink.

For a multi-site symmetric cell, `canonical!` starts the transfer solve
at the bond with the smallest neutral operator space. It tests the identity
and stored Schmidt environments against the relative transfer residual
before starting Krylov iteration. It propagates the results back to the
wraparound bond and uses an untruncated block-sparse LQ sweep followed by
left-to-right SVD truncation. The physical split keeps its original starting
bond and avoids forming the tensor with all of the cell's physical legs.

[`restore_canonical!`](@ref) also supports injective TensorKit states with
full numerical Schmidt support. It keeps every graded bond space and
normalizes the state per cell. It propagates the left and right transfer
environments site by site and checks the resulting gauges before updating
the state. Use `canonical!` when the bond spaces need truncation. The
TensorKit restoration method uses `method=:gram`.

Both functions assume a unique physical transfer fixed point. For a small
graded bond, `noninjective=:warn` or `:error` inspects the peripheral
transfer eigenvalues that act on symmetry-preserving environments. This
diagnostic covers neutral operator spaces of dimension at most 32.

## Sectors, charges, graded spaces

A *charge* (synonyms: *sector*, *irrep*, *QN*) is a label that lives on a
basis vector. For spin-1/2 the physical leg has two basis vectors, one with
`Sz = +1/2` and one with `Sz = -1/2`. We use the integer convention `2*Sz`
throughout, so the charges are `+1` and `-1`.

A *graded vector space* is a regular vector space whose basis vectors each
carry a charge. The spin-1/2 physical leg is the graded vector space
`P = (Sz=+1) ⊕ (Sz=-1)`. In `InfiniteTEBD.jl` you build it with:

```julia
using InfiniteTEBD
using TensorKit                # loads the symmetric extension
P = graded_space(:U1, 1=>1, -1=>1)
```

The pair `1=>1` reads "the `+1` charge has dimension 1 (one basis state)";
similarly for `-1=>1`.

## Flux

Every symmetric tensor has a *flux*: the net charge it adds (or removes) to a
state when it acts. For Abelian symmetries the flux is a single integer (or
tuple, for product symmetries).

The intuition:

- **Flux = 0** — the tensor doesn't move charge. Most operators you care
  about have flux 0: the identity, projectors, conserved quantities like
  `Sz` or particle number, Hamiltonian densities, and the MPS tensors of a
  state in a fixed charge sector.
- **Flux ≠ 0** — the tensor moves charge by that amount. `S+` adds +2 to
  `Sz` (in our `2·Sz` integer convention) so it has flux **+2**. `S-` has
  flux **−2**.

When you multiply or contract two symmetric tensors, the flux must "match
up" the way charges do on the contracted legs — otherwise the result is
zero or TensorKit raises a `SpaceMismatch` error. Adding two tensors with
different fluxes is also forbidden.

Worked example. With `Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)`:

| Operator | Flux | Why |
|---|---|---|
| `Sz`       | 0  | diagonal, doesn't move spin |
| `SzSz`     | 0  | two-site, both legs flux 0 |
| `SpSm`     | 0  | composite: `+2` on site 1 cancels `-2` on site 2 |
| `SmSp`     | 0  | symmetric to above |

**Practical consequence for `InfiniteTEBD.jl`:** the two-site Hamiltonian density
`h = Sz⊗Sz + (1/2)(S+⊗S- + S-⊗S+)` is meaningful because every term has
total flux 0: `Sz⊗Sz` is two flux-0 operators on adjacent sites; `S+⊗S-`
pairs a +2 with a −2, summing to 0; same for `S-⊗S+`. `InfiniteTEBD.jl`'s
`spin_half_ops(:U1)` returns these pre-assembled two-site terms
(`SzSz`, `SpSm`, `SmSp`) so you can add them and `exp` them freely. The
individual single-site `S+` and `S-` are not returned because composing
them naively (`S+ ⊗ S-` as a four-leg operator) requires extra "sided
operator" machinery that lives outside the v1 helper layer.

For most users of this package, "operators have a flux; flux-0 things
compose and add cleanly" is all you need to know. The next section shows how
the same bookkeeping applies to the MPS tensors themselves.

## How charge flows through a site, and the wraparound bond

Each MPS tensor `Γ[n]` has three legs: a **left bond**, a **physical** leg,
and a **right bond**. They are laid out so that charge flowing *in* through
the left bond and the physical leg must balance the charge flowing *out*
through the right bond:

```text
              │ physical   (charge +1 or −1 for spin-1/2)
              │
    left ─────●───── right
    bond              bond
```

For a flux-0 tensor that balance is exact at every site:

```text
    charge(left bond) + charge(physical) = charge(right bond)
```

so the bond charge simply accumulates the physical charges as you walk along
the chain. In the `Sz=0` Néel cell `product_iMPS(:U1, [-1, 1], [1, -1])` the
bond charge runs `0 → +1 → 0`: site 1 puts `+1` on its physical leg and
raises the bond to `+1`, site 2 puts `−1` and brings it back to `0`.

**Closing around the unit cell.** The lattice is infinite, so the right bond
of the last site is glued to the left bond of the first. Those two graded
spaces must be *identical* — the same charges with the same per-charge
dimensions, not merely equal in total dimension. Equivalently, the physical
charges around one unit cell must sum to the flux you intend (`0` for a
fixed-`Sz` state). When they do not, construction fails with
`fluxes must close around the unit cell` (see Common errors below). Choosing
the bond spaces so the running charge returns to its starting value is what
"build the wraparound space explicitly" means.

If you want to dig into the underlying formalism — the codomain/domain split,
dual spaces, and the arrow diagrams TensorKit uses to track all of this — see
the [TensorKit manual](https://quantumkithub.github.io/TensorKit.jl/stable/man/sectors/).

## End-to-end walkthrough: spin-1/2 XXZ in the Sz=0 sector

```@example xxz
using InfiniteTEBD, TensorKit

# Build the U(1)-symmetric spin-1/2 operators (pre-assembled two-site forms)
Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)

# Heisenberg density h = Sz⊗Sz + (1/2)(S+⊗S- + S-⊗S+)
h = SzSz + 0.5 * (SpSm + SmSp)

# Néel initial state in the Sz=0 sector
ψ = product_iMPS(:U1, [-1, 1], [1, -1])

# Imaginary-time iTEBD, two-site Trotter. The full converged run uses ~400
# steps; here we keep it short so docs build quickly.
dt = 0.05
gates = [(exp(-dt * h), 1, 2), (exp(-dt * h), 2, 1)]
evolve!(ψ, gates, 60; maxdim=16, cutoff=1e-10)

# Energy density (a full 400-step run converges to the Bethe-ansatz value
# 1/4 - log(2) ≈ -0.4431).
energy_density(ψ, h)
```

For Abelian nearest-neighbour gates, `truncerr` targets the fraction of
squared singular values discarded across the entire cut, including all
charge sectors. `maxdim` caps the total dimension. When using discarded-weight
truncation, `mindim` sets the minimum and can keep values below
`svd_min` (also called `cutoff`). Use `return_stats=true` to inspect the
required rank, actual kept rank and discarded weight.

```julia
ψ, stats = evolve!(ψ, gates, 1; maxdim=16, truncerr=1e-8, return_stats=true)
stats.max_discarded_weight
```

`applygate!(ψ, G, i, j; gauge=:gram)` restores the full graded gauge after
one gate, and `evolve!(ψ, gates, steps; gauge=:gram)` restores it after
each gate-list sweep. Both keep the bond spaces produced by the local gate
cuts. `recanonicalize=true` performs the legacy wraparound
canonicalization, which can truncate other bonds as well. Gate statistics
describe the local cut.

When using a low-level fixed-cap gate with a non-Abelian TensorKit state,
`observe` receives `stats=nothing` at each sample. Cut statistics are available
for Abelian bond spaces.

```@example xxz
# Schmidt spectrum on bond 1 — `schmidt_values` flattens all sectors
# into a single descending-sorted Vector{Float64}. To inspect the
# per-sector blocks instead, iterate `blocks(ψ.λ[1])` directly.
schmidt_values(ψ, 1)
```

## Common errors

- `DimensionMismatch: ... fluxes must close around the unit cell` — the
  right virtual space of `Γ[n]` differs from the left virtual space of
  `Γ[1]`. In practice this happens when the running flux around the unit
  cell does not return to zero. Either set every tensor's flux to 0, or
  build the wraparound space explicitly.
- *"This entry point requires the TensorKit backend. Run `using TensorKit`
  …"* from a `graded_space`, `spin_half_ops`, or symmetric
  `rand_iMPS`/`product_iMPS` call — the symmetric API lives in a package
  extension that loads only once TensorKit is imported. Run `using TensorKit`
  (or add it to your project) and retry.
- `graded_space(:ZN, N, …) requires N ≥ 2 (got 1)` — Z_1 is trivial;
  use `:Trivial` instead.
- `ArgumentError: canonical!: inconsistent left/right transfer rates` —
  the fixed-point solves disagree on the dominant transfer rate. Check
  convergence and whether one physical sector dominates. Degenerate
  peripheral eigenvalues can coexist with matching transfer rates.

## See also

- [MPSKitModels.jl](https://github.com/QuantumKitHub/MPSKitModels.jl) for
  ready-made symmetric Hamiltonians (Heisenberg, Hubbard, ...) that work
  unchanged with this backend.
- [TensorKit.jl manual](https://quantumkithub.github.io/TensorKit.jl/stable/)
  for the underlying symmetric-tensor library.
