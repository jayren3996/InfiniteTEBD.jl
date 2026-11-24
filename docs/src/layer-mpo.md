# Layer MPOs

Gate-by-gate evolution with [`evolve!`](@ref) needs a unit cell at least as
long as the gates: a three-site term needs three sites. When the physics has a
shorter period (for instance two sites for a staggered chain), the longer cell
is a repeated smaller cell, and truncating it bond by bond, as the SVD after each gate
does, singles out some bonds. That can seed a spurious doubling of the cell, which an iterated
projection such as ScarFinder then amplifies.

Layer MPOs remove the need for the longer cell.

## The construction

Split the Hamiltonian into layers whose terms commute wherever they overlap.
For the PXP chain with a staggered potential,

```math
H = \sum_j P_{j-1} X_j P_{j+1} + \sum_{j \in A} (\mu + \delta) n_j + \sum_{j \in B} (\mu - \delta) n_j ,
```

split the terms by the parity of their centre. Terms centred on A sites overlap
only on B sites, where they act diagonally, so they commute; the same holds
for B. A second-order step is then ``U_A(dt/2)\, U_B(dt)\, U_A(dt/2)``.

Each layer ``U_A`` is a product of commuting local gates, and
[`layer_mpo`](@ref) writes it as an exact infinite MPO: every gate is split by
an operator Schmidt decomposition (an SVD across each internal bond), and
pieces that share a site are multiplied. If each bond of the cell is crossed
by at most one gate, the MPO bond dimension equals the operator Schmidt rank of
that gate. A PXP gate flips the centre only if both neighbours are empty, so
the rank is 2 on every bond, with or without a blockade projector folded in.

## Supports longer than the cell

A layer is a list of `(G, i, j)` as for [`trotter_gates`](@ref), with one
extension: `(i, j)` with `i ≤ j` are chain sites `i, i+1, …, j`, which may be
more than the cell holds. Site `k` lives on cell site `mod1(k, n)`, and every
term is repeated with period `n`. For a three-site term centred on `c`, write
`(c - 1, c + 1)`; any shift by a multiple of `n` names the same term. In a
two-site cell:

| term | support | cell sites |
|------|---------|------------|
| centred on A = site 1 | `(2, 4)` (or `(0, 2)`) | 2, 1, 2 |
| centred on B = site 2 | `(1, 3)` | 1, 2, 1 |

The legacy wraparound form `j < i` still means `j + n`.

The precondition is checked: [`check_commuting`](@ref) compares every pair of
overlapping gates, including a gate and its own translates, and
[`layer_mpo`](@ref) throws an `ArgumentError` if a bond is crossed by two
gates. Layers whose overlapping gates do not commute are out of scope.

```julia
using InfiniteTEBD, LinearAlgebra
X = [0 1; 1 0]; P = [1 0; 0 0]; n = [0 0; 0 1]; I2 = Matrix(1.0I, 2, 2)
h(c) = kron(P, X, P) + c * kron(I2, n, I2)
layers = [[(h(1.2), 2, 4)], [(h(0.6), 1, 3)]]
U = trotter_mpos(layers, 0.05; n=2)   # [U_A(dt/2), U_B(dt), U_A(dt/2)]
bonddims.(U)                          # [[2, 2], [2, 2], [2, 2]]
```

`gate_map` modifies every exponentiated gate before it is split, e.g.
`gate_map = G -> P3 * G` folds the three-site blockade projector into each gate.

## Applying MPOs

[`apply!`](@ref) follows Orús and Vidal
([PRB 78, 155117](https://arxiv.org/abs/0711.3960), Sec. III): contract the
MPO into the stored tensors `B = Γλ` (bond dimensions multiply), restore the
canonical form, and truncate every bond at once to its own largest Schmidt
values (`maxdim`, `truncerr`, `cutoff`). It works for non-unitary MPOs:
imaginary-time layers, projectors.

The canonical form is not computed from transfer-matrix fixed points as in
[`canonical!`](@ref). Those carry squared Schmidt values, so precision drops to
about `√ε` and the gauge built from their square roots is ill-conditioned
(Vanderstraeten, Haegeman and Verstraete,
[SciPost Phys. Lect. Notes 7](https://arxiv.org/abs/1810.07006), Sec. 2.3).
`apply!` uses their single-layer algorithm instead: iterate
`(A_L, L) ← QRPos(L A)` around the cell until the seam factor is stationary,
do the same from the other side, and diagonalize the bond centers by SVD
(their Algorithms 1 and 2, with the Arnoldi step kept as a fallback for slow
convergence). Schmidt values are resolved to about `1e-15` of the largest.
The enlarged bond often has exact null directions, which would make the QR
factor non-unique; they are first removed by rank-revealing SVD sweeps that
rewrite the cell exactly with smaller bonds.

Use `orth_tol` on `apply!`, MPO `evolve!` or `truncate!` to set the
tolerance on the change in the normalized seam factor between sweeps.
The default targets `1e-13` and accepts a final residual up to `1e-10` if the
generic sweep stalls or reaches its iteration cap. An explicit value, such
as `orth_tol=1e-11`, is enforced on both sweeps. If either sweep misses it,
the generic update raises `CanonicalFormError` with the measured residual
and the requested tolerance.

Inside [`evolve!`](@ref), truncated intermediate states are only renormalized,
as in Orús and Vidal; the next application restores the canonical form, and
the state returned is canonical.

[`evolve!`](@ref)`(ψ, mpos, steps)` applies the step sequence repeatedly and
merges the last MPO of one step with the first of the next
(`U_A(dt/2) U_A(dt/2) = U_A(dt)`) through [`mpo_multiply`](@ref) and
[`compress_mpo`](@ref), when the merged MPO is no larger.

## Fast path for PXP-type layers

A PXP gate is controlled by its two neighbours: it acts on the center only
through blocks `M_ab` labelled by the outer occupations `a, b`
(`G = Σ_ab |a⟩⟨a| ⊗ M_ab ⊗ |b⟩⟨b|`). With `P3` folded in, `M_00` is the 2×2
gate `exp(−i dt (X + c n))` and every other block keeps only the empty center.
[`layer_mpo`](@ref) recognizes such layers (one three-site gate on a two-site
cell, local dimension 2, diagonal in the outer sites) and stores the blocks in
the MPO. [`apply!`](@ref) then factorizes each occupation block of the control
tensor, so the control site becomes a block identity, builds the center tensor
from the blocks `M_ab`, and runs the same QR/LQ canonicalization with the
control-site factorizations split by sector, starting from the input Schmidt
data. The result is the same state as the generic update, truncated by the same
code (agreement to ~1e-13 in Schmidt values and ~1e-15 in fidelity in the
tests); for the PXP ScarFinder benchmark it is several times faster.

The fast path is used for dense complex two-site cells with `renormalize=true`,
including inputs that leak out of the blockade and intermediate states that
were not regauged. It falls back to the generic update otherwise, if its
iteration does not converge, or on any numerical failure; `stats.path`
(`apply!`) and `num_structured` (`evolve!`) report which update ran. Pass `structured=false` to force
the generic update.

## Truncation and ScarFinder

[`truncate!`](@ref) is the bond-symmetric truncation: canonical form, then
every bond keeps its own `χ` largest Schmidt values. Tiling a two-site state
onto four sites and truncating gives the tiled truncated two-site state, so no
cell doubling is generated.

For site-dependent Hamiltonians, `energy_density(ψ, terms)` takes a list of
`(h, i, j)` and returns the energy per site, and [`energy_fix!`](@ref) moves
the energy to a target with imaginary-time layer MPOs built from the same terms.
It returns `(converged, residual, steps)` instead of only warning.

[`scarfinder_step!`](@ref) with a vector of MPOs evolves `nstep` steps,
truncates with `truncate!`, fixes the energy if `target` is given, and returns
`(loss, energy_error, converged)` with `loss = -log F`, `F` the per-site
fidelity between the states before and after the truncation:

```julia
P3 = Diagonal([!(((s>>2)&1==1 && (s>>1)&1==1) || ((s>>1)&1==1 && s&1==1)) for s in 0:7])
U = trotter_mpos(layers, 0.05; n=2, gate_map=G -> P3 * G)
psi = product_iMPS(ComplexF64, [[cos(0.35), sin(0.35)], [cos(0.35), sin(0.35)]])
apply!(psi, layer_mpo([(I - kron(n, n), 1, 2), (I - kron(n, n), 2, 3)], 2))   # blockade
terms = reduce(vcat, layers)
e = energy_density(psi, terms)
for k in 1:30
    loss, err = scarfinder_step!(psi, U, 2; nstep=20, maxdim=16, terms, target=e)
end
```

### Blockade leakage

`constraint_leakage(ψ, Blockade())` computes the per-site leakage from the
probability `⟨ψ|P|ψ⟩`. Its positive transfer uses two environments labelled
by the previous site's occupation. The sum of non-negative terms retains the
relative accuracy of small weights, and the result is invariant under tiling
and rescaling. `iMPO(Blockade(), n)` constructs the projector for an `n`-site
cell. Use `constraint_leakage(ψ, P)` with a general projector MPO.

### Several iterations

`scarfinder!(ψ, mpos, χ, N; refine=false, kwargs...)` runs exactly `N` calls of
`scarfinder_step!(ψ, mpos, χ; kwargs...)` with the same options.
Pass `history=Any[]` to keep the named tuple of each iteration. With `refine=true` it then scans `refine_step`
further steps for the minimum-entanglement point.

## Annihilation and the fast path

`apply!` throws `MPOAnnihilationError` only when the norm of `W·ψ` per cell is at
the roundoff level of the contraction, measured against the same contraction
with entrywise absolute values `|W|·|ψ|`. Identities and 0/1 projectors therefore
never reject a finite (even very small) weight, for any cell length.

The controlled-layer structure is read from the stored MPO tensors, so it
follows a coarse construction `cutoff`. Editing a tensor in place through `W[i]`
disables the fast path for that MPO, and for its copies and merged products;
the generic update is then used.

## Limitations

- Overlapping gates in one layer must commute, and each bond may be crossed by
  one gate only (so a gate spans at most `n + 1` sites); otherwise the layer
  must be split.
- The local dimension is uniform within a layer.
- `apply!`, `truncate!` and the MPO `evolve!` are implemented for the dense
  backend.
- The QR sweeps converge at the rate of the transfer-matrix gap; states with
  long correlation lengths need more sweeps (the Arnoldi fallback then helps).
- Per Trotter step the MPO path is slower than gate-by-gate evolution on a
  four-site cell (each application canonicalizes a bond of dimension `2χ`);
  see `bench/bench_layer_mpo.jl`.
