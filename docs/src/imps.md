# States and canonical form

This page covers four things:

1. what an `iMPS` actually stores in memory,
2. the absorbed-Schmidt convention used throughout the package (the most common source of confusion when comparing with the literature),
3. what [`canonical!`](@ref) does to a state,
4. when each truncation keyword matters in practice.

## What an `iMPS` stores

An `iMPS` holds one periodic unit cell of an infinite matrix-product state with three fields:

- `ψ.Γ` — a vector of three-leg local tensors, one per site of the unit cell.
- `ψ.λ` — a vector of Schmidt spectra, one per bond. `λ[i]` lives on the bond between site `i` and site `i+1`, with periodic wraparound at the end of the unit cell.
- `ψ.n` — the unit-cell length.

The Schmidt spectra are real and non-negative; the local tensors carry the element type of the state (`ComplexF64` by default).

## The absorbed-Schmidt convention

The single rule to remember:

> **The stored tensor has the right Schmidt values already multiplied in.**

Formally, after canonicalization,

```math
B_i = \Gamma_i \, \lambda_i,
```

where `Γ_i` is the bare Vidal tensor of site `i` and `λ_i` is the Schmidt spectrum on the bond to its right. The package stores `B_i` (right-canonical), not `Γ_i`:

- `ψ.Γ[i]` returns `B_i`.
- `ψ.λ[i]` returns `λ_i`.

So the symbol `Γ` in the struct field is a stored-tensor label and does not refer to the bare Vidal `Γ_i` from Vidal 2007. Most internal contractions ([`expect`](@ref), the transfer-matrix machinery used by `gtrm`, and the overlap in [`inner_product`](@ref)) work directly with `B_i`; reconstructing `Γ_i` is only needed when matching formulas in the literature.

## Recovering the bare Vidal tensor

Indexing returns the Vidal pair:

```julia
Γ_i, λ_i = ψ[i]
```

This divides the absorbed Schmidt values back out of `ψ.Γ[i]` on the fly and returns a fresh copy of both `Γ_i` and `λ_i` (mutating them does not affect the state). Two related but distinct objects therefore coexist:

- `ψ.Γ[i]` — stored right-canonical `B_i`,
- `ψ[i][1]` — bare Vidal `Γ_i`.

## What `canonical!` does

```julia
canonical!(ψ; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)
```

Dense cells use the stored wraparound bond as the initial cut. When a
fused prefix would exceed 1 MiB or the physical dimensions vary by site,
the implementation applies transfer maps site by site and builds an
untruncated right-isometric suffix using LQ. The subsequent left-to-right
SVD sweep starts at the same seam and sees the same nonzero cut spectra
as the grouped implementation, up to rounding. If a cutoff crosses a
degenerate spectrum, the retained basis can depend on the decomposition.
A second pass restores the gauge when truncation changes a bond dimension.

For a multi-site cell with `renormalize=false`, the split leaves the
overall scale of the retained state in the tensors. A cutoff or rank cap
can reduce that scale. With no truncation the per-cell norm is preserved.
The one-site path includes its historical final normalization. The
streaming path checks the transfer spectrum on small virtual bonds and
uses a diagonal-sector diagnostic on larger bonds. TensorKit has its
own canonicalization implementation.

[`restore_canonical!`](@ref) restores the gauge of an injective dense iMPS at
fixed bond dimensions. The default Gram method sweeps the left and right
transfer environments site by site, starts from a smallest virtual bond,
and normalizes the state per cell. It requires numerically invertible right
environments on the stored bonds.

For an ill-conditioned basis, call `restore_canonical!(ψ; method=:qr)`.
This iterates local LQ and QR factorizations until the virtual carry closes
across the seam. It obtains Schmidt values from SVDs on the virtual-bond
centers. Both methods avoid grouping physical legs. The QR method requires
full support at the stored bond dimensions. Use `tol` and `maxiter` to
control the seam solve. Mixed-transfer Arnoldi can propose a seam carry
when the sweeps are slow. Set `accelerate=false` for plain sweeps.
A failed solve leaves the original state intact.

For dense states, `restore_canonical!(ψ; method=:auto)` tries Gram on a
candidate state. It checks both the right-isometry and left Schmidt
residuals before accepting that candidate, then retries with QR from the
original state if that check or the Gram numerical solve fails. A small
seam is also checked for multiple peripheral transfer eigenvalues. The
default is `method=:gram`.

To compress every bond to `χ`, use [`truncate!`](@ref).

!!! warning "Injective setting"
    This canonicalization path assumes a non-degenerate transfer spectrum. For
    symmetry-broken or block-diagonal states (GHZ-like, spontaneously broken
    `Z_2`, …) see the `noninjective` and `symmetry_break` keywords in
    [`canonical!`](@ref).

## Choosing the keywords

For a healthy injective state the defaults are fine and you can ignore the keywords. Reach for them when:

- `maxdim` — lower it only when you are willing to accept truncation error in exchange for a smaller bond dimension. The default cap `MAXDIM` is large enough that physical states usually saturate `cutoff` first.
- `cutoff` — tighten it (e.g. `1e-14`) to discard tiny Schmidt tails for cleaner entanglement spectra; loosen it (e.g. `1e-8`) when speed matters more than the last few digits.
- `renormalize` — keep `true` for normal use. After non-trivial truncation the retained spectrum is rescaled to unit norm, which is what almost every downstream routine expects.

!!! warning "Truncation changes the state"
    An aggressive `cutoff` or small `maxdim` produces a genuinely truncated
    state. The bare Vidal tensor reconstructed by `ψ[i]` afterwards is the Vidal
    tensor *of that truncated state*, not of the original.

## Constructors

```julia
iMPS(Γs; renormalize=true)
iMPS(T, Γs; renormalize=true)
```

Both constructors interpret `Γs` as raw local tensors (any shape, any normalization). They populate the Schmidt vectors with all-ones placeholders and then, by default, call [`canonical!`](@ref) so the result obeys the storage convention immediately. Pass `renormalize=false` only when you intend to set up the canonical structure yourself; higher-level routines expect the canonical convention to hold.

[`rand_iMPS`](@ref) and [`product_iMPS`](@ref) are convenience constructors that always return a canonicalized state.

## Example: random state, both views of site 1

```@example states
using InfiniteTEBD

psi = rand_iMPS(ComplexF64, 1, 2, 4)
canonical!(psi; maxdim=4, cutoff=1e-12, renormalize=true)

stored_B = psi.Γ[1]
Gamma1, lambda1 = psi[1]

(;
    stored_size  = size(stored_B),
    vidal_size   = size(Gamma1),
    lambda       = lambda1,
    reconstructed = isapprox(reshape(Gamma1, :, length(lambda1)) .* lambda1',
                              reshape(stored_B, :, length(lambda1));
                              atol=1e-12),
)
```

`stored_B` is what lives in the struct; `Gamma1` is what you would write down in a paper. The last entry checks the convention explicitly: multiplying `Gamma1` by `lambda1` on the right reproduces `stored_B`.

## Example: product state has Schmidt rank 1

A Néel-like `Z_2` product state on a two-site unit cell has a single non-zero Schmidt value on every bond:

```@example states
using InfiniteTEBD

up   = ComplexF64[1, 0]
down = ComplexF64[0, 1]
neel = product_iMPS([up, down])

(;
    bond_dims = length.(neel.λ),
    lambdas   = neel.λ,
)
```

Both `λ` vectors have length 1 and entry 1.0: product states factorize across every cut, so their Schmidt spectrum is trivial. The same `λ` would survive any subsequent gate application followed by [`canonical!`](@ref), as long as the gate preserves the product structure.

## References

For the iTEBD algorithm and Schmidt canonical form, the standard references are:

- G. Vidal, [Classical Simulation of Infinite-Size Quantum Lattice Systems in
  One Spatial Dimension](https://link.aps.org/doi/10.1103/PhysRevLett.98.070201),
  Physical Review Letters 98, 070201 (2007).
- R. Orús and G. Vidal, [Infinite Time-Evolving Block Decimation Algorithm
  Beyond Unitary Evolution](https://link.aps.org/doi/10.1103/PhysRevB.78.155117),
  Physical Review B 78, 155117 (2008).
- U. Schollwöck, [The Density-Matrix Renormalization Group in the Age of Matrix
  Product States](https://doi.org/10.1016/j.aop.2010.09.012), Annals of Physics
  326, 96-192 (2011).

Full signatures for the constructors and canonicalization routines are on the [API Reference](api.md) page.

## Symmetric variant

If your model conserves `Sz`, particle number, parity, or any Abelian
combination of those, the same `iMPS` API works on top of an optional
[TensorKit.jl](https://github.com/Jutho/TensorKit.jl)-backed symmetric
infrastructure. See [Symmetric infinite MPS](symmetries.md) for a
walkthrough that explains charges, flux, and arrow conventions from zero.
