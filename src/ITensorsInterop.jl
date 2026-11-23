# Methods live in ext/InfiniteTEBDITensorsExt.jl and load with `using ITensorMPS`.
"""
    imps2mps(ψ, sites; L=length(sites))

Convert an `iMPS` into a length-`L` `ITensorMPS.MPS` window by unrolling the
periodic unit cell across the supplied physical-site index collection. Site `i`
of the output borrows its tensor from `ψ.Γ[mod1(i, ψ.n)]`, so the unit cell is
repeated `L ÷ ψ.n` times.

Parameters:
- `ψ`
  Source `iMPS`. Read-only; not mutated. The boundary contract below assumes
  `ψ` is in the package's Schmidt-canonical form (see [`canonical!`](@ref)).
- `sites`
  Collection of `ITensors.Index` objects of length at least `L`, one per
  physical site in the unrolled chain.

Keyword arguments:
- `L=length(sites)`
  Number of physical sites in the output `MPS`. Must be `>= 0`, no greater
  than `length(sites)`, and a multiple of the unit-cell length `ψ.n`
  (otherwise an `ArgumentError` is thrown): the Schmidt bases of an iMPS only
  match at unit-cell boundaries, so a window covering a partial unit cell
  would glue mismatched Schmidt bases at its edge.

Returns:
- A finite `ITensorMPS.MPS` of length `L` representing a window of the
  infinite state, with the following boundary contract:
  * Left edge: the Schmidt vector `ψ.λ[end]` on the bond entering site 1 is
    contracted (capped) into the first tensor, which carries an extra
    dimension-1 marker index tagged `"Link,l=0"`. This weights the window by
    the left half-infinite environment's Schmidt spectrum.
  * Right edge: the last tensor keeps a dangling bond index of the full bond
    dimension, tagged `"Link,l=L"`. Because the stored tensors are
    right-canonical (`B_i = Γ_i λ_i`), this dangling leg is the exact
    entanglement purification of the right half-infinite environment.
  * For a canonical input, `ITensorMPS.inner(mps, mps) ≈ 1` exactly: the
    dangling right legs of bra and ket contract with each other, the
    right-canonical tensors telescope to the identity, and the left cap
    closes the sum as `Σ λ² = 1`.

Notes:
- The output works with downstream `ITensorMPS` analysis (`inner`, `norm`,
  `ITensorMPS.expect`, `ITensorMPS.correlation_matrix`, `orthogonalize!`,
  ...). Note that this package exports its own [`expect`](@ref) for `iMPS`, so
  qualify `ITensorMPS.expect` when both are in scope.
- Boundary accuracy: the right environment of the window is exact, but the
  left cap replaces the exact environment `diag(λ²)` by the pure boundary
  condition `|λ⟩⟨λ|`. Observables therefore carry a left-edge artifact that
  decays into the bulk with the subdominant eigenvalue of the transfer matrix
  (e.g. `(1/3)^j` for AKLT at site `j`); expectation values away from the
  left edge match the infinite-state values, and sites near the right edge
  are essentially exact. Choose `L` a few correlation lengths larger than the
  region you analyze.
- A fully dangling (dimension-χ) left edge would make every site exact, but
  `ITensorMPS.correlation_matrix` cannot evaluate an MPS whose first tensor
  carries a nontrivial dangling leg (its left-environment contraction leaves
  the leg uncontracted and fails the scalar extraction), so the capped left
  edge is used instead.
- Bond dimensions of `ψ.Γ` must match the supplied `sites` (physical leg) and
  the unit-cell wraparound (virtual legs). Mismatches throw `ArgumentError`.
"""
function imps2mps end
