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
function imps2mps(ψ::iMPS, s; L::Integer=length(s))
    L >= 0 || throw(ArgumentError("L must be nonnegative"))
    length(s) >= L || throw(ArgumentError("site index collection must contain at least L indices"))
    L == 0 && return MPS(ITensor[])
    L % ψ.n == 0 || throw(ArgumentError(
        "window length L = $L must be a multiple of the unit-cell length n = $(ψ.n): " *
        "the Schmidt bases of an iMPS only match at unit-cell boundaries, so a window " *
        "covering a partial unit cell would glue mismatched Schmidt bases at its edge"))

    sites = collect(Iterators.take(s, L))
    links = [Index(size(ψ.Γ[mod1(i, ψ.n)], 3), "Link,l=$i") for i in 1:L]
    # Fresh dimension-1 marker index for the capped left boundary. It must be
    # distinct from every other Index so the first tensor never self-contracts,
    # and it must have dimension 1 so that downstream ITensorMPS routines
    # (e.g. `correlation_matrix`) can still extract scalars when their
    # environment contractions leave the boundary legs open.
    boundary_left = Index(1, "Link,l=0")
    tensors = Vector{ITensor}(undef, L)

    for i in 1:L
        Γ = ψ.Γ[mod1(i, ψ.n)]
        size(Γ, 2) == dim(sites[i]) ||
            throw(ArgumentError("site index dimension does not match iMPS physical dimension at site $i"))
        right = links[i]
        if i == 1
            # Cap the left edge with the Schmidt spectrum on the wraparound
            # bond: capped[s, 1, r] = Σ_l λ[end][l] · Γ[l, s, r]. Together with
            # the right-canonical stored tensors this makes the window state
            # normalized (Σ λ² = 1) and weighted by the left environment.
            λ = ψ.λ[end]
            Dl, d, Dr = size(Γ)
            length(λ) == Dl || throw(ArgumentError(
                "iMPS wraparound Schmidt vector length $(length(λ)) does not match " *
                "the left bond dimension $Dl of site 1"))
            capped = reshape(transpose(reshape(Γ, Dl, d * Dr)) * λ, d, 1, Dr)
            # The site index is placed first so that ITensorMPS's
            # first-unique-index heuristic (`siteind(mps, 1)`) finds the
            # physical index rather than a boundary index.
            tensors[i] = ITensor(capped, sites[i], boundary_left, right)
        else
            left = links[i - 1]
            size(Γ, 1) == dim(left) && size(Γ, 3) == dim(right) ||
                throw(ArgumentError("iMPS bond dimensions are incompatible at site $i"))
            tensors[i] = ITensor(Γ, left, sites[i], right)
        end
    end

    return MPS(tensors)
end
