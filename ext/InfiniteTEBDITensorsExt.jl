module InfiniteTEBDITensorsExt

using InfiniteTEBD: iMPS
using ITensors: ITensor, Index, dim
using ITensorMPS: MPS
import InfiniteTEBD: imps2mps

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

end
