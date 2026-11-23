#---------------------------------------------------------------------------------------------------
# Entropy
#---------------------------------------------------------------------------------------------------
export entanglement_entropy, natural_bonddim, adaptive_bonddim

"""
    entanglement_entropy(p; cutoff=1e-10)

Compute the von Neumann entropy `-sum(p .* log.(p))` of a probability vector
or Schmidt spectrum `p`, ignoring entries smaller than `cutoff`.

Parameters:
- `p`
  Probability vector or Schmidt-weight vector. The function normalizes `p` by
  `sum(p)` before applying the entropy formula.

Keyword arguments:
- `cutoff=1e-10`
  Entries not exceeding `cutoff` are ignored to avoid spurious contributions
  from numerical noise.

Returns:
- The scalar von Neumann entropy.

Example:
```julia
S = entanglement_entropy([0.5, 0.5])
```
"""
function entanglement_entropy(
    S::AbstractVector;
    cutoff::AbstractFloat=1e-10
)
    EE = 0.0
    total = sum(S)
    if total > 0 && isfinite(total)
        for si in S
            p = si / total
            if p > cutoff
                EE -= p * log(p)
            end
        end
    end
    EE
end

"""
    natural_bonddim(λ; q=1.0, alpha=0.1, cutoff=1e-12)

Return a smooth spectrum-derived estimate of the intrinsic bond dimension
associated with Schmidt values `λ`.

The score is based on the Rényi effective rank of order `q` on the normalized
weights `p_i = abs2(λ_i) / sum(abs2, λ)`, optionally amplified by the total
tail weight `1 - p_1` (where `p_1` is the largest normalized weight). The
boundary case `q = 0` reduces to the support-count rank (number of
above-cutoff Schmidt values); `q = 1` is the entropy rank; `q = 2` is the
participation-ratio rank. At `q = Inf` the rank is the inverse of the largest
normalized weight before the tail correction.
"""
function natural_bonddim(
    λ::AbstractVector;
    q::Real=1.0,
    alpha::Real=0.1,
    cutoff::Real=1e-12
)
    return _natural_bonddim(λ, q, alpha, cutoff)
end

_above_weight_cutoff(value, cutoff) = iszero(cutoff) ? value > 0 : abs2(value) > cutoff

function _natural_bonddim(λ, q, alpha, cutoff)
    q >= 0 || throw(ArgumentError("q must be non-negative"))
    alpha >= 0 || throw(ArgumentError("alpha must be non-negative"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative"))

    peak = 0.0
    total = 0.0
    support = 0
    for value in λ
        magnitude = Float64(abs(value))
        isfinite(magnitude) || throw(ArgumentError("Schmidt values must be finite"))
        _above_weight_cutoff(magnitude, cutoff) || continue
        support += 1
        # Accumulate the squared ratios relative to the largest value seen.
        if magnitude > peak
            total = 1 + total * abs2(peak / magnitude)
            peak = magnitude
        else
            total += abs2(magnitude / peak)
        end
    end
    support <= 1 && return 1.0
    isinf(alpha) && return floatmax(Float64)

    rank = Float64(support)
    if !isapprox(q, 0; atol=sqrt(eps(Float64)))
        shannon = isapprox(q, 1.0; atol=sqrt(eps(Float64)))
        power = 2.0 * q
        moment = 0.0
        for value in λ
            magnitude = Float64(abs(value))
            _above_weight_cutoff(magnitude, cutoff) || continue
            ratio = magnitude / peak
            if shannon
                p = abs2(ratio) / total
                p > cutoff && (moment -= p * log(p))
            else
                moment += iszero(ratio) ? exp(power * (log(magnitude) - log(peak))) : ratio^power
            end
        end
        # The largest ratio contributes one even at large q. Rearrange the
        # logarithm to avoid multiplying log(total) by q.
        logtotal = log(total)
        entropy = shannon ? moment : logtotal + (log(moment) - logtotal) / (1.0 - q)
        rank = min(rank, exp(entropy))
    end
    return min(rank * (1 + alpha * (1 - inv(total))), floatmax(Float64))
end

"""
    adaptive_bonddim(previous, λ; mindim=1, maxdim=MAXDIM, q=1.0, alpha=0.1, cutoff=1e-12)

Return a non-decreasing bond dimension by ratcheting the smooth intrinsic
estimate from [`natural_bonddim`](@ref) between `mindim` and `maxdim`.
"""
function adaptive_bonddim(
    previous::Integer,
    λ::AbstractVector;
    kwargs...
)
    return _adaptive_bonddim(previous, λ; kwargs...)
end

function _adaptive_bonddim(
    previous::Integer,
    λ;
    mindim::Integer=1,
    maxdim::Integer=MAXDIM,
    q::Real=1.0,
    alpha::Real=0.1,
    cutoff::Real=1e-12,
    ratchet::Bool=true
)
    mindim > 0 || throw(ArgumentError("mindim must be positive"))
    maxdim >= mindim || throw(ArgumentError("maxdim must be at least mindim"))

    raw = _natural_bonddim(λ, q, alpha, cutoff)
    raw >= maxdim && return maxdim
    target = max(mindim, ceil(promote_type(Int, typeof(maxdim)), raw))
    return ratchet ? min(maxdim, max(previous, target)) : target
end

#---------------------------------------------------------------------------------------------------
# Inner Product
#---------------------------------------------------------------------------------------------------
export inner_product, mixed_logoverlap, log_per_site_fidelity
"""
    inner_product(T; tol=nothing, maxiter=nothing, strict=false)
    inner_product(T1, T2; tol=nothing, maxiter=nothing, strict=false)

Return the overlap per unit cell defined by the dominant eigenvalue of a
transfer matrix.

For two translationally invariant tensor networks or `iMPS` objects with the
same unit cell, this routine first constructs the mixed unit-cell transfer
matrix

`E_cell(T1, T2)`,

and then returns

`|λ_max(E_cell(T1, T2))|`,

the magnitude of its dominant eigenvalue. For the one-argument form
`inner_product(T)`, this reduces to the corresponding norm per unit cell,
obtained from the dominant eigenvalue of the self-transfer matrix.

Keyword arguments:
- `tol=nothing`
  Convergence tolerance forwarded to `KrylovKit.eigsolve`. `nothing` keeps the
  KrylovKit default. Tighten (e.g. `1e-14`) for high-precision overlaps,
  loosen (e.g. `1e-6`) for cheap probes.
- `maxiter=nothing`
  Maximum number of Krylov restarts forwarded to `eigsolve`. `nothing` keeps
  the default. Raise it if the warning "Krylov solver did not converge" fires
  on a state whose dominant eigenvalue is close to the next one.
- `strict=false`
  Apply the Krylov tolerance after scaling the transfer action by the size
  of its first response, and raise if the solve fails to converge. With the
  default policy it can fall back to a bounded dense solve.

Returns:
- A scalar overlap or norm per unit cell extracted from the dominant
  transfer-matrix eigenvalue.

Notes:
- This is not the ordinary finite-system inner product obtained by contracting a
  finite chain.
- Instead, it is the infinite-system quantity associated with one periodic unit
  cell of the transfer matrix.
- For `iMPS` inputs, the relevant transfer matrix is the product over the whole
  unit cell, not a single local tensor.
- The implementation returns the absolute value of the dominant eigenvalue.
- Below `dense_threshold` (default 8), the dense `gtrm + eigen` path is used
  and `tol`/`maxiter` only affect the Krylov fallback path (large transfers).
"""
inner_product(T::iMPS; kwargs...) = inner_product(T, T; kwargs...)
inner_product(mps1::iMPS, mps2::iMPS; kwargs...) =
    inner_product(mps1.Γ, mps2.Γ; kwargs...)
#---------------------------------------------------------------------------------------------------
inner_product(T::AbstractArray{<:Number, 3}; kwargs...) = inner_product(T, T; kwargs...)
function inner_product(
    T1::AbstractArray{<:Number, 3},
    T2::AbstractArray{<:Number, 3};
    kwargs...
)
    inner_product([T1], [T2]; kwargs...)
end
#---------------------------------------------------------------------------------------------------
function inner_product(Ts::AbstractVector{<:AbstractArray{<:Number, 3}}; kwargs...)
    inner_product(Ts, Ts; kwargs...)
end
function inner_product(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}};
    kwargs...
)
    abs(_dominant_chain_eigenvalue(T1s, T2s; kwargs...))
end

# A common-period contraction reuses both cells without copying their tensors.
struct _PeriodicTensorCell{T,V<:AbstractVector{T}} <: AbstractVector{T}
    tensors::V
    n::Int
    offset::Int
end
_PeriodicTensorCell(tensors, n) = _PeriodicTensorCell(tensors, n, 0)
Base.size(cell::_PeriodicTensorCell) = (cell.n,)
Base.IndexStyle(::Type{<:_PeriodicTensorCell}) = IndexLinear()
function Base.getindex(cell::_PeriodicTensorCell, i::Int)
    @boundscheck checkbounds(cell, i)
    return cell.tensors[mod1(i + cell.offset, length(cell.tensors))]
end

_rotate_tensor_cell(tensors, seam) = _PeriodicTensorCell(tensors, length(tensors), seam)
_rotate_tensor_cell(cell::_PeriodicTensorCell, seam) =
    _PeriodicTensorCell(cell.tensors, cell.n, cell.offset + seam)

"""
    log_per_site_fidelity(a, b; period=nothing, tol=nothing, maxiter=nothing,
                         strict=true, self_overlaps=(nothing, nothing))

Log of the normalized overlap amplitude **per physical site** of two dense
iMPS or two vectors of stored local tensors. Neither input is mutated or
canonicalized. If `ρₐᵦ` is the mixed transfer radius over the common cell,
and `ρₐₐ` and `ρᵦᵦ` are self-transfer radii, then the result is
`log(ρₐᵦ)/nₐᵦ - (log(ρₐₐ)/nₐ + log(ρᵦᵦ)/nᵦ)/2`. The squared fidelity per
site is the exponential of twice the result. Orthogonal states return
`-Inf`. The logarithms avoid products or powers of the radii, but the
individual transfer eigensolves still require representable radii.

For different unit-cell lengths, pass `period` as a positive common multiple.
The cells are compared at the given site registration. Only their least
common cell must be contracted.

`self_overlaps` may supply either one-argument `inner_product` value over
its own native cell, for example `(inner_product(a; strict=true), nothing)`
when repeatedly measuring against a fixed `a`. A supplied value must match
the current tensors and be positive and finite. The default solves for both.

`strict=true` raises on Krylov nonconvergence without dense fallback. A
small transfer can still use a direct dense eigensolve. The default
`tol` is `1e-14` or ten machine epsilons for lower-precision tensors.
A positive log fidelity consistent with roundoff is rounded to zero. A
larger violation indicates insufficient transfer accuracy or inconsistent
supplied overlaps.
"""
log_per_site_fidelity(a::DenseIMPS, b::DenseIMPS; kwargs...) =
    log_per_site_fidelity(a.Γ, b.Γ; kwargs...)

function log_per_site_fidelity(
    a::AbstractVector{<:AbstractArray{<:Number,3}},
    b::AbstractVector{<:AbstractArray{<:Number,3}};
    period::Union{Nothing,Integer}=nothing, tol::Union{Nothing,Real}=nothing,
    maxiter::Union{Nothing,Integer}=nothing, strict::Bool=true,
    self_overlaps::Tuple{Union{Nothing,Real},Union{Nothing,Real}}=(nothing, nothing),
)
    na, nb = length(a), length(b)
    min(na, nb) > 0 || throw(ArgumentError("the fidelity needs nonempty tensor cells"))
    ϵ = max(eps(real(float(eltype(first(a))))), eps(real(float(eltype(first(b))))))
    isnothing(tol) && (tol = max(1e-14, 10ϵ))
    n = if isnothing(period)
        na == nb || throw(ArgumentError("supply period to compare unequal cell lengths"))
        na
    else
        period > 0 && iszero(rem(period, na)) && iszero(rem(period, nb)) ||
            throw(ArgumentError("period must be a positive multiple of both cell lengths"))
        lcm(na, nb)
    end
    ca = na == n ? a : _PeriodicTensorCell(a, n)
    cb = nb == n ? b : _PeriodicTensorCell(b, n)
    mixed = inner_product(ca, cb; tol, maxiter, strict)
    norm_a, norm_b = self_overlaps
    isnothing(norm_a) && (norm_a = a === b ? mixed : inner_product(a; tol, maxiter, strict))
    isnothing(norm_b) && (norm_b = a === b ? mixed : inner_product(b; tol, maxiter, strict))
    isfinite(norm_a) && norm_a > 0 ||
        throw(DomainError(norm_a, "the first self-transfer radius must be positive and finite"))
    isfinite(norm_b) && norm_b > 0 ||
        throw(DomainError(norm_b, "the second self-transfer radius must be positive and finite"))
    isfinite(mixed) || throw(DomainError(mixed, "the mixed transfer radius must be finite"))
    la, lb, lab = log(norm_a) / na, log(norm_b) / nb, log(mixed) / n
    logf = lab - (la + lb) / 2
    if logf > 0
        roundoff = 64 * ϵ * max(1, abs(la), abs(lb), abs(lab))
        logf <= roundoff || throw(DomainError(logf, "the normalized overlap exceeds one"))
        return zero(logf)
    end
    return logf
end

"""
    mixed_logoverlap(bra, ket; derivatives=nothing)

Return the logarithmic overlap amplitude per unit cell of two raw tensor
cells. The result contains `logabs = log(abs(λ))` for the dominant mixed
transfer eigenvalue and `gap_ratio`, the second-largest modulus divided by
the largest (zero when there is only one nonzero eigenvalue, one for a zero
transfer). No state is canonicalized or normalized.

For derivatives with respect to real parameters, pass `derivatives(j)`,
which returns a tuple or vector of directional derivatives of `bra[j]`.
`gradient[j][k]` is the contribution of the site's kth direction to
`logabs`. Add the site contributions for parameters shared across the cell.
`gradient` is `nothing` when omitted, when the leading modulus is zero or
tied within square-root machine precision, or when the left and right
eigenvectors have numerically zero pairing.

This routine uses a dense eigensolve for small mixed transfers, as in fits to
low-bond-dimension analytic iMPS. The ket is held fixed. When a candidate's
norm varies, subtract half its log self-overlap from the objective and
differentiate that term as well.
"""
function mixed_logoverlap(
    bra::AbstractVector{<:AbstractArray{<:Number,3}},
    ket::AbstractVector{<:AbstractArray{<:Number,3}};
    derivatives=nothing,
)
    n = length(bra)
    n > 0 || throw(ArgumentError("the mixed transfer needs a nonempty cell"))
    n == length(ket) || throw(DimensionMismatch("bra and ket cell lengths must agree"))
    transfers = [gtrm(bra[j], ket[j]) for j in 1:n]
    product = foldl(*, transfers)
    cell = convert(Matrix{float(eltype(product))}, product)
    vals, L, R = isnothing(derivatives) ? (eigvals!(cell), nothing, nothing) :
                 _mixed_eigenvectors!(cell)
    radii = abs.(vals)
    i = argmax(radii)
    ρ = radii[i]
    ρnext = maximum((radii[k] for k in eachindex(radii) if k != i); init=zero(ρ))
    gap_ratio = iszero(ρ) ? one(ρ) : ρnext / ρ
    result = (; logabs=log(ρ), gap_ratio, gradient=nothing)
    isnothing(derivatives) && return result
    gap_ratio >= 1 - sqrt(eps(typeof(ρ))) && return result

    # A simple leading eigenvalue of a real transfer must itself be real.
    # Thus LAPACK's real eigenvector columns need no complex-pair decoding.
    l, r = L[:, i], R[:, i]
    pairing = dot(l, r)
    abs(pairing) > eps(typeof(ρ)) || return result
    right = Vector{typeof(r)}(undef, n)
    right[n] = r
    for j in n:-1:2
        right[j-1] = transfers[j] * right[j]
    end
    left = l'
    gradient = map(1:n) do j
        local_grad = map(derivatives(j)) do dA
            real(left * gtrm(dA, ket[j]) * right[j] / pairing / vals[i])
        end
        j < n && (left = left * transfers[j])
        return local_grad
    end
    return (; result..., gradient)
end

function _mixed_eigenvectors!(cell::Matrix{<:Real})
    re, im, L, R = LAPACK.geev!('V', 'V', cell)
    return complex.(re, im), L, R
end

_mixed_eigenvectors!(cell::Matrix{<:Complex}) = LAPACK.geev!('V', 'V', cell)

# `_krylov_opts` lives in src/Krylov.jl (loaded earlier in the include chain).

# Threshold below which the dense `gtrm + eigen` path is competitive with the
# matrix-free Krylov sweep. For χ ≤ 8 the Arnoldi startup overhead matches or
# exceeds the cost of a full χ²×χ² eigendecomposition; above that, the
# matrix-free path is asymptotically better by O(χ³).
const _CHAIN_EIG_DENSE_THRESHOLD = 8
const _DENSE_TRANSFER_LIMIT = 4096

# Retain the phase for callers that must restore the scale of a tensor network.
function _dominant_chain_eigenvalue(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}};
    dense_threshold::Integer=_CHAIN_EIG_DENSE_THRESHOLD,
    dense_size::Integer=0,
    tol::Union{Nothing,Real}=nothing,
    maxiter::Union{Nothing,Integer}=nothing,
    strict::Bool=false,
)
    n = length(T1s)
    n == length(T2s) || throw(ArgumentError(
        "T1s and T2s have different lengths: $n vs $(length(T2s))"
    ))
    n > 0 || throw(ArgumentError("T1s and T2s must be non-empty"))

    χL_T1 = size(T1s[1], 1)
    χR_T1 = size(T1s[end], 3)
    χL_T2 = size(T2s[1], 1)
    χR_T2 = size(T2s[end], 3)
    χL_T1 == χR_T1 && χL_T2 == χR_T2 || throw(DimensionMismatch(
        "chain transfer not square: T1 has left=$χL_T1 right=$χR_T1, " *
        "T2 has left=$χL_T2 right=$χR_T2"
    ))

    # The dense path builds interior transfer matrices as well as the seam.
    dense_allowed() = all(size(A, 1) * size(B, 1) <= _DENSE_TRANSFER_LIMIT
                          for (A, B) in zip(T1s, T2s))
    if (max(χL_T1, χL_T2) <= dense_threshold || χL_T1 * χL_T2 <= dense_size) && dense_allowed()
        return _dominant_eigenvalue_dense(gtrm(T1s, T2s); overwrite=true)
    end

    T = promote_type(eltype(T1s[1]), eltype(T2s[1]))
    # Rotate both cells together to use the smallest virtual product space.
    # Keep the existing seam on ties and preserve the workspace precision.
    seam = n
    for j in 1:n-1
        d1, d2 = size(T1s[j], 3), size(T2s[j], 3)
        if d1 * d2 < χR_T1 * χR_T2 &&
           promote_type(eltype(T1s[j+1]), eltype(T2s[j+1])) === T
            seam = j
            χR_T1, χR_T2 = d1, d2
        end
    end
    cell1 = seam == n ? T1s : _rotate_tensor_cell(T1s, seam)
    cell2 = seam == n ? T2s : _rotate_tensor_cell(T2s, seam)

    seed = zeros(T, χR_T2, χR_T1)
    @inbounds for i in 1:min(χR_T2, χR_T1)
        seed[i, i] = one(T)
    end
    if T1s !== T2s
        # For two different states the (rectangular) identity can lie in an invariant
        # subspace of the mixed transfer that excludes its dominant eigenvector, e.g. when
        # the states differ by a sign gauge on the virtual bond; Arnoldi then converges,
        # with a small residual, to the wrong eigenvalue. A fixed, reproducible
        # unstructured component removes that structure. The self-transfer keeps the
        # identity, which is its fixed point in canonical form.
        _add_unstructured_seed!(seed)
    end

    apply_cell = _chain_transfer_action(cell1, cell2; dir=:r)
    f = ρ -> vec(apply_cell(reshape(ρ, χR_T2, χR_T1)))

    v₀ = vec(seed)
    # An absolute residual test alone can accept an arbitrary Ritz value
    # if raw tensors have an overall small scalar. Scale the strict solve
    # relative to the transfer action, without touching the input tensors.
    scale = strict ? norm(f(v₀)) / norm(v₀) : one(real(T))
    isfinite(scale) && scale > 0 || (scale = one(scale))
    action = isone(scale) ? f : (v -> (y = f(v); y ./= scale))
    strict && isnothing(tol) && (tol = max(1e-14, 10 * eps(real(float(T)))))
    opts = _krylov_opts(; tol, maxiter)
    vals, _, info = eigsolve(action, v₀, 1, :LM; ishermitian=false, opts...)
    if info.converged < 1
        strict && error("dominant chain eigenvalue Krylov solver did not converge")
        if !dense_allowed()
            @warn "Dominant chain eigenvalue Krylov solver did not converge; returning NaN" info
            return NaN
        end
        @warn "Dominant chain eigenvalue Krylov solver did not converge; " *
              "falling back to dense path" info
        return _dominant_eigenvalue_dense(gtrm(cell1, cell2); overwrite=true)
    end
    return scale * vals[1]
end

# Deterministic quasi-random entries (no RNG state), same magnitude as the identity part.
function _add_unstructured_seed!(seed::AbstractMatrix{T}) where {T}
    m, n = size(seed)
    @inbounds for j in 1:n, i in 1:m
        x = sin(1.0 + 1.6180339887 * i + 2.7182818285 * j + 0.5772156649 * i * j)
        if T <: Complex
            y = cos(0.5 + 2.2360679775 * i + 1.4142135624 * j + 0.3183098862 * i * j)
            seed[i, j] += T(x, y) / 2
        else
            seed[i, j] += T(x) / 2
        end
    end
    return seed
end

function _dominant_eigenvalue_dense(trmat::AbstractMatrix; overwrite=false)
    values = overwrite && trmat isa StridedMatrix{<:LinearAlgebra.BlasFloat} ?
             eigvals!(trmat) : eigvals(trmat)
    return argmax(abs, values)
end
