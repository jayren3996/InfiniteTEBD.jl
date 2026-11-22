#---------------------------------------------------------------------------------------------------
# Constraint diagnostics: nearest-neighbour blockade
#---------------------------------------------------------------------------------------------------
export Blockade

"""
    Blockade(; occupied=2)

The nearest-neighbour blockade projector `P = ∏_j (1 − n_j n_{j+1})` on a chain,
with `n = |occupied⟩⟨occupied|` (basis index `occupied`, `2` = `|1⟩` for spin-½).
It forbids two adjacent sites from both being in state `occupied`; any local
dimension is allowed.

[`constraint_leakage`](@ref) computes the blockade probability from a positive
transfer on two environments labelled by the previous site's occupation.
This sum of non-negative terms retains the relative accuracy of small weights.

`iMPO(b, n; d=2)` gives the same projector as a general layer MPO, for
comparison or for code that needs an `iMPO`.
"""
struct Blockade
    occupied::Int
    function Blockade(occupied::Integer)
        occupied >= 1 || throw(ArgumentError("Blockade: occupied must be a positive basis index"))
        return new(Int(occupied))
    end
end
Blockade(; occupied::Integer=2) = Blockade(occupied)

function iMPO(b::Blockade, n::Integer; d::Integer=2)
    b.occupied <= d || throw(ArgumentError("Blockade: occupied = $(b.occupied) exceeds d = $d"))
    N = zeros(ComplexF64, d, d)
    N[b.occupied, b.occupied] = 1
    edge = Matrix{ComplexF64}(I, d^2, d^2) - kron(N, N)
    return layer_mpo([(edge, k, k + 1) for k in 1:n], n; d, structured=false)
end

function _check_constraint(ψ::iMPS, b::Blockade)
    ψ isa DenseIMPS || throw(ArgumentError("Blockade diagnostics support dense iMPS states"))
    for i in 1:ψ.n
        d = size(ψ.Γ[i], 2)
        b.occupied <= d || throw(DimensionMismatch(
            "site $i: Blockade occupied index $(b.occupied) exceeds the physical dimension $d"))
    end
    return nothing
end

# --- positive transfer -----------------------------------------------------------------------

# Spectral radius of the cell map `x ↦ sites[n](⋯ sites[1](x))`, a product of linear site
# maps that preserve positive semidefinite matrices, started from the positive definite
# `x0`. Returns its log (`-Inf` when the map annihilates `x0`, i.e. the radius is exactly 0).
#
# Each site map is divided by a fixed scale σᵢ taken from per-site-normalized power
# iterations, so the scaled cell map has radius ~1 for any cell length (no underflow when a
# small weight per site is repeated over a long cell) and the eigensolver tolerance is
# relative. log ρ = Σᵢ log σᵢ + log ρ(scaled). Small maps use a dense, balanced eigenvalue
# computation.
const _POSITIVE_DENSE_MAX = 64
const _POSITIVE_WARMUP = 4

_log_positive_radius(f, x0::AbstractVector) = _log_positive_radius([f], x0)

function _log_positive_radius(sites::AbstractVector, x0::AbstractVector)
    x = x0 ./ norm(x0)
    logσ = zeros(length(sites))
    for _ in 1:_POSITIVE_WARMUP, (i, f) in enumerate(sites)
        y = f(x)
        s = norm(y)
        iszero(s) && return -Inf
        isfinite(s) || throw(ArgumentError("constraint transfer: non-finite values"))
        x = y ./ s
        logσ[i] = log(s)
    end
    σ = exp.(logσ)
    cell(v) = foldl((u, (f, c)) -> f(u) ./ c, zip(sites, σ); init=v)
    m = length(x0)
    if m <= _POSITIVE_DENSE_MAX
        M = Matrix{eltype(x)}(undef, m, m)
        e = zeros(eltype(x), m)
        for k in 1:m
            e[k] = 1
            M[:, k] = cell(e)
            e[k] = 0
        end
        r = maximum(abs, eigvals!(M); init=0.0)
    else
        vals, _, info = eigsolve(cell, x, 1, :LM; ishermitian=false, tol=1e-13,
                                 krylovdim=min(m, 30), maxiter=200)
        # Plain power iteration on the Perron vector if Arnoldi does not converge.
        r = info.converged < 1 ? _positive_power(cell, x) : abs(vals[1])
    end
    iszero(r) && return -Inf
    return sum(logσ) + log(r)
end

# Power iteration with two-step ratios (robust to a period-2 peripheral spectrum).
function _positive_power(f, x; maxiter=10_000, tol=1e-13)
    prev = NaN
    for _ in 1:maxiter
        y = f(f(x))
        s = norm(y)
        iszero(s) && return 0.0
        r = sqrt(s)
        abs(r - prev) <= tol * r && return r
        prev = r
        x = y ./ s
    end
    @warn "constraint transfer: power iteration did not converge" prev
    return prev
end

# Site blocks B[:, s, :] as dense matrices.
_site_blocks(A) = [Matrix(view(A, :, s, :)) for s in 1:size(A, 2)]

# log ⟨ψ|P|ψ⟩ per cell for the blockade, relative to log ⟨ψ|ψ⟩ per cell (≤ 0; -Inf if P
# annihilates ψ). Left environments L₀/L₁ are labelled by whether the previous site is
# occupied: L₀′ = Σ_{s≠occ} B_s†(L₀+L₁)B_s, L₁′ = B_occ† L₀ B_occ. The vector holds
# (vec L₀, vec L₁).
function _blockade_log_weight(ψ::DenseIMPS, b::Blockade)
    blocks = [_site_blocks(A) for A in ψ.Γ]
    occ = b.occupied
    T = eltype(ψ)
    function site_map(Bs)
        l, r = size(Bs[1])
        S = zeros(T, l, l)
        tmp = zeros(T, r, l)
        return function (v)
            L0 = reshape(view(v, 1:l^2), l, l)
            L1 = reshape(view(v, l^2+1:2l^2), l, l)
            out = zeros(T, 2r^2)
            N0 = reshape(view(out, 1:r^2), r, r)
            N1 = reshape(view(out, r^2+1:2r^2), r, r)
            S .= L0 .+ L1
            for s in eachindex(Bs)
                if s == occ
                    mul!(tmp, Bs[s]', L0)
                    mul!(N1, tmp, Bs[s])
                else
                    mul!(tmp, Bs[s]', S)
                    mul!(N0, tmp, Bs[s], true, true)
                end
            end
            return out
        end
    end
    self_map(Bs) = v -> begin
        l, r = size(Bs[1])
        L = reshape(v, l, l)
        vec(sum(B' * L * B for B in Bs))
    end
    χ = size(ψ.Γ[1], 1)
    Id = vec(Matrix{T}(I, χ, χ))
    lognorm = _right_canonical(blocks) ? 0.0 : _log_positive_radius(self_map.(blocks), Id)
    isfinite(lognorm) || throw(ArgumentError("constraint_leakage: the state has zero norm"))
    logw = _log_positive_radius(site_map.(blocks), vcat(Id, Id))
    return min(logw - lognorm, 0.0)
end

# Σ_s B_s B_s† = 1 on every site: the cell transfer has the identity as right fixed point with
# eigenvalue 1, so the norm per cell is exactly 1 (states returned by `canonical!`,
# `apply!`, `truncate!` with renormalization).
const _RIGHT_CANONICAL_TOL = 1e-13
function _right_canonical(blocks)
    for Bs in blocks
        G = sum(B * B' for B in Bs)
        size(G, 1) == size(G, 2) || return false
        norm(G - I, Inf) <= _RIGHT_CANONICAL_TOL || return false
    end
    return true
end

"""
    constraint_leakage(ψ, b::Blockade)

Per-site leakage `−log⟨ψ|P|ψ⟩ / (2n)` out of the blockade subspace, computed from
the positive transfer of [`Blockade`](@ref) (same value as
`constraint_leakage(ψ, iMPO(b, ψ.n))`, without building `P|ψ⟩`). Invariant under
rescaling `ψ` and under repeating its unit cell. Returns `Inf` only when the
projection is exactly zero for the stored tensors.
"""
function constraint_leakage(ψ::iMPS, b::Blockade)
    _check_constraint(ψ, b)
    logw = _blockade_log_weight(ψ, b)
    return isfinite(logw) ? max(-logw / (2ψ.n), 0.0) : Inf
end
