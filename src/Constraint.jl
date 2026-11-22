#---------------------------------------------------------------------------------------------------
# Constraint diagnostics: nearest-neighbour blockade and projected observables
#---------------------------------------------------------------------------------------------------
export Blockade, projected_energy_density

"""
    Blockade(; occupied=2)

The nearest-neighbour blockade projector `P = ∏_j (1 − n_j n_{j+1})` on a chain,
with `n = |occupied⟩⟨occupied|` (basis index `occupied`, `2` = `|1⟩` for spin-½).
It forbids two adjacent sites from both being in state `occupied`; any local
dimension is allowed.

Pass it wherever a constraint projector is accepted (`constraint_leakage`,
`projected_energy_density`, `scarfinder_step!(…; constraint)`). Because the
projector is identified explicitly, these use an exact classical
representation instead of a general projector MPO:

- leakage: the probability `⟨ψ|P|ψ⟩` per cell is the spectral radius of a
  completely positive transfer map on two environments labelled by the
  previous site's occupation, a sum of non-negative terms (no cancellation), so
  small weights keep their relative accuracy, and exact annihilation is an
  exact zero;
- projection: `P|ψ⟩` is formed by copying blocks of `ψ`'s tensors into a
  bond of twice the dimension (the 0/1 automaton of the blockade), without
  arithmetic.

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

const _Constraint = Union{iMPO,Blockade}

_check_constraint(ψ::iMPS, P::iMPO) = _check_mpo_state(ψ, P)
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

# Spectral radius of a linear map `f` on vectors of length `m` that preserves the cone of
# positive semidefinite matrices, started from the positive definite `x0`. Returns its log
# (`-Inf` when the map annihilates `x0`, i.e. when the radius is exactly zero).
#
# The map is rescaled by a power-iteration estimate before Arnoldi, so the eigensolver
# tolerance is relative to the radius, however small it is. Small maps use a dense,
# balanced eigenvalue computation.
const _POSITIVE_DENSE_MAX = 64
const _POSITIVE_WARMUP = 4

function _log_positive_radius(f, x0::AbstractVector)
    x = x0 ./ norm(x0)
    logscale = 0.0
    for _ in 1:_POSITIVE_WARMUP
        y = f(x)
        s = norm(y)
        iszero(s) && return -Inf
        isfinite(s) || throw(ArgumentError("constraint transfer: non-finite values"))
        x = y ./ s
        logscale = log(s)
    end
    ρ0 = exp(logscale)
    m = length(x0)
    if m <= _POSITIVE_DENSE_MAX
        M = Matrix{eltype(x)}(undef, m, m)
        e = zeros(eltype(x), m)
        for k in 1:m
            e[k] = 1
            M[:, k] = f(e) ./ ρ0
            e[k] = 0
        end
        r = maximum(abs, eigvals!(M); init=0.0)
    else
        vals, _, info = eigsolve(v -> f(v) ./ ρ0, x, 1, :LM; ishermitian=false, tol=1e-13,
                                 krylovdim=min(m, 30), maxiter=200)
        if info.converged < 1
            # Fall back to plain power iteration on the Perron vector (positive map).
            r = _positive_power(v -> f(v) ./ ρ0, x)
        else
            r = abs(vals[1])
        end
    end
    iszero(r) && return -Inf
    return logscale + log(r)
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
# occupied: L₀′ = Σ_{s≠occ} B_s†(L₀+L₁)B_s, L₁′ = B_occ† L₀ B_occ.
function _blockade_log_weight(ψ::DenseIMPS, b::Blockade)
    blocks = [_site_blocks(A) for A in ψ.Γ]
    occ = b.occupied
    T = eltype(ψ)
    χ = size(ψ.Γ[1], 1)
    k = χ^2
    # buffers: environments entering each site, and B†L products
    envs = [(zeros(T, size(A, 1), size(A, 1)), zeros(T, size(A, 1), size(A, 1))) for A in ψ.Γ]
    S = [zeros(T, size(A, 1), size(A, 1)) for A in ψ.Γ]
    tmp = [zeros(T, size(A, 3), size(A, 1)) for A in ψ.Γ]
    function cell(v)
        L0, L1 = envs[1]
        copyto!(L0, 1, v, 1, k)
        copyto!(L1, 1, v, k + 1, k)
        out = similar(v)
        for (i, Bs) in enumerate(blocks)
            L0, L1 = envs[i]
            last = i == length(blocks)
            N0, N1 = last ? (zeros(T, χ, χ), zeros(T, χ, χ)) : envs[i+1]
            S[i] .= L0 .+ L1
            fill!(N0, 0)
            for s in eachindex(Bs)
                Bs_ = Bs[s]
                if s == occ
                    mul!(tmp[i], Bs_', L0)
                    mul!(N1, tmp[i], Bs_)
                else
                    mul!(tmp[i], Bs_', S[i])
                    mul!(N0, tmp[i], Bs_, true, true)
                end
            end
            if last
                copyto!(out, 1, N0, 1, k)
                copyto!(out, k + 1, N1, 1, k)
            end
        end
        return out
    end
    Id = vec(Matrix{T}(I, χ, χ))
    lognorm = _right_canonical(blocks) ? 0.0 : _log_positive_radius(v -> _self_cell(blocks, v, χ), Id)
    isfinite(lognorm) || throw(ArgumentError("constraint_leakage: the state has zero norm"))
    logw = _log_positive_radius(cell, vcat(Id, Id))
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

function _self_cell(blocks, v, χ)
    L = reshape(v, χ, χ)
    for Bs in blocks
        L = sum(B' * L * B for B in Bs)
    end
    return vec(L)
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

# --- projection ------------------------------------------------------------------------------

# P|ψ⟩ as an exact iMPS: bond index (label q, χ index) with q = occupation of the site to
# the left of the bond, χ index fastest. Entries are copied, not computed.
function _blockade_project(ψ::DenseIMPS, b::Blockade)
    occ = b.occupied
    T = eltype(ψ)
    Γ = Vector{Array{T,3}}(undef, ψ.n)
    λ = similar(ψ.λ)
    for i in 1:ψ.n
        B = ψ.Γ[i]
        l, d, r = size(B)
        C = zeros(T, 2l, d, 2r)
        for s in 1:d
            q = s == occ ? 1 : 0
            for p in 0:1
                (p == 1 && s == occ) && continue
                C[p*l+1:(p+1)*l, s, q*r+1:(q+1)*r] = view(B, :, s, :)
            end
        end
        Γ[i] = C
        λ[i] = vcat(ψ.λ[i], ψ.λ[i])            # placeholder sizes only
    end
    return iMPS(Γ, λ, ψ.n)
end

# Normalized canonical P|ψ⟩ with no truncation. Throws MPOAnnihilationError if Pψ = 0.
function _normalized_projection(ψ::DenseIMPS, b::Blockade)
    isfinite(_blockade_log_weight(ψ, b)) || throw(MPOAnnihilationError(
        "the blockade projection of the state is exactly zero"))
    φ = _blockade_project(ψ, b)
    _precise_canonical!(φ)
    return φ
end

function _normalized_projection(ψ::DenseIMPS, P::iMPO)
    φ = _contract_mpo(ψ, P)
    lognorm = _precise_canonical!(φ)
    _annihilated(lognorm, ψ, P) && throw(MPOAnnihilationError(
        "the projector maps the state to zero (norm per cell at the roundoff level)"))
    return φ
end

"""
    projected_energy_density(ψ, P, terms; n=ψ.n)

Energy density of the normalized projection `P|ψ⟩ / ‖P|ψ⟩‖`, the physical
component of `ψ` for a constraint projector `P` (an [`iMPO`](@ref) projector or a
[`Blockade`](@ref)), measured with `energy_density(·, terms; n)`.

The projection is exact: `P|ψ⟩` keeps its full bond dimension (at most
`bonddim(P) · χ`, minus exact null directions) and is not truncated back to `χ`,
so the value is not contaminated by a truncation. `ψ` is not modified.

Throws `MPOAnnihilationError` when `P|ψ⟩ = 0`: the projected energy is then
undefined. For a `Blockade` this is decided exactly (an exact zero of the
positive transfer); for a general projector `iMPO` by the roundoff test of
[`apply!`](@ref).
"""
function projected_energy_density(ψ::iMPS, P::_Constraint, terms::AbstractVector; n::Integer=ψ.n)
    _check_constraint(ψ, P)
    return energy_density(_normalized_projection(ψ, P), terms; n)
end

_check_constraint(::iMPS, c) = throw(ArgumentError(
    "constraint must be a projector iMPO or a Blockade, got $(typeof(c))"))
