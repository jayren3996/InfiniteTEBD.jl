#---------------------------------------------------------------------------------------------------
# QUANTUM GATE
#---------------------------------------------------------------------------------------------------
"""
    tensor_applygate!(G, Γs, λl; keywords...)

Apply a dense local gate `G` to a contiguous block of stored local tensors.

Conceptually this routine:

1. groups the local tensors `Γs` into a single block tensor,
2. inserts the incoming Schmidt values `λl` on the left bond,
3. applies the dense gate `G` on the physical legs,
4. decomposes the updated block back into site-local tensors.

Parameters:
- `G`
  Dense local operator acting on the physical Hilbert space of the grouped
  block. Its matrix dimension must match the total physical dimension of `Γs`.
  The operator must already be in the internal fused-leg layout (leftmost
  site's index fastest); [`applygate!`](@ref) converts from the documented
  `kron` convention before calling this routine, and direct callers can use
  [`convert_operator`](@ref) to do the same.
- `Γs`
  Stored local tensors for the contiguous block to be updated.
- `λl`
  Schmidt values on the bond immediately to the left of that block.

Keyword arguments:
- `maxdim=MAXDIM`
  Hard cap on the retained bond dimension during the decomposition of the
  updated block.
- `mindim=1`
  Minimum retained bond dimension on each updated internal bond.
- `truncerr=0.0`
  Target discarded weight used to choose the smallest admissible kept bond
  dimension on each updated internal bond. Set `truncerr = 0` for fixed-cap
  evolution.
- `svd_min=SVDTOL`
  Optional absolute singular-value floor applied after the discarded-weight
  target is evaluated.
- `renormalize=false`
  Whether to renormalize Schmidt values produced by the decomposition.

Returns:
- `(Γs_new, λs_new)` where `Γs_new` are the updated stored local tensors and
  `λs_new` are the Schmidt spectra on the internal bonds of the updated block.

Notes:
- This is a low-level helper. Most user-facing evolution code should call
  [`applygate!`](@ref) or [`evolve!`](@ref) instead.
- The tensors are assumed to already follow the package storage convention
  `B_i = Γ_i λ_i`.
"""
function tensor_applygate!(
    G::AbstractMatrix{<:Number}, Γs::AbstractVector{<:AbstractArray{<:Number, 3}},
    λl::AbstractVector{<:Number};
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Union{Nothing,Real}=nothing,
    cutoff::Union{Nothing,Real}=nothing,
    renormalize::Bool=false,
    return_stats::Bool=false,
    bond_indices::Union{Nothing,AbstractVector{<:Integer}}=nothing
)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_truncation_args(maxdim, mindim, truncerr, svd_floor)
    n = length(Γs)
    isone(n) && return return_stats ? ([tensor_umul(G, Γs[1])], Vector{Vector{eltype(λl)}}(), BondStat[]) : ([tensor_umul(G, Γs[1])], Vector{Vector{eltype(λl)}}())
    if n == 2 && _needs_inverse_free_reconstruction(λl)
        bond = isnothing(bond_indices) ? 1 : first(bond_indices)
        return _apply_two_site_gate(G, Γs[1], Γs[2], λl, bond;
            maxdim, mindim, truncerr, svd_min=svd_floor, renormalize, return_stats)
    end
    Γ = tensor_group(Γs)
    tensor_lmul!(λl, Γ)
    GΓ = tensor_umul(G, Γ)
    _tensor_decomp!(
        GΓ,
        λl,
        [size(A, 2) for A in Γs];
        maxdim,
        mindim,
        truncerr,
        svd_min=svd_floor,
        renormalize,
        return_stats,
        bond_indices,
    )
end

function _needs_inverse_free_reconstruction(λ)
    scale = float(maximum(abs, λ; init=0))
    threshold = scale * sqrt(eps(typeof(scale)))
    return any(s -> 0 < abs(s) < threshold, λ)
end

function _apply_two_site_gate(G, A, B, λl, bond; maxdim, mindim, truncerr,
                              svd_min, renormalize, return_stats)
    left, dleft, _ = size(A)
    dright = size(B, 2)
    C = tensor_umul(G, tensor_group_2(A, B))
    weighted = copy(C)
    tensor_lmul!(λl, weighted)
    factor, choice = _svd_rank_choice(reshape(weighted, left*dleft, :);
        maxdim, mindim, truncerr, svd_min)
    k = choice.chi_keep
    S = factor.S[1:k]
    Vt = factor.Vt[1:k, :]
    tensors, spectra = _reconstruct_two_site(C, λl, S, Vt, dleft, dright; renormalize)
    if return_stats
        stats = BondStat(bond, choice.chi_req, choice.chi_keep,
            choice.discarded_weight, choice.saturated, choice.target_met,
            choice.smallest_kept_sv, choice.largest_discarded_sv)
        return tensors, spectra, [stats]
    end
    return tensors, spectra
end

function _reconstruct_two_site(C, λl, S, Vt, dleft, dright; renormalize)
    left, _, right = size(C)
    # From Λ_left C V = U S, reconstruct the stored left tensor using
    # C V rather than multiplying U S by a numerically large inverse.
    left_tensor = reshape(reshape(C, left*dleft, :) * Vt', left, dleft, :)
    for a in eachindex(λl)
        iszero(λl[a]) && fill!(view(left_tensor, a, :, :), zero(eltype(left_tensor)))
    end
    if renormalize
        scale = norm(S)
        isfinite(scale) && scale > 0 || throw(ArgumentError(
            "cannot renormalize singular values with zero or non-finite norm"))
        S ./= scale
        left_tensor ./= scale
    end
    return [left_tensor, reshape(Vt, :, dright, right)], [convert(Vector{eltype(λl)}, S)]
end
#---------------------------------------------------------------------------------------------------
export applygate!
"""
    applygate!(ψ, G, i, j; maxdim=MAXDIM, mindim=1, truncerr=0.0, svd_min=SVDTOL, renormalize=true, recanonicalize=false, gauge=nothing, return_stats=false)

Apply a local gate `G` in place to the contiguous region from site `i` to site
`j` of the periodic unit cell.

Parameters:
- `ψ`
  State to update in place.
- `G`
  Dense local operator. Its dimension must match the total physical dimension of
  the block from site `i` to site `j`.
- `i`, `j`
  Start and end sites of the support inside the unit cell. If `j < i`, the
  support is interpreted with periodic wraparound.

Keyword arguments:
- `maxdim=MAXDIM`
  Hard cap on the retained bond dimension during the decomposition of the
  updated block.
- `mindim=1`
  Minimum retained bond dimension on each updated internal bond.
- `truncerr=0.0`
  Target discarded weight used to choose the smallest admissible kept bond
  dimension on each updated internal bond. Set `truncerr = 0` for fixed-cap
  evolution.
- `svd_min=SVDTOL`
  Optional absolute singular-value floor applied after the discarded-weight
  target is evaluated.
- `renormalize=true`
  Whether to renormalize the retained Schmidt values.
- `recanonicalize=false`
  After a wrap-around gate (`j < i`), whether to re-canonicalise the unit
  cell. The SVD chain inside `tensor_applygate!` keeps the affected block
  right-canonical locally, and for unitary gates with no truncation it
  preserves the global canonical form exactly — so the default `false`
  skips the O(χ⁶) `canonical!` call, which is the right choice for real-time
  unitary evolution. For non-unitary gates (e.g. imaginary-time
  `exp(-dt·h)`), or whenever truncation discards weight, the wraparound seam
  drifts from canonical form; pass `recanonicalize=true` so observables and
  downstream truncations stay accurate. Note: over long unitary evolutions
  the Vidal-gauge division by small Schmidt values accumulates drift too, so
  re-canonicalise periodically if you need high accuracy.
- `gauge=nothing`
  Request a fixed-bond-dimension gauge checkpoint after this gate with
  `:gram`, `:qr`, or `:auto`. Requires `renormalize=true` and
  `recanonicalize=false`. A failed gauge checkpoint leaves the input intact.
- `return_stats=false`
  If `true`, also return per-bond truncation diagnostics for this gate update.

Returns:
- `ψ` when `return_stats = false`.
- `(ψ, stats)` when `return_stats = true`, where `stats.bond_stats` contains one
  entry per updated internal bond.

Notes:
- For a one-site update with `i == j`, the operator is applied directly to the
  stored local tensor without a block decomposition.
- For multi-site updates, this routine works directly with the stored
  right-canonical tensors `ψ.Γ`.
- Site indices are interpreted periodically through the finite unit cell.
- Multi-site gates follow the documented `kron` convention: the leftmost site
  of the support corresponds to the leftmost `kron` factor, so
  `kron(A, B)` applied to `(i, i+1)` acts `A` on site `i` and `B` on site
  `i+1`. The conversion to the internal fused-leg layout happens inside this
  routine.
- The gate's element type must be representable in the state's element type:
  applying a complex gate to a real-eltype state throws an `ArgumentError`
  (construct the state with `ComplexF64`, or convert it with
  [`mps_promote_type`](@ref)).

Example:
```julia
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
X = [0 1; 1 0]
applygate!(psi, kron(X, X), 1, 2; maxdim=4)
```
"""
function applygate!(
    ψ::iMPS, G::AbstractMatrix,
    i::Integer, j::Integer;
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Union{Nothing,Real}=nothing,
    cutoff::Union{Nothing,Real}=nothing,
    renormalize::Bool=true,
    recanonicalize::Bool=false,
    gauge::Union{Nothing,Symbol}=nothing,
    return_stats::Bool=false
)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_truncation_args(maxdim, mindim, truncerr, svd_floor)
    _check_gate_eltype(ψ, G)
    i0, j0 = _normalize_gate_sites(ψ, i, j)
    _validate_gauge_checkpoint(ψ, gauge, renormalize, recanonicalize)
    if !isnothing(gauge)
        trial = iMPS(copy(ψ.Γ), copy(ψ.λ), ψ.n)
        i0 == j0 && (trial.Γ[i0] = copy(trial.Γ[i0]))
        result = applygate!(trial, G, i0, j0;
            maxdim, mindim, truncerr, svd_min=svd_floor, renormalize, return_stats)
        restore_canonical!(trial; method=gauge)
        ψ.Γ .= trial.Γ
        ψ.λ .= trial.λ
        return return_stats ? (ψ, result[2]) : ψ
    end
    if isequal(i0, j0)
        tensor_umul!(G, ψ.Γ[i0])
        if renormalize
            # Single-site gates can be non-unitary (e.g. imaginary-time terms
            # like exp(-dτ * h_local)), so renormalize must undo the norm
            # change. A unitary gate cannot change the norm, and checking
            # unitarity is O(d³) on the tiny physical dimension — so the hot
            # real-time path skips the normalization entirely. The previous
            # code ran a whole-unit-cell Krylov eigensolve (inner_product) on
            # every renormalized single-site gate, costing ~20× the gate
            # update itself for a mathematical no-op. Genuinely non-unitary
            # gates keep the exact global norm computation: any local
            # estimate goes stale as repeated non-unitary updates drive the
            # state away from canonical form.
            if norm(G' * G - I) > 1e-10 * size(G, 1)
                scale = inner_product(ψ)
                isfinite(scale) && scale > 0 && (ψ.Γ[i0] ./= sqrt(scale))
            end
        end
        if return_stats
            return ψ, (
                support=(i0, j0),
                bond_stats=BondStat[],
                max_discarded_weight=0.0,
                num_saturated=0,
            )
        end
        return ψ
    end
    inds = _gate_indices(ψ, i0, j0)
    Γs = ψ.Γ[inds]
    λl = ψ.λ[mod(i0-2,ψ.n)+1]
    # User-facing gates follow the documented kron convention (leftmost kron
    # factor acts on the leftmost site of the support); the internal fused
    # physical leg puts the leftmost site's index fastest, which is the mirror
    # ordering. Reverse the site order once at this boundary.
    G_int = _operator_to_internal(G, Int[size(Γk, 2) for Γk in Γs])
    if return_stats
        Γs, λs, bond_stats = tensor_applygate!(
            G_int,
            Γs,
            λl;
            maxdim,
            mindim,
            truncerr,
            svd_min=svd_floor,
            renormalize,
            return_stats=true,
            bond_indices=inds[1:end-1],
        )
    else
        Γs, λs = tensor_applygate!(
            G_int,
            Γs,
            λl;
            maxdim,
            mindim,
            truncerr,
            svd_min=svd_floor,
            renormalize,
        )
    end
    push!(λs, ψ.λ[j0])
    for k in eachindex(inds)
        ψ.Γ[inds[k]] = Γs[k]
        ψ.λ[inds[k]] = λs[k]
    end
    # Wrap-around gates (j0 < i0) update the block in place but leave the
    # wraparound seam non-canonical *unless* the gate is unitary AND no
    # truncation occurred. The default `recanonicalize=false` keeps the fast
    # path for unitary evolution; callers doing imaginary-time evolution (or
    # who otherwise need a canonical state every step) opt in with
    # `recanonicalize=true`.
    if recanonicalize && j0 < i0
        canonical!(ψ; maxdim, cutoff=svd_floor, renormalize)
    end
    return_stats && return ψ, _gate_update_stats(i0, j0, bond_stats)
    return ψ
end

function _validate_gauge_checkpoint(ψ, gauge, renormalize, recanonicalize)
    isnothing(gauge) && return nothing
    allowed = ψ isa DenseIMPS ? (:gram, :qr, :auto) : (:gram,)
    gauge in allowed || throw(ArgumentError("gauge must be one of $allowed for this state"))
    renormalize && !recanonicalize || throw(ArgumentError(
        "a gauge checkpoint requires renormalize=true and recanonicalize=false"))
    return nothing
end

"""
    _gate_indices(ψ, i, j)

Return the periodic list of unit-cell indices covered by the contiguous support
from site `i` to site `j`.

Notes:
- If `j < i`, the support is interpreted with periodic wraparound.
- This is an internal helper used by [`evolve!`](@ref).
"""
function _gate_indices(ψ::iMPS, i::Integer, j::Integer)
    i0, j0 = _normalize_gate_sites(ψ, i, j)
    i0 == j0 ? (i0:i0) : (j0 > i0 ? (i0:j0) : [i0:ψ.n; 1:j0])
end

function _normalize_gate_site(ψ::iMPS, i::Integer)
    ψ.n > 0 || throw(ArgumentError("unit cell must contain at least one site"))
    return mod(i - 1, ψ.n) + 1
end

function _normalize_gate_sites(ψ::iMPS, i::Integer, j::Integer)
    return _normalize_gate_site(ψ, i), _normalize_gate_site(ψ, j)
end

# The iMPS tensor storage is concretely typed, so an in-place update cannot
# promote a real state to complex; without this guard a complex gate on a real
# state fails deep inside the update with an InexactError (single-site path)
# or at the tensor write-back (multi-site path) — possibly after mutating part
# of the state. Real gates on complex states promote fine and are allowed.
function _check_gate_eltype(ψ::iMPS, G::AbstractMatrix)
    if eltype(ψ) <: Real && eltype(G) <: Complex
        throw(ArgumentError(
            "cannot apply a gate with complex element type $(eltype(G)) in place to an " *
            "iMPS with real element type $(eltype(ψ)). Construct the state with a " *
            "complex element type (e.g. rand_iMPS(ComplexF64, n, d, dim) or " *
            "product_iMPS(ComplexF64, vectors)), or convert an existing state with " *
            "mps_promote_type(ComplexF64, ψ)."))
    end
    return nothing
end

function _updated_bond_indices(ψ::iMPS, i::Integer, j::Integer)
    inds = collect(_gate_indices(ψ, i, j))
    length(inds) <= 1 && return Int[]
    return inds[1:end-1]
end

const _SUZUKI_FOURTH_P = 1 / (4 - 4^(1 / 3))
const _BARTHEL_A1 = 0.09584850274120368
const _BARTHEL_A2 = -0.07811115892163792
const _BARTHEL_A3 = 0.5 - (_BARTHEL_A1 + _BARTHEL_A2)
const _BARTHEL_B1 = 0.42652466131587616
const _BARTHEL_B2 = -0.12039526945509727
const _BARTHEL_B3 = 1 - 2 * (_BARTHEL_B1 + _BARTHEL_B2)

function _validate_truncation_args(
    maxdim::Integer,
    mindim::Integer,
    truncerr::Real,
    svd_min::Real
)
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    mindim > 0 || throw(ArgumentError("mindim must be positive"))
    maxdim >= mindim || throw(ArgumentError("maxdim must be at least mindim"))
    isfinite(truncerr) && truncerr >= 0 || throw(ArgumentError("truncerr must be finite and non-negative"))
    isfinite(svd_min) && svd_min >= 0 || throw(ArgumentError("svd_min must be finite and non-negative"))
    return nothing
end

function _resolve_svd_min(svd_min::Union{Nothing,Real}, cutoff::Union{Nothing,Real})
    if !isnothing(cutoff)
        isfinite(cutoff) && cutoff >= 0 ||
            throw(ArgumentError("cutoff must be finite and non-negative"))
        isnothing(svd_min) ||
            throw(ArgumentError("cutoff and svd_min are aliases; pass only one"))
        return cutoff
    end
    return isnothing(svd_min) ? SVDTOL : svd_min
end

function _validate_chi_policy(chi_policy::Symbol)
    chi_policy in (:fixed, :adaptive) ||
        throw(ArgumentError("chi_policy must be one of :fixed or :adaptive"))
    return nothing
end

function _validate_evolve_args(
    steps::Integer,
    maxdim::Integer,
    mindim::Integer,
    truncerr::Real,
    svd_min::Real
)
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    _validate_truncation_args(maxdim, mindim, truncerr, svd_min)
    return nothing
end

function _gate_update_stats(i::Integer, j::Integer, bond_stats)
    max_discarded_weight = isempty(bond_stats) ? 0.0 : maximum(stat.discarded_weight for stat in bond_stats)
    num_saturated = count(stat -> stat.saturated, bond_stats)
    (
        support=(i, j),
        bond_stats=bond_stats,
        max_discarded_weight=max_discarded_weight,
        num_saturated=num_saturated,
    )
end

function _aggregate_evolution_stats(gate_updates)
    bond_stats = BondStat[]
    for update in gate_updates
        append!(bond_stats, update.bond_stats)
    end

    max_discarded_weight = isempty(bond_stats) ? 0.0 : maximum(stat.discarded_weight for stat in bond_stats)
    mean_discarded_weight = isempty(bond_stats) ? 0.0 : sum(stat.discarded_weight for stat in bond_stats) / length(bond_stats)
    max_kept_dim = isempty(bond_stats) ? 0 : maximum(stat.chi_keep for stat in bond_stats)
    num_saturated = count(stat -> stat.saturated, bond_stats)

    (
        gate_updates=gate_updates,
        max_discarded_weight=max_discarded_weight,
        mean_discarded_weight=mean_discarded_weight,
        max_kept_dim=max_kept_dim,
        num_saturated=num_saturated,
    )
end

function _validate_trotter_scheme(num_layers::Integer, trotter::Symbol)
    num_layers > 0 || throw(ArgumentError("layers must contain at least one commuting layer"))
    trotter in (:second, :fourth, :fourth_opt) ||
        throw(ArgumentError("unknown trotter scheme $(repr(trotter)); use :second, :fourth, or :fourth_opt"))
    if trotter === :fourth && num_layers < 2
        throw(ArgumentError(
            "trotter = :fourth requires at least two commuting layers " *
            "in this interface. Use :second for a single commuting layer, " *
            "which already gives its exact propagator before truncation."))
    end
    if trotter === :fourth_opt && num_layers != 2
        throw(ArgumentError("trotter = :fourth_opt requires exactly two commuting layers"))
    end
    return nothing
end

function _validate_trotter_evolution(trotter::Symbol, evolution::Symbol)
    evolution in (:real, :imaginary) ||
        throw(ArgumentError("unknown evolution $(repr(evolution)); use :real or :imaginary"))
    if evolution === :imaginary && trotter !== :second
        throw(ArgumentError("higher-order Trotterization is currently supported only for evolution = :real"))
    end
    return nothing
end

function _push_trotter_stage!(
    stages::Vector{Tuple{Int, Float64}},
    layer::Integer,
    coeff::Real
)
    abs(coeff) <= ZEROTOL && return stages

    coeff_f = Float64(coeff)
    if !isempty(stages) && last(stages)[1] == layer
        merged = last(stages)[2] + coeff_f
        if abs(merged) <= ZEROTOL
            pop!(stages)
        else
            stages[end] = (layer, merged)
        end
    else
        push!(stages, (layer, coeff_f))
    end
    return stages
end

function _append_strang_step!(
    stages::Vector{Tuple{Int, Float64}},
    num_layers::Integer,
    scale::Real
)
    if num_layers == 1
        # A single-layer schedule has no half-step boundaries, so the generic
        # push-with-merge would fuse *every* macro step into one ever-growing
        # stage: evolve!(ψ, layers, dt, steps) would build exp(-i·steps·dt·h)
        # and apply it once, skipping every intermediate truncation. That is
        # exact only when all terms in the layer commute; outside that
        # contract it silently changes the physics. Push each macro step as
        # its own stage instead so the evolution stays step-by-step.
        abs(scale) <= ZEROTOL && return stages
        push!(stages, (1, Float64(scale)))
        return stages
    end

    for layer in 1:num_layers-1
        _push_trotter_stage!(stages, layer, scale / 2)
    end
    _push_trotter_stage!(stages, num_layers, scale)
    for layer in (num_layers - 1):-1:1
        _push_trotter_stage!(stages, layer, scale / 2)
    end
    return stages
end

function _append_trotter_macro_step!(
    stages::Vector{Tuple{Int, Float64}},
    num_layers::Integer,
    trotter::Symbol
)
    if trotter === :second
        return _append_strang_step!(stages, num_layers, 1.0)
    elseif trotter === :fourth
        p = _SUZUKI_FOURTH_P
        q = 1 - 4p
        _append_strang_step!(stages, num_layers, p)
        _append_strang_step!(stages, num_layers, p)
        _append_strang_step!(stages, num_layers, q)
        _append_strang_step!(stages, num_layers, p)
        _append_strang_step!(stages, num_layers, p)
        return stages
    elseif trotter === :fourth_opt
        _push_trotter_stage!(stages, 1, _BARTHEL_A1)
        _push_trotter_stage!(stages, 2, _BARTHEL_B1)
        _push_trotter_stage!(stages, 1, _BARTHEL_A2)
        _push_trotter_stage!(stages, 2, _BARTHEL_B2)
        _push_trotter_stage!(stages, 1, _BARTHEL_A3)
        _push_trotter_stage!(stages, 2, _BARTHEL_B3)
        _push_trotter_stage!(stages, 1, _BARTHEL_A3)
        _push_trotter_stage!(stages, 2, _BARTHEL_B2)
        _push_trotter_stage!(stages, 1, _BARTHEL_A2)
        _push_trotter_stage!(stages, 2, _BARTHEL_B1)
        _push_trotter_stage!(stages, 1, _BARTHEL_A1)
        return stages
    end

    throw(ArgumentError("unknown trotter scheme $(repr(trotter)); use :second, :fourth, or :fourth_opt"))
end

function _trotter_stage_schedule(num_layers::Integer, trotter::Symbol, steps::Integer)
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    _validate_trotter_scheme(num_layers, trotter)

    stages = Tuple{Int, Float64}[]
    for _ in 1:steps
        _append_trotter_macro_step!(stages, num_layers, trotter)
    end
    return stages
end

export trotter_stages

"""
    trotter_stages(num_layers; trotter=:second, steps=1)

Return the ordered `(layer, coefficient)` pairs used by
[`trotter_gates`](@ref). Apply layer `k` for time `coefficient * dt`.
Use the coefficients to prepare custom gates, including structured gates
or gates on separated endpoints. This schedule uses the same boundary
merges as the materialized gate-list interface.
"""
function trotter_stages(num_layers::Integer; trotter::Symbol=:second, steps::Integer=1)
    return _trotter_stage_schedule(num_layers, trotter, steps)
end

function _trotter_time_prefactor(dt::Real, coeff::Real, evolution::Symbol)
    if evolution === :real
        return -1im * coeff * dt
    elseif evolution === :imaginary
        return -coeff * dt
    end
    throw(ArgumentError("unknown evolution $(repr(evolution)); use :real or :imaginary"))
end

function _materialize_trotter_gates(
    layers,
    dt::Real,
    stages::AbstractVector{<:Tuple{Int, <:Real}};
    evolution::Symbol=:real,
    stage_ends::Union{Nothing,Vector{Int}}=nothing,
)
    prefactor = _trotter_time_prefactor(dt, 1.0, evolution)
    T = Union{}
    has_diagonal = false
    has_dense = false
    for layer in layers, term in layer
        h = term[1]
        T = promote_type(T, eltype(h))
        has_diagonal |= h isa Diagonal
        has_dense |= !(h isa Diagonal)
    end
    T = T === Union{} ? ComplexF64 : promote_type(T, typeof(prefactor))
    diagonal_type = Diagonal{T,Vector{T}}
    gate_type = has_diagonal ? (has_dense ? Union{diagonal_type,Matrix{T}} : diagonal_type) : Matrix{T}
    cache = IdDict{Any, Dict{Float64, gate_type}}()
    gates = Tuple{gate_type, Int, Int}[]
    if all(layer -> layer isa AbstractVector || layer isa Tuple, layers)
        sizehint!(gates, sum(length(layers[layer]) for (layer, _) in stages; init=0))
    end

    for (layer_idx, coeff) in stages
        scale = _trotter_time_prefactor(dt, coeff, evolution)
        for (h, i, j) in layers[layer_idx]
            by_coefficient = get!(() -> Dict{Float64, gate_type}(), cache, h)
            G = get!(by_coefficient, Float64(coeff)) do
                h isa Diagonal ? Diagonal(Vector{T}(exp.(scale .* h.diag))) :
                    convert(Matrix{T}, exp(scale * h))
            end
            push!(gates, (G, i, j))
        end
        isnothing(stage_ends) || push!(stage_ends, length(gates))
    end

    return gates
end

# The published multi-layer schedules begin and end in the same layer.
# Reuse the gates for that merged boundary and for the interior stages.
function _iterated_trotter_gates(layers, dt, steps; trotter, evolution)
    n = length(layers)
    _validate_trotter_evolution(trotter, evolution)
    stages = _trotter_stage_schedule(n, trotter, min(steps, 1))
    if steps <= 1 || n == 1
        gates = _materialize_trotter_gates(layers, dt, stages; evolution)
        return steps <= 1 || isempty(gates) ? gates : Iterators.flatten(Iterators.repeated(gates, steps))
    end

    first_layer, first_coeff = first(stages)
    last_layer, last_coeff = last(stages)
    first_layer == last_layer || error("Trotter schedule must have matching boundary layers")
    m = length(stages)
    # An empty extra stage for a canceled boundary has no gate applications.
    if abs(last_coeff + first_coeff) > ZEROTOL
        push!(stages, (first_layer, last_coeff + first_coeff))
    end
    stage_ends = Int[]
    gates = _materialize_trotter_gates(layers, dt, stages; evolution, stage_ends)
    isempty(gates) && return gates
    initial = view(gates, 1:stage_ends[m-1])
    interior = view(gates, stage_ends[1]+1:stage_ends[m-1])
    boundary = view(gates, stage_ends[m]+1:length(gates))
    final = view(gates, stage_ends[m-1]+1:stage_ends[m])
    repeating = Iterators.flatten(Iterators.repeated((boundary, interior), steps-1))
    return Iterators.flatten(Iterators.flatten(((initial,), repeating, (final,))))
end

function _evolve_gate_sequence!(
    ψ::iMPS,
    gates;
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Union{Nothing,Real}=nothing,
    cutoff::Union{Nothing,Real}=nothing,
    renormalize::Bool=true,
    recanonicalize::Bool=false,
    return_stats::Bool=false,
    chi_policy::Symbol=:fixed,
    q::Real=1.0,
    alpha::Real=0.1
)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_chi_policy(chi_policy)
    gate_updates = return_stats ? Any[] : nothing
    χ = chi_policy === :adaptive ? min(maxdim, max(mindim, maximum(_bond_dim, ψ.λ; init=mindim))) : maxdim
    for gate in gates
        G, i, j = gate
        if return_stats
            _, stats = applygate!(
                ψ,
                G,
                i,
                j;
                maxdim,
                mindim,
                truncerr,
                svd_min=svd_floor,
                renormalize,
                recanonicalize,
                return_stats=true,
            )
            push!(gate_updates, stats)
        else
            applygate!(ψ, G, i, j; maxdim, mindim, truncerr, svd_min=svd_floor, renormalize, recanonicalize)
        end

        if chi_policy === :adaptive
            updated_bonds = _updated_bond_indices(ψ, i, j)
            for bond in updated_bonds
                χ = adaptive_bonddim(
                    χ,
                    ψ.λ[bond];
                    mindim,
                    maxdim,
                    q,
                    alpha,
                    cutoff=svd_floor,
                )
            end
            if !isempty(updated_bonds)
                canonical!(ψ; maxdim=χ, cutoff=svd_floor, renormalize)
            end
        end
    end

    return gate_updates
end

export trotter_gates
"""
    trotter_gates(layers, dt; trotter=:second, evolution=:real, steps=1)

Materialize `steps` Trotter macro-steps from a Hamiltonian split into commuting
layers, as used by `evolve!(ψ, layers, dt, steps)`. For multiple layers the
schedule merges adjacent substeps of the same layer across step boundaries.
For one layer it keeps the individual steps and their intermediate truncations.

Each element of `layers` is a collection of local Hamiltonian terms
`(h, i, j)`. Terms inside a single layer are assumed to commute, so their dense
gates can be applied sequentially in the supplied order. The package does not
check commutation automatically.

Parameters:
- `layers`
  Hamiltonian split into commuting layers. Each local term reuses the tuple
  shape `(h, i, j)` with dense operator `h` supported on the unit-cell region
  `i:j`.
- `dt`
  Microscopic time step.

Keyword arguments:
- `trotter=:second`
  Trotter scheme. Use `:second` for Strang splitting, `:fourth` for Suzuki's
  recursive fourth-order composition, or `:fourth_opt` for the two-layer
  optimized real-time formula of Barthel and Zhang.
- `evolution=:real`
  Use `:real` to build gates `exp(-1im * c * dt * h)` or `:imaginary` to build
  second-order imaginary-time gates `exp(-c * dt * h)`.
- `steps=1`
  Number of macro-steps to generate. Zero returns an empty gate list.

Returns:
- A vector of local gates `(G, i, j)` for the full requested interval.
  Diagonal terms yield `Diagonal` gates.
  Apply the vector once with `evolve!(ψ, gates, 1)`.
  Entries can share a matrix; copy it before editing an individual gate.

Notes:
- `trotter = :fourth_opt` requires exactly two layers.
- Higher-order Trotter schemes currently reject `evolution = :imaginary`
  because their negative substeps are only supported for real-time evolution.

Example:
```julia
X = [0 1; 1 0]
H = kron(X, X)
layers = [[(H, 1, 2)], [(H, 2, 1)]]
gates = trotter_gates(layers, 0.1; trotter=:fourth_opt)
```
"""
function trotter_gates(
    layers,
    dt::Real;
    trotter::Symbol=:second,
    evolution::Symbol=:real,
    steps::Integer=1
)
    _validate_trotter_scheme(length(layers), trotter)
    _validate_trotter_evolution(trotter, evolution)

    stages = trotter_stages(length(layers); trotter, steps)
    return _materialize_trotter_gates(layers, dt, stages; evolution)
end

export evolve!
"""
    evolve!(ψ, gates, steps; maxdim=MAXDIM, mindim=1, truncerr=0.0, svd_min=SVDTOL, renormalize=true, gauge=nothing, return_stats=false)

Apply a sequence of local gates repeatedly for `steps` sweeps.

Each element of `gates` must be a tuple `(G, i, j)`. A matrix acts on the full
periodic interval `i:j`. A prepared [`EndpointGate`](@ref) acts at its
two endpoints. A [`ControlledGate`](@ref) acts on a three-site interval.

Parameters:
- `ψ`
  State to evolve in place.
- `gates`
  Iterable of `(G, i, j)` tuples, with matrices or prepared gates.
- `steps`
  Number of full sweeps through the gate list.

Keyword arguments:
- `maxdim=MAXDIM`
  Hard cap on the retained bond dimension during each local update.
- `mindim=1`
  Minimum retained bond dimension on each updated internal bond.
- `truncerr=0.0`
  Target discarded weight used to choose the smallest admissible kept bond
  dimension on each updated internal bond. Set `truncerr = 0` for fixed-cap
  evolution.
- `svd_min=SVDTOL`
  Optional absolute singular-value floor applied after the discarded-weight
  target is evaluated.
- `renormalize=true`
  Whether to renormalize Schmidt values after decomposition.
- `recanonicalize=false`
  Whether to re-canonicalise the unit cell after each wrap-around gate. The
  default `false` is the fast path for unitary real-time evolution. For
  imaginary-time evolution to a ground state, pass `recanonicalize=true` so
  every step truncates in the canonical basis and the final observables are
  accurate.
- `gauge=nothing`
  Pass `:gram`, `:qr`, or `:auto` for a fixed-bond-dimension checkpoint
  after each full gate-list sweep. TensorKit supports `:gram`.
  Requires `renormalize=true` and `recanonicalize=false`. If a sweep or
  its checkpoint fails, the state retains the last completed sweep.
- `return_stats=false`
  If `true`, also return aggregated truncation diagnostics for the full
  evolution.

Returns:
- `ψ` when `return_stats = false`.
- `(ψ, stats)` when `return_stats = true`.

Notes:
- `truncerr = 0` reproduces the standard fixed-cap behavior with `maxdim` as the
  hard cap.
- `truncerr > 0` enables standard local SVD truncation control based on the
  discarded weight of the post-gate singular values.
- The `gates` argument is kept explicit rather than hiding the sweep pattern in
  a higher-level model object.

Example:
```julia
X = [0 1; 1 0]
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
gates = [(kron(X, X), 1, 2), (kron(X, X), 2, 1)]
evolve!(psi, gates, 5; maxdim=4)
```
"""
function evolve!(
    ψ::iMPS,
    gates,
    steps::Integer;
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Union{Nothing,Real}=nothing,
    cutoff::Union{Nothing,Real}=nothing,
    chi_policy::Symbol=:fixed,
    q::Real=1.0,
    alpha::Real=0.1,
    renormalize::Bool=true,
    recanonicalize::Bool=false,
    gauge::Union{Nothing,Symbol}=nothing,
    return_stats::Bool=false
)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_evolve_args(steps, maxdim, mindim, truncerr, svd_floor)
    _validate_chi_policy(chi_policy)
    _validate_gauge_checkpoint(ψ, gauge, renormalize, recanonicalize)

    gate_updates = return_stats ? Any[] : nothing
    for _ in 1:steps
        trial = isnothing(gauge) ? ψ : deepcopy(ψ)
        updates = _evolve_gate_sequence!(
            trial,
            gates;
            maxdim,
            mindim,
            truncerr,
            svd_min=svd_floor,
            chi_policy,
            q,
            alpha,
            renormalize,
            recanonicalize,
            return_stats,
        )
        if !isnothing(gauge)
            restore_canonical!(trial; method=gauge)
            ψ.Γ .= trial.Γ
            ψ.λ .= trial.λ
        end
        return_stats && append!(gate_updates, updates)
    end

    return_stats ? (ψ, _aggregate_evolution_stats(gate_updates)) : ψ
end

"""
    evolve!(ψ, layers, dt, steps; trotter=:second, evolution=:real, maxdim=MAXDIM, mindim=1, truncerr=0.0, svd_min=SVDTOL, renormalize=true, gauge=nothing, return_stats=false)

Evolve `ψ` under a Hamiltonian split into commuting layers.

Each layer is a collection of local Hamiltonian terms `(h, i, j)`. The full
evolution interval is first expanded into a gate sequence with
[`trotter_gates`](@ref), after which the resulting dense gates are applied in
place with the same local truncation controls as the gate-list [`evolve!`](@ref)
method.

Parameters:
- `ψ`
  State to evolve in place.
- `layers`
  Hamiltonian split into commuting layers. Each local term is a tuple
  `(h, i, j)` where `h` is a dense local operator supported on `i:j`.
- `dt`
  Microscopic time step.
- `steps`
  Number of Trotter macro-steps.

Keyword arguments:
- `trotter=:second`
  Trotter scheme. Supported values are `:second`, `:fourth`, and
  `:fourth_opt`.
- `evolution=:real`
  Use `:real` for real-time gates `exp(-1im * c * dt * h)` or `:imaginary` for
  second-order imaginary-time gates `exp(-c * dt * h)`.
- `maxdim`, `mindim`, `truncerr`, `svd_min`, `renormalize`, `recanonicalize`, `return_stats`
  Match the gate-list [`evolve!`](@ref) interface. In particular,
  `recanonicalize=false` by default — pass `recanonicalize=true` for
  imaginary-time evolution.

Returns:
- The same object `ψ`, mutated in place.

Notes:
- Terms inside one layer are applied in the supplied order, and the caller is
  responsible for ensuring that they commute.
- `trotter = :fourth_opt` requires exactly two layers.
- Higher-order Trotter schemes currently reject `evolution = :imaginary`.
- A `gauge` checkpoint keeps Trotter macro-step boundaries explicit, so
  adjacent half-steps across those boundaries are applied separately.

Example:
```julia
X = [0 1; 1 0]
H = kron(X, X)
layers = [[(H, 1, 2)], [(H, 2, 1)]]
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
evolve!(psi, layers, 0.1, 5; trotter=:fourth, maxdim=8)
```
"""
function evolve!(
    ψ::iMPS,
    layers,
    dt::Real,
    steps::Integer;
    trotter::Symbol=:second,
    evolution::Symbol=:real,
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Union{Nothing,Real}=nothing,
    cutoff::Union{Nothing,Real}=nothing,
    chi_policy::Symbol=:fixed,
    q::Real=1.0,
    alpha::Real=0.1,
    renormalize::Bool=true,
    recanonicalize::Bool=false,
    gauge::Union{Nothing,Symbol}=nothing,
    return_stats::Bool=false
)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_evolve_args(steps, maxdim, mindim, truncerr, svd_floor)
    _validate_chi_policy(chi_policy)
    _validate_gauge_checkpoint(ψ, gauge, renormalize, recanonicalize)

    if !isnothing(gauge) && steps > 0
        gates = trotter_gates(layers, dt; trotter, evolution)
        isempty(gates) || _check_gate_eltype(ψ, gates[1][1])
        return evolve!(ψ, gates, steps; maxdim, mindim, truncerr, svd_min=svd_floor,
            chi_policy, q, alpha, renormalize, gauge, return_stats)
    end
    gates = _iterated_trotter_gates(layers, dt, steps; trotter, evolution)
    # Fail before touching ψ: real-time gates are complex, so a default
    # Float64 state would otherwise error mid-sweep with part of the unit
    # cell already updated.
    head = iterate(gates)
    isnothing(head) || _check_gate_eltype(ψ, head[1][1])

    updates = _evolve_gate_sequence!(
        ψ,
        gates;
        maxdim,
        mindim,
        truncerr,
        svd_min=svd_floor,
        chi_policy,
        q,
        alpha,
        renormalize,
        recanonicalize,
        return_stats,
    )
    return_stats ? (ψ, _aggregate_evolution_stats(updates)) : ψ
end

#---------------------------------------------------------------------------------------------------
# Multi-Site Operators
#---------------------------------------------------------------------------------------------------
"""
    _operator_to_internal(mat, dims)

Map a multi-site operator from the documented user convention — `kron(A, B)`
acts `A` on the leftmost site of the support — to the internal fused-leg
layout used by [`tensor_applygate!`](@ref) and `ocontract`, where the leftmost
site's physical index is the *fastest* index of the fused leg. Julia's `kron`
makes the rightmost factor's index fastest, so the two conventions are mirror
images of each other: the map reverses the site order of the operator.

`dims` lists the per-site physical dimensions of the support from left to
right. The map is an involution when the dimensions are uniform.

User-facing routines (`applygate!`, `expect`, and everything built on them)
perform this conversion internally; only callers of the low-level fused-leg
helpers need it explicitly.
"""
function _operator_to_internal(mat::AbstractMatrix, dims::AbstractVector{<:Integer})
    n = length(dims)
    n <= 1 && return mat
    D = prod(dims)
    size(mat) == (D, D) || throw(DimensionMismatch(
        "operator has size $(size(mat)); expected ($D, $D) for site dimensions $dims"))
    rd = reverse(collect(Int, dims))
    tensor = reshape(mat, (rd..., rd...))
    tensor = permutedims(tensor, [n:-1:1; 2n:-1:n+1])
    return reshape(tensor, D, D)
end

function _operator_to_internal(mat::Diagonal, dims::AbstractVector{<:Integer})
    length(dims) <= 1 && return mat
    length(mat.diag) == prod(dims) || throw(DimensionMismatch(
        "diagonal operator has size $(size(mat)); incompatible with site dimensions $dims"))
    order = length(dims):-1:1
    return Diagonal(vec(permutedims(reshape(mat.diag, Tuple(reverse(dims))), order)))
end

"""
    convert_operator(mat, d, n)

Reverse the site ordering of a `d^n × d^n` local operator, converting between
the documented `kron` convention (leftmost site ↔ leftmost factor) and the
package's internal fused-leg layout (leftmost site's index fastest).

Parameters:
- `mat`
  Dense operator acting on `n` sites of local dimension `d`.
- `d`
  Local Hilbert-space dimension per site.
- `n`
  Number of sites acted on by the operator.

Returns:
- A dense matrix with the same shape as `mat` and the site order reversed:
  for example `convert_operator(kron(A, B), d, 2) == kron(B, A)`.

Notes:
- The map is an involution: applying it twice returns the original operator.
- User-facing routines such as [`applygate!`](@ref) and `expect` accept
  operators in the documented `kron` convention and perform this conversion
  internally, so most users never need to call it. It remains useful when
  driving the low-level fused-leg helpers (`tensor_applygate!`, `ocontract`)
  directly.
"""
function convert_operator(mat::AbstractMatrix, d::Integer, n::Integer)
    return _operator_to_internal(mat, fill(Int(d), n))
end
