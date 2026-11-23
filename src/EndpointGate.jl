export EndpointGate, ControlledGate, apply_endpoint_gate!

struct EndpointGate{L<:AbstractMatrix,R<:AbstractMatrix}
    left::Vector{L}
    right::Vector{R}

    function EndpointGate(left::Vector{L}, right::Vector{R}) where {L<:AbstractMatrix,R<:AbstractMatrix}
        length(left) == length(right) > 0 || throw(ArgumentError(
            "endpoint factor lists must have the same positive length"))
        for factors in (left, right)
            d = size(first(factors), 1)
            d > 0 && all(A -> size(A) == (d, d), factors) ||
                throw(DimensionMismatch("endpoint factors must share a square site dimension"))
        end
        return new{L,R}(left, right)
    end
end

function _endpoint_svd(M, op_rtol)
    factors = svd(M)
    rtol = isnothing(op_rtol) ? eps(eltype(factors.S)) * maximum(size(M)) : op_rtol
    rank = max(1, count(>(rtol * first(factors.S)), factors.S))
    return factors.U, sqrt.(factors.S[1:rank]), factors.Vt
end

function _endpoint_factors(G::AbstractMatrix, dl, dr, op_rtol)
    # Reshuffle (outL,outR; inL,inR) into (outL,inL; outR,inR).
    M = reshape(permutedims(reshape(G, dr, dl, dr, dl), (2, 4, 1, 3)), dl^2, dr^2)
    U, weights, Vt = _endpoint_svd(M, op_rtol)
    left = [reshape(U[:, k] .* weights[k], dl, dl) for k in eachindex(weights)]
    right = [reshape(Vt[k, :] .* weights[k], dr, dr) for k in eachindex(weights)]
    return left, right
end

function _endpoint_factors(G::Diagonal, dl, dr, op_rtol)
    M = transpose(reshape(G.diag, dr, dl))
    U, weights, Vt = _endpoint_svd(M, op_rtol)
    left = [Diagonal(U[:, k] .* weights[k]) for k in eachindex(weights)]
    right = [Diagonal(Vt[k, :] .* weights[k]) for k in eachindex(weights)]
    return left, right
end

function _validate_endpoint_operator(G, dl, dr, op_rtol)
    dl > 0 && dr > 0 || throw(ArgumentError("endpoint dimensions must be positive"))
    size(G) == (dl*dr, dl*dr) || throw(DimensionMismatch("gate size must match the endpoint dimensions"))
    isnothing(op_rtol) || isfinite(op_rtol) && 0 <= op_rtol < 1 ||
        throw(ArgumentError("op_rtol must be in [0, 1)"))
    _finite_entries(G isa Diagonal ? G.diag : G, "endpoint gate")
    return nothing
end

"""
    EndpointGate(G, (dleft, dright); op_rtol=nothing)

Prepare the operator Schmidt factors of a two-site gate once. Reuse the
result with [`applygate!`](@ref) or in an `evolve!` list of
`(gate, i, j)` tuples. The factor matrices are shared across applications.
`op_rtol` defaults to a roundoff threshold. Use zero to retain all
positive operator Schmidt values.
"""
function EndpointGate(G::AbstractMatrix, dims::Tuple{Integer,Integer}; op_rtol::Union{Nothing,Real}=nothing)
    dl, dr = dims
    _validate_endpoint_operator(G, dl, dr, op_rtol)
    return EndpointGate(_endpoint_factors(G, dl, dr, op_rtol)...)
end

"""
    EndpointGate(A, B, scale; op_rtol=nothing)

Prepare `exp(scale * kron(A, B))` for Hermitian local operators `A` and
`B`, directly from their local eigensystems. This uses a phase table of
size `size(A, 1) × size(B, 1)` in place of a fused two-site exponential.
`op_rtol` controls the relative cutoff in the operator Schmidt decomposition.
Use `scale=-im*dt` for real-time evolution.
"""
function EndpointGate(A::AbstractMatrix, B::AbstractMatrix, scale::Number;
                      op_rtol::Union{Nothing,Real}=nothing)
    isnothing(op_rtol) || isfinite(op_rtol) && 0 <= op_rtol < 1 ||
        throw(ArgumentError("op_rtol must be in [0, 1)"))
    isfinite(scale) || throw(ArgumentError("gate scale must be finite"))
    a, left_basis = _gate_eigenbasis(A)
    b, right_basis = _gate_eigenbasis(B)
    phases = exp.(scale .* (a * transpose(b)))
    _finite_entries(phases, "gate phase table")
    U, weights, Vt = _endpoint_svd(phases, op_rtol)
    left = [_rotate_gate_diagonal(left_basis, U[:, k] .* weights[k]) for k in eachindex(weights)]
    right = [_rotate_gate_diagonal(right_basis, Vt[k, :] .* weights[k]) for k in eachindex(weights)]
    return EndpointGate(left, right)
end

function _gate_eigenbasis(A)
    size(A, 1) > 0 && ishermitian(A) ||
        throw(ArgumentError("local gate factors must be nonempty Hermitian matrices"))
    _finite_entries(A isa Diagonal ? A.diag : A, "local gate factor")
    A isa Diagonal && return (real.(A.diag), nothing)
    factor = eigen(Hermitian(A))
    return factor.values, factor.vectors
end

_rotate_gate_diagonal(::Nothing, values) = Diagonal(values)
_rotate_gate_diagonal(basis, values) = (basis .* transpose(values)) * basis'

"""
    ControlledGate(left::Diagonal, U::AbstractMatrix, right::Diagonal)

Prepare a three-site gate which acts with `U` on the central site when
both outer sites are in the diagonal projectors `left` and `right`.
Elsewhere the gate is the identity. The projectors have entries 0 or 1.
Use the prepared gate with `applygate!` or an `evolve!` gate list.
"""
struct ControlledGate{E<:EndpointGate,C<:AbstractMatrix}
    ends::E
    center_delta::C

    function ControlledGate(left::Diagonal, U::AbstractMatrix, right::Diagonal)
        for P in (left, right)
            !isempty(P.diag) && all(x -> iszero(x) || isone(x), P.diag) ||
                throw(ArgumentError("controls must be nonempty diagonal projectors"))
        end
        size(U, 1) == size(U, 2) > 0 ||
            throw(ArgumentError("central gate must be a nonempty square matrix"))
        _finite_entries(U isa Diagonal ? U.diag : U, "central gate")
        left_ops = [Diagonal(fill(true, size(left, 1))), Diagonal(Vector{Bool}(isone.(left.diag)))]
        right_ops = [Diagonal(fill(true, size(right, 1))), Diagonal(Vector{Bool}(isone.(right.diag)))]
        ends = EndpointGate(left_ops, right_ops)
        center_delta = U - I
        return new{typeof(ends),typeof(center_delta)}(ends, center_delta)
    end
end

function _endpoint_right_cores(tensors, right_ops, ::Type{T}, middle, mindim) where {T}
    n, w = length(tensors), length(right_ops)
    cores = Vector{Array{T,3}}(undef, n)
    last_tensor = last(tensors)
    dl, d, dr = size(last_tensor)
    block = Array{T}(undef, w * dl, d, dr)
    for k in 1:w
        block[(k-1)*dl+1:k*dl, :, :] .= tensor_umul(right_ops[k], last_tensor)
    end
    # Right-isometrize the gated interval without forming its physical
    # tensor. LQ is untruncated. The carry connects the MPO channels to
    # the next site; an intermediate physical leg only sees the identity.
    for i in n:-1:2
        l, d, r = size(block)
        matrix = reshape(block, l, d * r)
        minrows = min(mindim, d * r)
        if minrows > l
            # Preserve the available zero-weight directions when mindim
            # requests them. Only the original channel rows are propagated.
            padded = zeros(T, minrows, d * r)
            copyto!(view(padded, 1:l, :), matrix)
            factor = lq!(padded)
        else
            factor = lq(matrix)
        end
        carry = Matrix(factor.L)
        cores[i] = reshape(Matrix(factor.Q), size(carry, 2), d, r)
        size(carry, 1) > l && (carry = carry[1:l, :])
        i == 2 && return cores, carry

        A = tensors[i-1]
        dl, d, dr = size(A)
        q = size(carry, 2)
        block = Array{T}(undef, dl * w, d, q)
        for k in 1:w
            rows = (k-1)*dr+1:k*dr
            # The controlled three-site gate is I + P_L ⊗ (U - I) ⊗ P_R.
            local_tensor = !isnothing(middle) && i == 3 && k == 2 ? tensor_umul(middle, A) : A
            propagated = reshape(local_tensor, dl*d, dr) * view(carry, rows, :)
            block[(k-1)*dl+1:k*dl, :, :] .= reshape(propagated, dl, d, q)
        end
    end
end

function _endpoint_mpo!(ψ, gate::EndpointGate, inds; maxdim, mindim, truncerr, svd_min,
                        renormalize, return_stats, middle=nothing)
    tensors = ψ.Γ[inds]
    left_ops, right_ops = gate.left, gate.right
    T = promote_type(eltype(ψ), eltype(first(left_ops)), eltype(first(right_ops)))
    isnothing(middle) || (T = promote_type(T, eltype(middle)))
    cores, carry = _endpoint_right_cores(tensors, right_ops, T, middle, mindim)
    dl, d, dr = size(first(tensors))
    block = zeros(T, dl*d, size(carry, 2))
    for k in eachindex(left_ops)
        A = reshape(tensor_umul(left_ops[k], first(tensors)), dl*d, dr)
        mul!(block, A, view(carry, (k-1)*dr+1:k*dr, :), true, true)
    end
    block = reshape(block, dl, d, :)
    λ = ψ.λ[mod1(first(inds)-1, ψ.n)]
    stats = BondStat[]
    # Split from the same end as the contiguous dense kernel. Because the
    # remaining cores are right-isometric, these SVDs see the same cut spectra.
    for i in 1:length(inds)-1
        tensor_lmul!(λ, block)
        l, d, r = size(block)
        U, S, Vt, choice = _svd_truncate_by_error(reshape(block, l*d, r);
            maxdim, mindim, truncerr, svd_min, renormalize)
        A = reshape(U, l, d, :)
        tensor_lmul!(_safe_reciprocal(λ), A)
        tensor_rmul!(A, S)
        bond = inds[i]
        ψ.Γ[bond], ψ.λ[bond] = A, S
        if return_stats
            push!(stats, BondStat(bond, choice.chi_req, choice.chi_keep,
                choice.discarded_weight, choice.saturated, choice.target_met,
                choice.smallest_kept_sv, choice.largest_discarded_sv))
        end
        core = cores[i+1]
        _, d, r = size(core)
        block = reshape(Vt * reshape(core, :, d*r), length(S), d, r)
        λ = S
    end
    ψ.Γ[last(inds)] = block
    return stats
end

function _endpoint_swap!(ψ, i, j; maxdim, mindim, truncerr, svd_min,
                         renormalize, return_stats)
    λl = ψ.λ[mod1(i-1, ψ.n)]
    A, B = ψ.Γ[i], ψ.Γ[j]
    l, d, r = size(A, 1), size(A, 2), size(B, 3)
    fused = tensor_group_2(A, B)
    block = reshape(permutedims(reshape(fused, l, d, d, r), (1, 3, 2, 4)), l, d*d, r)
    tensor_lmul!(λl, block)
    result = tensor_decomp!(block, λl, 2; maxdim, mindim, truncerr, svd_min,
                             renormalize, return_stats, bond_indices=i:i)
    ψ.Γ[i], ψ.Γ[j] = result[1]
    ψ.λ[i] = only(result[2])
    return return_stats ? result[3] : BondStat[]
end

function _endpoint_indices(ψ, i, j)
    abs(big(j) - big(i)) < ψ.n || throw(ArgumentError("endpoint interval spans more than one unit cell"))
    i0, j0 = _normalize_gate_sites(ψ, i, j)
    i0 != j0 || throw(ArgumentError("endpoint gates require distinct sites"))
    return _gate_indices(ψ, i0, j0)
end

function _check_endpoint_dimensions(ψ, G::EndpointGate, inds)
    dl, dr = size(ψ.Γ[first(inds)], 2), size(ψ.Γ[last(inds)], 2)
    size(first(G.left)) == (dl, dl) && size(first(G.right)) == (dr, dr) ||
        throw(DimensionMismatch("prepared gate dimensions must match the endpoints"))
    return nothing
end

function _finish_endpoint_update!(ψ, trial, inds, stats, controls; recanonicalize, gauge)
    recanonicalize && canonical!(trial; maxdim=controls.maxdim, cutoff=controls.svd_min,
                                  renormalize=controls.renormalize)
    isnothing(gauge) || restore_canonical!(trial; method=gauge)
    ψ.Γ .= trial.Γ
    ψ.λ .= trial.λ
    return controls.return_stats ? (ψ, _gate_update_stats(first(inds), last(inds), stats)) : ψ
end

function _apply_factored_endpoint!(ψ, gate, inds, controls; middle=nothing, recanonicalize, gauge)
    # The kernel replaces tensors and spectra, so copying their containers
    # suffices to keep the input intact until the update and gauge checks pass.
    trial = iMPS(copy(ψ.Γ), copy(ψ.λ), ψ.n)
    stats = _endpoint_mpo!(trial, gate, inds; controls..., middle)
    return _finish_endpoint_update!(ψ, trial, inds, stats, controls; recanonicalize, gauge)
end

"""
    applygate!(ψ, gate::Union{EndpointGate,ControlledGate}, i, j; kwargs...)

Apply a prepared gate on the forward periodic interval from `i` to `j`.
The interval must fit in one unit cell. [`EndpointGate`](@ref) acts on its
two ends. [`ControlledGate`](@ref) requires a three-site interval.

The usual truncation controls apply to the state. Set the operator Schmidt
cutoff `op_rtol` when constructing an `EndpointGate`.
`return_stats=true` reports local bond cuts in sweep order.
`recanonicalize=true` runs `canonical!` after an endpoint update or after
a controlled update that crosses the cell seam. This can truncate bonds
outside the interval and is excluded from the local cut statistics. Use
`gauge=:auto` (or `:gram` or `:qr`) for fixed-dimension gauge restoration.
"""
function applygate!(ψ::DenseIMPS, gate::Union{EndpointGate,ControlledGate}, i::Integer, j::Integer;
    maxdim::Integer=MAXDIM, mindim::Integer=1, truncerr::Real=0.0,
    svd_min::Union{Nothing,Real}=nothing, cutoff::Union{Nothing,Real}=nothing,
    renormalize::Bool=true, recanonicalize::Bool=false, gauge::Union{Nothing,Symbol}=nothing,
    return_stats::Bool=false)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_truncation_args(maxdim, mindim, truncerr, svd_floor)
    _validate_gauge_checkpoint(ψ, gauge, renormalize, recanonicalize)
    inds = _endpoint_indices(ψ, i, j)
    if gate isa ControlledGate
        length(inds) == 3 || throw(ArgumentError("a ControlledGate requires a three-site interval"))
        ends, middle = gate.ends, gate.center_delta
        _check_endpoint_dimensions(ψ, ends, inds)
        d = size(ψ.Γ[inds[2]], 2)
        size(middle) == (d, d) ||
            throw(DimensionMismatch("central gate dimension must match the central site"))
        _check_gate_eltype(ψ, middle)
        recanonicalize = recanonicalize && last(inds) < first(inds)
    else
        ends, middle = gate, nothing
        _check_endpoint_dimensions(ψ, ends, inds)
        _check_gate_eltype(ψ, first(ends.left))
        _check_gate_eltype(ψ, first(ends.right))
    end

    controls = (; maxdim, mindim, truncerr, svd_min=svd_floor, renormalize, return_stats)
    return _apply_factored_endpoint!(ψ, ends, inds, controls; middle, recanonicalize, gauge)
end

"""
    apply_endpoint_gate!(ψ, G, i, j; method=:mpo, op_rtol=nothing, kwargs...)

Compatibility entry point for a two-site matrix in the usual
`kron(left, right)` convention acting at the ends of the forward interval
from `i` to `j`. For prepared gates use [`applygate!`](@ref).

`method=:mpo` factors the matrix using the relative operator Schmidt cutoff
`op_rtol`. `method=:swap` uses adjacent ordinary swaps, requires equal
physical dimensions along the interval, and can truncate during routing.
Both methods accept the usual state truncation and gauge options. With
`return_stats=true`, report the local cuts in update order.
"""
function apply_endpoint_gate!(ψ::DenseIMPS, G::AbstractMatrix, i::Integer, j::Integer;
    method::Symbol=:mpo, op_rtol::Union{Nothing,Real}=nothing,
    maxdim::Integer=MAXDIM, mindim::Integer=1, truncerr::Real=0.0,
    svd_min::Union{Nothing,Real}=nothing, cutoff::Union{Nothing,Real}=nothing,
    renormalize::Bool=true, recanonicalize::Bool=false, gauge::Union{Nothing,Symbol}=nothing,
    return_stats::Bool=false)
    method in (:mpo, :swap) || throw(ArgumentError("method must be :mpo or :swap"))
    method === :mpo || isnothing(op_rtol) || throw(ArgumentError("op_rtol requires method=:mpo"))
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_truncation_args(maxdim, mindim, truncerr, svd_floor)
    _validate_gauge_checkpoint(ψ, gauge, renormalize, recanonicalize)
    inds = _endpoint_indices(ψ, i, j)
    i0, j0 = first(inds), last(inds)
    dl, dr = size(ψ.Γ[i0], 2), size(ψ.Γ[j0], 2)
    _check_gate_eltype(ψ, G)
    _validate_endpoint_operator(G, dl, dr, op_rtol)
    controls = (; maxdim, mindim, truncerr, svd_min=svd_floor, renormalize, return_stats)
    if length(inds) == 2 && dl == dr && isnothing(op_rtol) && !recanonicalize
        return applygate!(ψ, G, i0, j0; controls..., gauge)
    end
    if method === :mpo
        gate = EndpointGate(_endpoint_factors(G, dl, dr, op_rtol)...)
        return _apply_factored_endpoint!(ψ, gate, inds, controls; recanonicalize, gauge)
    end

    all(k -> size(ψ.Γ[k], 2) == dl, inds) || throw(ArgumentError(
        "the SWAP route requires equal physical dimensions along the interval"))
    trial = iMPS(copy(ψ.Γ), copy(ψ.λ), ψ.n)
    stats = BondStat[]
    for k in length(inds)-1:-1:2
        append!(stats, _endpoint_swap!(trial, inds[k], inds[k+1]; controls...))
    end
    result = applygate!(trial, G, inds[1], inds[2]; controls...)
    return_stats && append!(stats, result[2].bond_stats)
    for k in 2:length(inds)-1
        append!(stats, _endpoint_swap!(trial, inds[k], inds[k+1]; controls...))
    end
    return _finish_endpoint_update!(ψ, trial, inds, stats, controls; recanonicalize, gauge)
end

function apply_endpoint_gate!(ψ::DenseIMPS, gate::EndpointGate, i::Integer, j::Integer;
    method::Symbol=:mpo, op_rtol::Union{Nothing,Real}=nothing, kwargs...)
    method === :mpo && isnothing(op_rtol) || throw(ArgumentError(
        "an already prepared EndpointGate uses its existing MPO factors"))
    return applygate!(ψ, gate, i, j; kwargs...)
end

"""
    expect(ψ, op::EndpointGate, i, j)

Measure a factored two-site operator on a Schmidt-canonical dense iMPS.
Transfers across the intermediate sites are contracted individually.
"""
function expect(ψ::DenseIMPS, op::EndpointGate, i::Integer, j::Integer)
    inds = _endpoint_indices(ψ, i, j)
    _check_endpoint_dimensions(ψ, op, inds)
    A, B = ψ.Γ[first(inds)], ψ.Γ[last(inds)]
    χr = size(B, 3)
    boundary = Matrix{eltype(ψ)}(I, χr, χr)
    weights = abs2.(ψ.λ[mod1(first(inds)-1, ψ.n)])
    total = zero(promote_type(eltype(ψ), eltype(first(op.left)), eltype(first(op.right))))
    for (left_op, right_op) in zip(op.left, op.right)
        ρ = apply_transfer(B, tensor_umul(right_op, B), boundary)
        for k in length(inds)-1:-1:2
            C = ψ.Γ[inds[k]]
            ρ = apply_transfer(C, C, ρ)
        end
        ρ = apply_transfer(A, tensor_umul(left_op, A), ρ)
        total += dot(weights, diag(ρ))
    end
    return real(total)
end
