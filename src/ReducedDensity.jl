export reduced_density, measurement_environments

struct _TransferEnvironments{T}
    tensors::Vector{Array{T,3}}
    left_roots::Vector{Matrix{T}}
    right_roots::Vector{Matrix{T}}
end

"""
    measurement_environments(ψ; tol=nothing, maxiter=nothing)

Prepare a measurement snapshot of a dense iMPS in an arbitrary gauge. The
snapshot owns its tensors and the left and right transfer environments on
each bond. Use it for repeated [`reduced_density`](@ref) calls on this state,
even after evolving or otherwise editing `ψ`. Preparing the snapshot does not
change the state or its Schmidt values.

The transfer must have a unique dominant sector. The method checks for
peripheral degeneracy on small seams and assumes uniqueness on larger seams.
Nonconvergence throws an error. `tol` and `maxiter` control the matrix-free
seam solves.
"""
function measurement_environments(ψ::DenseIMPS; tol::Union{Nothing,Real}=nothing,
                                  maxiter::Union{Nothing,Integer}=nothing)
    T = float(eltype(ψ))
    ϵ = eps(real(T))
    tol = isnothing(tol) ? max(1e-12, 10ϵ) : tol
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    isnothing(maxiter) || maxiter > 0 || throw(ArgumentError("maxiter must be positive"))

    # Independent site scalings cancel in normalized measurements. Rescale
    # the owned copy to keep raw-state magnitudes out of the transfer solves.
    tensors = map(ψ.Γ) do A
        scale = maximum(abs, A)
        isfinite(scale) && scale > 0 || throw(ArgumentError("invalid tensor scale"))
        Array{T,3}(A ./ scale)
    end
    n = length(tensors)
    seam = argmin(j -> size(tensors[j], 3), eachindex(tensors))
    order = [seam+1:n; 1:seam]
    cell = tensors[order]
    _check_small_transfer_uniqueness(cell; message="measurement needs a unique dominant transfer sector")

    left_roots = Vector{Matrix{T}}(undef, n)
    right_roots = similar(left_roots)
    threshold = max(10tol, 100ϵ)
    seam_right = _cell_environment(cell, :r; tol, maxiter)
    right = seam_right
    for j in reverse(order)
        A = tensors[j]
        right_roots[j] = _rdm_environment_root(right, threshold)
        right, _ = _normalize_cell_environment(apply_transfer(A, A, right))
    end
    norm(right - seam_right) <= threshold ||
        error("right transfer environment did not close the seam")

    # The library uses a transposed left transfer. Conjugate its environment
    # to obtain the adjoint-channel environment multiplying the ket tensor.
    seam_left = _cell_environment(cell, :l; tol, maxiter)
    left = seam_left
    for j in order
        A = tensors[j]
        left, _ = _normalize_cell_environment(apply_transfer(A, A, left; dir=:l))
        left_roots[j] = conj!(_rdm_environment_root(left, threshold))
    end
    norm(left - seam_left) <= threshold ||
        error("left transfer environment did not close the seam")
    return _TransferEnvironments(tensors, left_roots, right_roots)
end

function _rdm_environment_root(ρ, threshold)
    values, vectors = eigen(ρ)
    first(values) >= -threshold * last(values) ||
        throw(ArgumentError("transfer environment is not positive semidefinite"))
    return (vectors .* transpose(sqrt.(max.(values, 0)))) * vectors'
end

"""
    reduced_density(ψ, start, nsites)
    reduced_density(env, start, nsites)

Return the normalized reduced density matrix of `nsites` consecutive sites,
starting at `start`. Indices wrap through the cell and the interval may span
multiple cells. The basis follows the package's `kron` convention, with the
leftmost site as the leftmost factor.

Pass a dense state in normalized Schmidt-canonical form to use its stored
Schmidt values directly. For an arbitrary gauge, pass a snapshot prepared
with [`measurement_environments`](@ref). The same snapshot can be used for
multiple starting positions and window sizes.

The result is dense and stores the square of the product of the local
physical dimensions. Use this for short windows.
"""
function reduced_density(ψ::DenseIMPS, start::Integer, nsites::Integer)
    sites = _rdm_sites(ψ.n, start, nsites)
    incoming = ψ.λ[mod1(first(sites) - 1, ψ.n)]
    return _reduced_density(ψ.Γ[sites], Diagonal(incoming), I)
end

function reduced_density(env::_TransferEnvironments, start::Integer, nsites::Integer)
    n = length(env.tensors)
    sites = _rdm_sites(n, start, nsites)
    left = env.left_roots[mod1(first(sites) - 1, n)]
    right = env.right_roots[last(sites)]
    return _reduced_density(env.tensors[sites], left, right)
end

function _rdm_sites(n, start, nsites)
    nsites > 0 || throw(ArgumentError("nsites must be positive"))
    start = mod1(start, n)
    return [mod1(start + offset, n) for offset in 0:nsites-1]
end

function _reduced_density(tensors, left, right)
    n = length(tensors)
    # The environment weights touch only the outside virtual legs. Apply
    # them before grouping so the matrix products see one physical leg.
    first_tensor = first(tensors)
    weighted_first = reshape(left * reshape(first_tensor, size(first_tensor, 1), :), size(first_tensor))
    last_tensor = isone(n) ? weighted_first : last(tensors)
    weighted_last = if right === I
        last_tensor
    else
        reshape(reshape(last_tensor, :, size(last_tensor, 3)) * right, size(last_tensor))
    end
    block = isone(n) ? weighted_last :
            tensor_group(vcat([weighted_first], view(tensors, 2:n-1), [weighted_last]))
    Dl, p, Dr = size(block)

    # Group both virtual legs as rows and the physical legs as columns.
    # Reverse the physical axes for the public kron basis.
    dims = [size(A, 2) for A in tensors]
    packed = permutedims(reshape(block, Dl, dims..., Dr), (1, n+2, ((n+1):-1:2)...))
    amplitudes = reshape(packed, Dl * Dr, p)
    ρ = conj!(amplitudes' * amplitudes)
    weight = real(tr(ρ))
    isfinite(weight) && weight > 0 || throw(ArgumentError("invalid local density norm"))
    ρ ./= weight
    return Hermitian(ρ)
end
