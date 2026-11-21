# Private adjacent-gate prototype. A projection tolerance certifies the
# written state, not the smallest acceptable rank in the full spectrum.
function _thin_qr_basis(A, k)
    return qr(A).Q[:, 1:k]
end

function _projected_right_basis(M, seed, width)
    k = min(size(seed, 2), width)
    Q = k == 0 ? similar(M, size(M, 2), 0) : _thin_qr_basis(seed, k)
    if k < width
        Y = M * Q
        # Enrich using rows with the most weight outside the old right
        # space, without forming M - (M Q) Q† over the full cut.
        weights = vec(sum(abs2, M; dims=2) - sum(abs2, Y; dims=2))
        rows = partialsortperm(weights, 1:width-k; rev=true)
        directions = M[rows, :]' - Q * Y[rows, :]'
        Q = _thin_qr_basis(hcat(Q, directions), width)
    end
    left = _thin_qr_basis(M * Q, width)
    return _thin_qr_basis(M' * left, width)
end

function _qr_two_site_split(M, seed; maxdim, projection_tol, expansion, svd_min=SVDTOL)
    isfinite(projection_tol) && 0 <= projection_tol <= 1 ||
        throw(ArgumentError("projection_tol must be in [0, 1]"))
    expansion >= 0 || throw(ArgumentError("expansion must be nonnegative"))
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    size(seed, 1) == size(M, 2) && size(seed, 2) > 0 ||
        throw(DimensionMismatch("seed must span the right leg of the cut"))
    _finite_entries(M, "QR-TEBD cut")
    weight = sum(abs2, M)
    isfinite(weight) && weight > 0 || throw(ArgumentError("invalid QR-TEBD cut norm"))

    limit = min(size(M)...)
    width = min(limit, max(maxdim, size(seed, 2)) + expansion)
    if width < limit
        Q = _projected_right_basis(M, seed, width)
        # Polish the *right*-projected state, which is what C*V writes
        # back. A two-sided projected core would certify another state.
        F = svd(M * Q)
        k = max(1, min(maxdim, count(>=(svd_min), F.S)))
        S, Vt = F.S[1:k], F.Vt[1:k, :] * Q'
        discarded = max(0, 1 - sum(abs2, S) / weight)
        roundoff = 100 * eps(typeof(weight))
        if discarded <= roundoff || abs(discarded - projection_tol) <= roundoff
            discarded = sum(abs2, M - (M * Vt') * Vt) / weight
        end
        if discarded <= projection_tol
            return S, Vt, (used_qr=true, projection_error=discarded, target_met=true, width)
        end
    end

    F, choice = _svd_rank_choice(M; maxdim, mindim=1, truncerr=0.0, svd_min)
    k = choice.chi_keep
    S, Vt = F.S[1:k], F.Vt[1:k, :]
    discarded = choice.discarded_weight
    if discarded <= 100 * eps(typeof(weight))
        discarded = sum(abs2, M - (M * Vt') * Vt) / weight
    end
    return S, Vt, (used_qr=false, projection_error=discarded,
                    target_met=discarded <= projection_tol, width)
end

# Experimental caller supplies a real-time unitary gate and canonical input.
# The agreement with the existing truncation API is explicit: no change to
# applygate!, truncerr, or BondStat. An unmet target may still return the
# optimal full-SVD result at maxdim, with target_met=false.
function _applygate_qr_experiment!(ψ::DenseIMPS, G::AbstractMatrix, i::Integer, j::Integer;
                                   maxdim::Integer, projection_tol::Real,
                                   expansion::Integer=max(2, cld(maxdim, 4)))
    i0, j0 = _normalize_gate_sites(ψ, i, j)
    ψ.n >= 2 && j0 == mod1(i0+1, ψ.n) ||
        throw(ArgumentError("QR-TEBD experiment requires adjacent sites in a multi-site cell"))
    _check_gate_eltype(ψ, G)
    A, B = ψ.Γ[i0], ψ.Γ[j0]
    l, dl, _ = size(A)
    _, dr, r = size(B)
    C = tensor_umul(_operator_to_internal(G, [dl, dr]), tensor_group_2(A, B))
    λl = ψ.λ[mod1(i0-1, ψ.n)]
    weighted = copy(C)
    tensor_lmul!(λl, weighted)
    seed = adjoint(reshape(B, :, dr*r))
    S, Vt, info = _qr_two_site_split(reshape(weighted, l*dl, :), seed;
                                    maxdim, projection_tol, expansion)
    tensors, spectra = _reconstruct_two_site(C, λl, S, Vt, dl, dr; renormalize=true)
    ψ.Γ[i0], ψ.Γ[j0] = tensors
    ψ.λ[i0] = only(spectra)
    return info
end
