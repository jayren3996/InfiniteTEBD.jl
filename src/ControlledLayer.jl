#---------------------------------------------------------------------------------------------------
# Sector-resolved update for controlled-center layers (two-site cell, PXP-type gates)
#---------------------------------------------------------------------------------------------------
# A layer of three-site gates on (control, center, control) that are diagonal in the two
# outer occupations, G = Σ_{ab} |a⟩⟨a| ⊗ M_ab ⊗ |b⟩⟨b|, leaves the control sites' occupations
# untouched. Writing each occupation block of the control tensor as a rank factorization
# B_c[:, a, :] = L_a R_a turns the control site into a block identity on the sector bond
# (dimension Σ_a rank_a ≤ 2χ, already free of the null directions an MPO contraction would
# create), and the center tensor into blocks R_a (Σ_s' M_ab[s, s'] B_t[:, s', :]) L_b. This is
# the same state as W·ψ for the layer MPO W.
#
# The canonical form is then found with the single-layer QR/LQ iteration used by the generic
# path (`_right_orthonormalize`, `_left_factors`), except that the factorization through the
# control site splits into independent blocks, one per occupation. The iteration is started
# from the input Schmidt data (exact for a layer acting as the identity) and stops on the same
# seam residual as the generic path. If it does not converge, or anything fails numerically,
# the update falls back to the generic path on the same input, so results never depend on the
# seed. Truncation is the shared per-bond truncation (`_truncate_canonical!`).

# Sweep budget per direction before falling back (a Ref so tests can force the fallback).
const _CONTROLLED_MAXITER = Ref(100)
const _CONTROLLED_DETECT_TOL = 1e-14

# Recognize a controlled-center layer: one three-site gate on a two-site cell, local
# dimension 2, diagonal (to `_CONTROLLED_DETECT_TOL` relative) in its outer sites.
function _detect_controlled_center(terms, n, d)
    (n == 2 && length(terms) == 1 && d == 2) || return nothing
    G, i, j = terms[1]
    a0, _, L, _ = _support_span(G, i, j, n, d)
    L == 3 || return nothing
    Gm = Matrix{ComplexF64}(G)
    scale = maximum(abs, Gm; init=0.0)
    scale > 0 || return nothing
    T = reshape(Gm, 2, 2, 2, 2, 2, 2)          # kron convention: (o3, o2, o1, i3, i2, i1)
    M = zeros(ComplexF64, 2, 2, 2, 2)
    for i1 in 1:2, i2 in 1:2, i3 in 1:2, o1 in 1:2, o2 in 1:2, o3 in 1:2
        v = T[o3, o2, o1, i3, i2, i1]
        if o1 == i1 && o3 == i3
            # entries at the roundoff level of the stored pieces are exact zeros of the gate
            M[o2, i2, o1, o3] = abs(v) > _CONTROLLED_DETECT_TOL * scale ? v : zero(v)
        elseif abs(v) > _CONTROLLED_DETECT_TOL * scale
            return nothing
        end
    end
    return ControlledCenter(mod1(a0, 2), mod1(a0 + 1, 2), M)
end

# Product of two controlled layers on the same sites (`first` acts after `last`).
function _merged_structure(first::ControlledCenter, last::ControlledCenter)
    (first.control == last.control && first.center == last.center) || return nothing
    M = similar(first.M)
    for a in 1:2, b in 1:2
        M[:, :, a, b] = first.M[:, :, a, b] * last.M[:, :, a, b]
    end
    return ControlledCenter(first.control, first.center, M)
end
_merged_structure(::Any, ::Any) = nothing

function _applicable_controlled_structure(ψ, W::iMPO, renormalize::Bool)
    renormalize && ψ isa DenseIMPS{ComplexF64} && ψ.n == 2 && W.n == 2 &&
        all(A -> size(A, 2) == 2, ψ.Γ) || return nothing
    return _valid_structure(W)
end

# Sector representation of W·ψ: block-identity control tensor and the center tensor AT.
function _controlled_tensors(ψ::DenseIMPS, S::ControlledCenter; rtol=_NULL_RTOL)
    BC, BT = ψ.Γ[S.control], ψ.Γ[S.center]
    L = Vector{Matrix{ComplexF64}}(undef, 2)
    R = similar(L)
    sv = Vector{Vector{Float64}}(undef, 2)
    d = zeros(Int, 2)
    for a in 1:2
        F = svd(BC[:, a, :])
        d[a] = isempty(F.S) || !(F.S[1] > 0) ? 0 : count(>(rtol * F.S[1]), F.S)
        sv[a] = F.S[1:d[a]]
        s = sqrt.(sv[a])
        L[a] = F.U[:, 1:d[a]] .* transpose(s)
        R[a] = s .* F.Vt[1:d[a], :]
    end
    D = sum(d)
    D == 0 && return nothing
    ids = (1:d[1], d[1]+1:D)
    AT = zeros(ComplexF64, D, 2, D)
    B0, B1 = BT[:, 1, :], BT[:, 2, :]
    for a in 1:2, b in 1:2
        (d[a] == 0 || d[b] == 0) && continue
        for s in 1:2
            m1, m2 = S.M[s, 1, a, b], S.M[s, 2, a, b]
            (iszero(m1) && iszero(m2)) && continue
            AT[ids[a], s, ids[b]] = R[a] * (m1 .* B0 .+ m2 .* B1) * L[b]
        end
    end
    return (; AT, D, ids, sv, L)
end

# Seeds from the input Schmidt data: right factor √S per sector (right of the center),
# left factor from λ_left · L_a per sector (right of the control).
function _controlled_seeds(ψ, S, ct)
    D, ids = ct.D, ct.ids
    C0 = zeros(ComplexF64, D, D)
    D0 = zeros(ComplexF64, D, D)
    λl = ψ.λ[mod1(S.control - 1, 2)]
    for a in 1:2
        isempty(ids[a]) && continue
        C0[ids[a], ids[a]] = Diagonal(sqrt.(ct.sv[a]))
        D0[ids[a], ids[a]] = last(_pos_qr(λl .* ct.L[a]))
    end
    return C0 ./ norm(C0), D0 ./ norm(D0)
end

# Canonical form of the sector representation; `nothing` if the iteration does not converge.
# Like `_right_orthonormalize`/`_left_factors`, a slowly converging iteration (small transfer
# gap, e.g. near-crystal χ = 2 states) is accelerated by an Arnoldi step on the fixed-point
# map of the seam factor; convergence is always decided by the same residual test.
function _controlled_canonical(ψ, S, ct; orth_tol=nothing, maxiter=_CONTROLLED_MAXITER[])
    AT, D, ids = ct.AT, ct.D, ct.ids
    C0, D0 = _controlled_seeds(ψ, S, ct)
    AT2 = reshape(AT, 2D, D)
    ATr = reshape(AT, D, 2D)
    ATs = (AT[:, 1, :], AT[:, 2, :])
    blocks = [a for a in 1:2 if !isempty(ids[a])]
    # right: through the center (dense LQ), then the control (block LQ)
    CT = RT = RC = nothing
    logc = 0.0
    done, best, since = false, Inf, 0
    δ, δprev = Inf, Inf
    iter = 0
    while !done
        (iter += 1) > maxiter && return nothing
        if iter > _ACCEL_AFTER && δ > _ACCEL_MIN && δ > _ACCEL_RATIO * δprev
            Rs = (RT[:, 1, :], RT[:, 2, :])
            Qs = [RC[ids[a], a, :] for a in 1:2]
            map_right(X) = begin
                Y = ATs[1] * X * Rs[1]' .+ ATs[2] * X * Rs[2]'
                Z = zeros(eltype(Y), D, D)
                for a in blocks
                    Z[ids[a], ids[a]] = Y[ids[a], :] * Qs[a]'
                end
                Z
            end
            X = _accelerate(map_right, C0, δ)
            C = zeros(eltype(C0), D, D)
            for a in blocks
                C[ids[a], ids[a]] = first(_pos_lq(X[ids[a], ids[a]]))
            end
            nC = norm(C)
            (isfinite(nC) && nC > 0) && (C0 = C ./ nC)
        end
        Lt, Q = _pos_lq(reshape(AT2 * C0, D, 2D))
        nt = _checked_norm(Lt)
        CT = Lt ./ nt
        RT = reshape(Q, D, 2, D)
        Lc = zeros(ComplexF64, D, D)
        RC = zeros(ComplexF64, D, 2, D)
        for a in blocks
            f, q = _pos_lq(CT[ids[a], :])
            Lc[ids[a], ids[a]] = f
            RC[ids[a], a, :] = q
        end
        nc = _checked_norm(Lc)
        Cn = Lc ./ nc
        logc = log(nt) + log(nc)
        δprev, δ = δ, norm(Cn - C0)
        done, best, since = _orth_progress(δ, best, since, orth_tol)
        C0 = Cn
    end
    # left: through the center (dense QR), then the control (block QR)
    DT = nothing
    Qt = nothing
    Qc = Vector{Matrix{ComplexF64}}(undef, 2)
    done, best, since = false, Inf, 0
    δ, δprev = Inf, Inf
    iter = 0
    while !done
        (iter += 1) > maxiter && return nothing
        if iter > _ACCEL_AFTER && δ > _ACCEL_MIN && δ > _ACCEL_RATIO * δprev
            Qts = (Qt[1:D, :], Qt[D+1:2D, :])
            map_left(X) = begin
                Y = Qts[1]' * X * ATs[1] .+ Qts[2]' * X * ATs[2]
                Z = zeros(eltype(Y), D, D)
                for a in blocks
                    Z[ids[a], ids[a]] = Qc[a]' * Y[:, ids[a]]
                end
                Z
            end
            X = _accelerate(map_left, D0, δ)
            Dn = zeros(eltype(D0), D, D)
            for a in blocks
                Dn[ids[a], ids[a]] = last(_pos_qr(X[ids[a], ids[a]]))
            end
            nD = norm(Dn)
            (isfinite(nD) && nD > 0) && (D0 = Dn ./ nD)
        end
        Qt, Rt = _pos_qr(reshape(D0 * ATr, 2D, D))
        DT = Rt ./ _checked_norm(Rt)
        Rc = zeros(ComplexF64, D, D)
        for a in blocks
            Qc[a], Rc[ids[a], ids[a]] = _pos_qr(DT[:, ids[a]])
        end
        Dn = Rc ./ _checked_norm(Rc)
        δprev, δ = δ, norm(Dn - D0)
        done, best, since = _orth_progress(δ, best, since, orth_tol)
        D0 = Dn
    end
    FC = svd(D0 * CT)        # bond control → center
    FT = svd(DT * C0)        # bond center → control
    Bc = reshape(reshape(FT.Vt * reshape(RC, D, 2D), 2D, D) * FC.V, D, 2, D)
    Bt = reshape(reshape(FC.Vt * reshape(RT, D, 2D), 2D, D) * FT.V, D, 2, D)
    tensors = Vector{Array{ComplexF64,3}}(undef, 2)
    spectra = Vector{Vector{Float64}}(undef, 2)
    tensors[S.control], tensors[S.center] = Bc, Bt
    spectra[S.control], spectra[S.center] = FC.S ./ norm(FC.S), FT.S ./ norm(FT.S)
    return tensors, spectra, logc
end

# Structured W·ψ followed by the shared truncation; `nothing` means "use the generic path".
# ψ is not modified.
function _apply_controlled(ψ, S; maxdim, cutoff, truncerr, renormalize, recanonicalize, orth_tol)
    result = try
        ct = _controlled_tensors(ψ, S)
        isnothing(ct) ? nothing : _controlled_canonical(ψ, S, ct; orth_tol)
    catch err
        # Any numerical failure, including a zero norm, is left to the generic path, which
        # then decides (e.g. throws MPOAnnihilationError under its usual conditions).
        err isa Union{MPOAnnihilationError,LAPACKException,ArgumentError,DimensionMismatch} ||
            rethrow()
        nothing
    end
    isnothing(result) && return nothing
    tensors, spectra, lognorm = result
    trial = iMPS(tensors, spectra, 2)
    stats, sliced = _truncate_canonical!(trial; maxdim, cutoff, truncerr, renormalize,
                                         recanonicalize, orth_tol)
    return trial, stats, lognorm, sliced
end
