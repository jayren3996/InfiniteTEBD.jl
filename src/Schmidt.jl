#---------------------------------------------------------------------------------------------------
# Schmidt Canonical Form
#---------------------------------------------------------------------------------------------------
export restore_canonical!

function canonical_gauging(
    Γ::AbstractArray{<:Number, 3},
    L::AbstractMatrix,
    R::AbstractMatrix
)
    @tensor Γ2[:] := R[-1,1] * Γ[1,-2,2] * L[2,-3]
    Γ2
end

function _sanitize_schmidt_values(S::AbstractVector)
    !isempty(S) || throw(ArgumentError("incoming Schmidt spectrum must be nonempty"))
    _finite_entries(S, "incoming Schmidt spectrum")
    return Float64.(abs.(S))
end

function _tolerance(scale::Real; zerotol::Real=ZEROTOL, rtol::Real=sqrt(eps(Float64)))
    return max(Float64(zerotol), Float64(rtol) * max(Float64(scale), 1.0))
end

function _tolerance(vals::AbstractVector{<:Number}; zerotol::Real=ZEROTOL, rtol::Real=sqrt(eps(Float64)))
    scale = isempty(vals) ? 0.0 : maximum(abs.(vals))
    return _tolerance(scale; zerotol, rtol)
end

function _positive_eigensystem(H::AbstractMatrix; zerotol::Real=ZEROTOL, rtol::Real=sqrt(eps(Float64)))
    vals, vecs = eigen(Hermitian(H))
    realvals = Float64.(real.(vals))
    _finite_entries(realvals, "fixed-point eigenvalues")

    tol = _tolerance(realvals; zerotol, rtol)
    clean = similar(realvals)
    for i in eachindex(realvals)
        clean[i] = max(realvals[i], 0.0)
    end

    support = findall(>(tol), clean)
    if isempty(support)
        idx = argmax(abs.(realvals))
        clean[idx] = max(abs(realvals[idx]), eps(Float64))
        support = [idx]
    end
    return clean[support], vecs[:, support]
end

function _transfer_degeneracy(Γ::AbstractArray{<:Number,3})
    Dl, _, Dr = size(Γ)
    Dl == Dr || return (degenerate=false, count=0, leading=0.0, tol=0.0)

    # The dense eigvals below acts on a (Dl·Dr)×(Dl·Dr) nonsymmetric matrix, so
    # its cost is (Dl·Dr)³ — a hard cliff. The previous threshold of 2500 meant
    # the default MAXDIM=50 state (Dl·Dr = 2500 exactly) paid ~6 s and ~300 MiB
    # per schmidt_canonical call for a diagnostic. Keep the exact dense check
    # only for genuinely small bonds (χ ≤ 20) and use the cheap structural
    # heuristic above that size, matching what already happened for χ ≥ 51.
    if Dl * Dr > 400
        sector = _simple_sector_selection(Γ)
        if !isnothing(sector)
            return (degenerate=true, count=sector.sectors, leading=NaN, tol=0.0)
        end
        # Degeneracy detection is diagnostic. Avoid a Krylov preflight here:
        # on moderately large random tensors it can dominate or stall
        # canonicalization before the actual fixed-point solve starts.
        return (degenerate=false, count=0, leading=0.0, tol=0.0)
    end

    return _transfer_degeneracy(eigvals(kraus_mat(Γ, conj(Γ); dir=:r)))
end

function _transfer_degeneracy(vals::AbstractVector)
    isempty(vals) && return (degenerate=false, count=0, leading=0.0, tol=0.0)
    mags = sort!(Float64.(abs.(vals)); rev=true)
    leading = mags[1]
    tol = _tolerance(leading; zerotol=100*ZEROTOL, rtol=100*sqrt(eps(Float64)))
    leading_count = count(m -> abs(m - leading) <= tol, mags)
    return (degenerate=leading_count > 1 && leading > tol, count=leading_count, leading=leading, tol=tol)
end

function _simple_sector_selection(Γ::AbstractArray{<:Number,3})
    Dl, _, Dr = size(Γ)
    Dl == Dr || return nothing

    weights = zeros(Float64, Dl)
    offdiag = 0.0
    for l in 1:Dl, r in 1:Dr
        w = sum(abs2, @view Γ[l, :, r])
        if l == r
            weights[l] += w
        else
            offdiag += w
        end
    end

    total = sum(weights) + offdiag
    tol = _tolerance(total; zerotol=ZEROTOL, rtol=100*eps(Float64))
    active = findall(>(tol), weights)
    (length(active) > 1 && offdiag <= tol) || return nothing

    choice = active[argmax(weights[active])]
    sector = Array(Γ[choice:choice, :, choice:choice])
    nrm = sqrt(sum(abs2, sector))
    if isfinite(nrm) && nrm > 0
        sector ./= nrm
    end
    return (Γ=sector, S=ones(Float64, 1), index=choice, sectors=length(active))
end

function _handle_noninjective!(degeneracy, noninjective::Symbol)
    if degeneracy.degenerate && noninjective == :error
        throw(ArgumentError(
            "likely non-injective/degenerate transfer spectrum detected " *
            "($(degeneracy.count) leading eigenvalues within $(degeneracy.tol))"
        ))
    end
    return nothing
end
#---------------------------------------------------------------------------------------------------
"""
schmidt_canonical(Γ; kerwords)

Schmidt Canonical Form
1. Return a Schmidt canonical form.
2. This algorithm assume there is no degeneracy.

Parameters:
- `Γ`
  Local three-leg tensor or grouped local tensor representing one periodic block.
- `S`
  Schmidt values on the incoming bond of that block.

Keyword arguments:
- `maxdim=MAXDIM`
  Maximum number of Schmidt values to retain in the canonicalized result.
- `cutoff=SVDTOL`
  Singular values smaller than this threshold are discarded.
- `renormalize=true`
  Whether to renormalize the retained Schmidt values. With
  `renormalize=false` the returned tensor keeps the state's original norm,
  while the returned Schmidt spectrum is rescaled to the canonical unit-norm
  convention (`Σλ² = 1` up to genuinely discarded truncation weight) — the
  raw singular values would otherwise carry an arbitrary gauge factor from
  the unit-Frobenius normalization of the transfer fixed points.
- `zerotol=ZEROTOL`
  Threshold used when selecting positive eigenvalues of the transfer-matrix
  fixed points.

Returns:
- `(Γ_new, S_new)` where `Γ_new` is the gauged canonical tensor and `S_new` is
  the resulting Schmidt spectrum.

Notes:
- This routine is the low-level canonicalization kernel used by
  [`canonical!`](@ref).
- It assumes the state is in the non-degenerate injective setting.
- The returned tensor has the right Schmidt values absorbed on its right bond,
  matching the package storage convention.
- The pseudoinverse construction implicitly performs low-rank compression when
  near-zero eigenvalues of the transfer-matrix fixed points are filtered by
  `_positive_eigensystem`. A warning is emitted if the rank of either fixed
  point is significantly smaller than the bond dimension.
"""
function schmidt_canonical(
    Γ::AbstractArray{<:Number,3}, S::AbstractVector;
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true,
    zerotol=ZEROTOL,
    noninjective::Symbol=:warn,
    symmetry_break::Symbol=:none,
    tol::Union{Nothing,Real}=nothing,
    maxiter::Union{Nothing,Integer}=nothing,
)
    _validate_canonical_options(maxdim, cutoff, noninjective, symmetry_break)
    S_in = _sanitize_schmidt_values(S)
    length(S_in) == size(Γ, 3) ||
        throw(ArgumentError("incoming Schmidt spectrum length must match the right bond dimension"))
    length(S_in) == size(Γ, 1) ||
        throw(ArgumentError("incoming Schmidt spectrum length must match the left bond dimension"))

    # With noninjective=:ignore and symmetry_break=:none the degeneracy result
    # is provably unused (no error, no warning, no sector selection), so skip
    # the diagnostic entirely. This is the configuration canonical! uses for
    # its second gauge-restoration pass.
    degeneracy = if noninjective === :ignore && symmetry_break === :none
        (degenerate=false, count=0, leading=0.0, tol=0.0)
    else
        _transfer_degeneracy(Γ)
    end
    _handle_noninjective!(degeneracy, noninjective)
    if degeneracy.degenerate && symmetry_break == :auto
        sector = _simple_sector_selection(Γ)
        if !isnothing(sector)
            if noninjective == :warn
                @warn "Detected likely non-injective/degenerate transfer spectrum; symmetry_break=:auto selected virtual sector $(sector.index) of $(sector.sectors). Proceeding with a symmetry-broken sector; full non-injective block canonical decomposition is not implemented."
            end
            return schmidt_canonical(
                sector.Γ,
                sector.S;
                maxdim,
                cutoff,
                renormalize,
                zerotol,
                noninjective=:ignore,
                symmetry_break=:none,
                tol,
                maxiter,
            )
        end
    end
    if degeneracy.degenerate && noninjective == :warn
        @warn "Detected likely non-injective/degenerate transfer spectrum; proceeding without symmetry-sector selection. Full non-injective block canonical decomposition is not implemented."
    end

    # The left transfer acts on the seam-weighted representation.
    R = steady_mat(Γ; dir=:r, tol, maxiter)
    Γc = copy(Γ)
    tensor_rmul!(Γc, _safe_reciprocal(S_in; atol=ZEROTOL, rtol=0.0))
    Γl = copy(Γc)
    tensor_lmul!(S_in, Γl)
    L = steady_mat(Γl; dir=:l, tol, maxiter)

    left, right, weights, λ = _canonical_seam(R, L, S_in; maxdim, cutoff, renormalize, zerotol)
    Γ_new = canonical_gauging(Γc, right, left)
    tensor_rmul!(Γ_new, weights)
    return Γ_new, λ
end

function _canonical_seam(R, L, S_in; maxdim, cutoff, renormalize, zerotol=ZEROTOL)
    er, vr = _positive_eigensystem(R; zerotol)
    el, vl = _positive_eigensystem(L; zerotol)
    if length(er) < size(vr, 1)
        @warn "Low-rank compression in schmidt_canonical: right fixed-point rank $(length(er)) < bond dimension $(size(vr, 1))."
    end
    if length(el) < size(vl, 1)
        @warn "Low-rank compression in schmidt_canonical: left fixed-point rank $(length(el)) < bond dimension $(size(vl, 1))."
    end

    # Avoid Diagonal allocations by using broadcasting
    sqrt_er = sqrt.(er)
    sqrt_el = sqrt.(el)
    inv_sqrt_er = _safe_reciprocal(sqrt_er; atol=ZEROTOL, rtol=0.0)
    inv_sqrt_el = _safe_reciprocal(sqrt_el; atol=ZEROTOL, rtol=0.0)

    X = (vr .* sqrt_er') * vr'
    Yt = (vl .* sqrt_el') * vl'
    X_inv = (vr .* inv_sqrt_er') * vr'
    Yt_inv = (vl .* inv_sqrt_el') * vl'

    M = Yt * (S_in .* X)
    U, S_new, V = svd_trim(M; maxdim, svd_min=cutoff, renormalize)
    R_mat = Yt_inv * U
    L_mat = V * X_inv
    # The raw center has an arbitrary scale from the two Frobenius-normalized
    # fixed points. Keep that scale in the tensor construction, and remove
    # it from the separately stored Schmidt spectrum.
    c = renormalize ? one(eltype(S_new)) : norm(M)
    λ = !renormalize && isfinite(c) && c > 0 ? S_new ./ c : S_new
    return L_mat, R_mat, S_new, λ
end
#---------------------------------------------------------------------------------------------------
# Multiple tensors
"""
    schmidt_canonical(Γs, S; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false)

Canonicalize a full unit cell represented as a vector of local tensors.

Parameters:
- `Γs`
  Vector of local three-leg tensors forming one periodic unit cell.
- `S`
  Schmidt spectrum on the incoming bond to the grouped unit cell.

Keyword arguments:
- `maxdim`, `cutoff`, `renormalize`
  Passed through to the single-block [`schmidt_canonical`](@ref) kernel and the
  subsequent decomposition.

Returns:
- `(Γs_new, λs_new)` where `Γs_new` are the stored right-canonical tensors for
  the unit cell and `λs_new` are the Schmidt spectra on all bonds of that unit
  cell.

Notes:
- For `length(Γs) == 1`, this routine returns a one-site unit cell and applies a
  final normalization so the stored tensor is properly right-canonical.
- For longer unit cells, the grouped canonical tensor is decomposed back into
  site-local tensors with [`tensor_decomp!`](@ref).
"""
function schmidt_canonical(
    Γs::AbstractVector{<:AbstractArray{<:Number, 3}}, S::AbstractVector;
    maxdim=MAXDIM,
    cutoff=SVDTOL,
    renormalize=false,
    noninjective::Symbol=:warn,
    symmetry_break::Symbol=:none,
    tol::Union{Nothing,Real}=nothing,
    maxiter::Union{Nothing,Integer}=nothing,
)
    n = length(Γs)
    Γ_grouped = tensor_group(Γs)

    A, λ = schmidt_canonical(
        Γ_grouped,
        S;
        maxdim,
        cutoff,
        renormalize,
        noninjective,
        symmetry_break,
        tol,
        maxiter,
    )
    if isone(n)
        Dl, d, Dr = size(A)
        overlap = zeros(eltype(A), Dl, Dl)
        for s in 1:d
            As = reshape(A[:, s, :], Dl, Dr)
            overlap .+= As * As'
        end
        scale = sqrt(real(tr(overlap)) / Dl)
        if isfinite(scale) && scale > 0
            A ./= scale
        end
        return [A], [λ]
    end
    tensor_lmul!(λ, A)
    Γs, λs = tensor_decomp!(A, λ, n; maxdim, svd_min=cutoff, renormalize)
    Γs, push!(λs, λ)
end

struct _GramGaugeFailure <: Exception
    cause::Exception
end

_gram_failure(cause, checked) = throw(checked ? _GramGaugeFailure(cause) : cause)

function _normalize_cell_environment(ρ; checked=false)
    weight = real(tr(ρ))
    isfinite(weight) && weight > 0 || _gram_failure(
        ArgumentError("invalid unit-cell transfer environment"), checked)
    return Hermitian((ρ + ρ') / (2 * weight)), weight
end

function _cell_environment(tensors, dir; tol, maxiter, checked=false, seed=nothing)
    χ = size(first(tensors), 1)
    isnothing(seed) && (seed = Matrix{eltype(first(tensors))}(I, χ, χ))
    action = _chain_transfer_action(tensors, tensors; dir)
    tol = isnothing(tol) ? max(1e-12, 10eps(real(eltype(seed)))) : tol
    _, vectors, info = eigsolve(action, seed, 1, :LR; ishermitian=false,
                                _krylov_opts(; tol, maxiter)...)
    info.converged >= 1 || _gram_failure(
        ErrorException("unit-cell transfer fixed point did not converge for dir=$dir"), checked)
    ρ = first(vectors)
    phase = tr(ρ)
    isfinite(phase) && !iszero(phase) || _gram_failure(
        ArgumentError("invalid unit-cell transfer fixed point"), checked)
    return first(_normalize_cell_environment(ρ / phase; checked))
end

"""
    restore_canonical!(ψ; method=:gram, tol=nothing, maxiter=nothing, accelerate=true)

Restore the Schmidt-canonical gauge of a dense injective `iMPS`. Normalizes
the state per unit cell, replaces its stored Schmidt spectra, and preserves
its bond dimensions.

`method=:gram` solves sitewise transfer fixed points and requires invertible
right environments. `method=:qr` instead iterates periodic LQ and QR
sweeps on the site tensors, for gauges where forming a Gram matrix would
lose precision. It tests seam closure, then uses SVD only on virtual-bond
centers. For QR, `tol` controls the relative seam residual (default
`max(1e-12, 50eps)`), and `maxiter` bounds the sweeps in each direction
(default 500). Rank loss or failure to close the seam raises without
committing the candidate. Injectivity is assumed; small QR seam spaces
are also checked for multiple peripheral transfer eigenvalues.
QR uses mixed-transfer Arnoldi to propose seam carries when plain sweeps
have not closed. Set `accelerate=false` to use only local sweeps.
`method=:auto` checks a Gram candidate and retries QR from the original
state if Gram fails numerically or the canonical residuals are too large.
It also checks small seam spaces for multiple peripheral eigenvalues.
"""
function restore_canonical!(ψ::DenseIMPS; method::Symbol=:gram, tol::Union{Nothing,Real}=nothing,
                            maxiter::Union{Nothing,Integer}=nothing, accelerate::Bool=true)
    if method === :gram
        _restore_canonical!(ψ; tol, maxiter)
    elseif method === :qr
        _restore_canonical_qr!(ψ; tol, maxiter, accelerate)
    elseif method === :auto
        _restore_canonical_auto!(ψ; tol, maxiter, accelerate)
    else
        throw(ArgumentError("method must be :gram, :qr or :auto"))
    end
    return ψ
end

function _restore_canonical_auto!(ψ; tol, maxiter, accelerate)
    threshold = 10 * _qr_target(ψ, tol, maxiter)[1]
    trial = iMPS(copy(ψ.Γ), copy(ψ.λ), ψ.n)
    gram_error = try
        _restore_canonical!(trial; tol, maxiter, checked=true)
        _check_restored_gauge(trial, threshold) ? nothing :
            ErrorException("candidate failed local isometry or Schmidt-environment checks")
    catch err
        err isa _GramGaugeFailure || rethrow()
        err.cause
    end
    if isnothing(gram_error)
        seam = argmin(length.(trial.λ))
        _check_small_transfer_uniqueness(trial.Γ[[seam+1:trial.n; 1:seam]])
        ψ.Γ .= trial.Γ
        ψ.λ .= trial.λ
        return ψ
    end
    try
        _restore_canonical_qr!(ψ; tol, maxiter, accelerate)
    catch err
        err isa Union{ArgumentError,ErrorException,LAPACKException} || rethrow()
        throw(ArgumentError("canonical gauge restoration failed. Gram: " *
            sprint(showerror, gram_error) * ". QR/LQ: " * sprint(showerror, err)))
    end
    return ψ
end

function _check_restored_gauge(ψ, tol)
    for i in 1:ψ.n
        A = ψ.Γ[i]
        l, _, r = size(A)
        eye = Matrix{eltype(A)}(I, r, r)
        rc = apply_transfer(A, A, eye)
        norm(rc - I) <= tol * sqrt(l) || return false
        λ = ψ.λ[i]
        all(s -> isfinite(s) && s > 0, λ) || return false
        abs(norm(λ) - 1) <= tol || return false
        ρ = Matrix(Diagonal(ψ.λ[mod1(i-1, ψ.n)] .^ 2))
        norm(conj(apply_transfer(A, A, ρ; dir=:l)) - Diagonal(λ .^ 2)) <= tol || return false
    end
    return true
end

function _restore_canonical!(ψ::DenseIMPS; tol, maxiter, reduce_right_support::Bool=false,
                            checked::Bool=false)
    n = ψ.n
    # Solve the eigenproblems on the smallest bond. All subsequent local
    # contractions follow the same cyclic order.
    seam_index = argmin(length.(ψ.λ))
    order = [seam_index+1:n; 1:seam_index]
    tensors = ψ.Γ[order]
    seam_right = _cell_environment(tensors, :r; tol, maxiter, checked)
    right = Vector{typeof(seam_right)}(undef, n)
    right[n] = seam_right
    scales = Vector{typeof(real(tr(seam_right)))}(undef, n)
    for i in n:-1:1
        ρ, scales[i] = _normalize_cell_environment(
            apply_transfer(tensors[i], tensors[i], right[i]); checked)
        i > 1 && (right[i-1] = ρ)
    end

    # The :l contraction acts by transpose in gtrm's vectorization.
    # Its conjugate gives the adjoint environment used in the Schmidt gauge.
    λ = ψ.λ[seam_index]
    left_seed = nothing
    if all(>(0), λ)
        weights = convert(Vector{real(eltype(ψ))}, abs2.(λ ./ maximum(λ)))
        all(>(0), weights) && (left_seed = Matrix{eltype(ψ)}(Diagonal(weights)))
    end
    seam_left = _cell_environment(tensors, :l; tol, maxiter, checked, seed=left_seed)
    left = seam_left
    gauges = Vector{Matrix{eltype(seam_right)}}(undef, n)
    inverses = similar(gauges)
    spectra = similar(ψ.λ)
    for i in 1:n
        left = i == n ? seam_left : first(_normalize_cell_environment(
            apply_transfer(tensors[i], tensors[i], left; dir=:l); checked))
        r = eigen(right[i])
        threshold = length(r.values) * eps(last(r.values))
        first_supported = findfirst(>(threshold), r.values)
        isnothing(first_supported) && _gram_failure(
            ArgumentError("empty right transfer support on bond $i"), checked)
        first_supported == 1 || reduce_right_support || _gram_failure(ArgumentError(
            "right transfer environment on bond $i is numerically singular"), checked)
        support = first_supported:lastindex(r.values)
        roots = sqrt.(r.values[support])
        vectors = r.vectors[:, support]
        X = vectors .* transpose(roots)
        s = eigen(Hermitian(X' * conj(left) * X))
        descending = lastindex(s.values):-1:firstindex(s.values)
        weights = max.(s.values[descending], 0)
        weight = sum(weights)
        isfinite(weight) && weight > 0 || _gram_failure(
            ArgumentError("invalid Schmidt spectrum on bond $i"), checked)
        spectra[i] = sqrt.(weights ./ weight)
        U = s.vectors[:, descending]
        gauges[i] = X * U
        inverses[i] = U' * (vectors' ./ roots)
    end

    gauged = similar(tensors)
    for i in 1:n
        gauged[i] = canonical_gauging(tensors[i], gauges[i], inverses[mod1(i-1, n)])
        gauged[i] ./= sqrt(scales[i])
    end
    ψ.Γ[order] = gauged
    ψ.λ[order] = spectra
    return sum(log, scales) / 2
end

function _qr_phases(carry)
    pivots = diag(carry)
    scale = norm(carry)
    threshold = eps(typeof(scale)) * size(carry, 1) * scale
    isfinite(scale) && all(p -> abs(p) > threshold, pivots) ||
        throw(ArgumentError("periodic QR/LQ found a numerically rank-deficient carry"))
    return pivots ./ abs.(pivots), scale
end

function _right_lq_sweep(tensors, seed, B=nothing)
    carry = seed
    for i in reverse(eachindex(tensors))
        A = tensors[i]
        l, d, r = size(A)
        l <= d*r || throw(ArgumentError("bond before site $i cannot be right-isometric at fixed dimension"))
        block = reshape(reshape(A, l*d, r) * carry, l, d*r)
        F = lq!(block)
        carry = Matrix(F.L)
        phases, scale = _qr_phases(carry)
        if !isnothing(B)
            # L D† · D Q keeps the local factorization intact.
            Q = Matrix(F.Q)
            Q .*= phases
            B[i] = reshape(Q, l, d, r)
        end
        carry .*= transpose(conj.(phases)) ./ scale
    end
    return carry
end

function _mixed_qr_seed(action, carry, tol, direction)
    # The current carry can already satisfy a loose *linear* Ritz
    # tolerance while missing the tighter relative triangular closure.
    krylov_tol = max(tol / 100, eps(real(eltype(carry))))
    _, vectors, info = eigsolve(action, carry, 1, :LM;
                                tol=krylov_tol, maxiter=4, ishermitian=false)
    info.converged >= 1 || return carry
    candidate = first(vectors)
    phase = dot(carry, candidate)
    iszero(phase) && return carry
    candidate = candidate .* (conj(phase) / abs(phase))
    if eltype(carry) <: Real
        norm(imag(candidate)) <= tol * norm(candidate) || return carry
        candidate = real(candidate)
    end
    proposed = direction === :right ? Matrix(lq(candidate).L) : Matrix(qr(candidate).R)
    phases, scale = try
        _qr_phases(proposed)
    catch err
        err isa ArgumentError || rethrow()
        return carry
    end
    proposed .*= direction === :right ? transpose(conj.(phases)) ./ scale : conj.(phases) ./ scale
    return proposed
end

function _periodic_right_orth(tensors, seed, tol, maxiter, accelerate)
    carry = seed
    for sweep in 1:maxiter
        previous = carry
        carry = _right_lq_sweep(tensors, previous)
        # A normwise difference can hide a large relative seam error in
        # one of the small virtual directions. Materialize isometries only
        # for the sweep whose carry actually closes.
        residual = norm(LowerTriangular(previous) \ (carry - previous))
        if residual <= tol
            B = similar(tensors)
            _right_lq_sweep(tensors, previous, B)
            return B
        elseif accelerate && sweep < maxiter && sweep % 8 == 0
            B = similar(tensors)
            _right_lq_sweep(tensors, previous, B)
            action = _chain_transfer_action(B, tensors; dir=:r)
            proposed = _mixed_qr_seed(action, carry, tol, :right)
            # Arnoldi is a proposal, not a replacement for the closure test.
            check = _right_lq_sweep(tensors, proposed)
            norm(LowerTriangular(proposed) \ (check - proposed)) < residual && (carry = proposed)
        end
    end
    error("periodic LQ did not close the right seam after $maxiter sweeps")
end

function _left_qr_sweep(B, seed, centers, AL=nothing)
    carry = seed
    for i in eachindex(B)
        l, d, r = size(B[i])
        r <= l*d || throw(ArgumentError("bond after site $i cannot be left-isometric at fixed dimension"))
        block = reshape(carry * reshape(B[i], l, d*r), l*d, r)
        F = qr!(block)
        carry = Matrix(F.R)
        phases, scale = _qr_phases(carry)
        carry .*= conj.(phases) ./ scale
        centers[i] = carry
        if !isnothing(AL)
            Q = F.Q[:, 1:r]
            Q .*= transpose(phases)
            AL[i] = reshape(Q, l, d, r)
        end
    end
    return carry
end

function _periodic_left_centers(B, seed, tol, maxiter, accelerate)
    centers = Vector{typeof(seed)}(undef, length(B))
    carry = seed
    for sweep in 1:maxiter
        previous = carry
        carry = _left_qr_sweep(B, previous, centers)
        residual = norm((carry - previous) / UpperTriangular(previous))
        if residual <= tol
            return centers
        elseif accelerate && sweep < maxiter && sweep % 8 == 0
            AL = similar(B)
            _left_qr_sweep(B, previous, centers, AL)
            # The library's :l map is a transpose. This converts it to
            # Σ AL† C B for the left center equation.
            transfer = _chain_transfer_action(AL, B; dir=:l)
            action = C -> Matrix(transpose(transfer(transpose(C))))
            proposed = _mixed_qr_seed(action, carry, tol, :left)
            check = _left_qr_sweep(B, proposed, centers)
            norm((check - proposed) / UpperTriangular(proposed)) < residual && (carry = proposed)
        end
    end
    error("periodic QR did not close the left seam after $maxiter sweeps")
end

function _check_small_transfer_uniqueness(tensors)
    χ = size(first(tensors), 1)
    1 < χ <= 4 || return nothing
    # A periodic sweep can close even when the transfer has more than one
    # peripheral eigenvalue. Check the small seam without fusing the cell.
    E = _seam_transfer_matrix(tensors)
    gap_floor = 100 * eps(real(eltype(E))) * χ^2
    count(v -> abs(abs(v) - 1) <= gap_floor, eigvals(E)) <= 1 ||
        throw(ArgumentError("gauge restoration found multiple peripheral transfer eigenvalues"))
    return nothing
end

function _qr_target(ψ, tol, maxiter)
    ϵ = eps(real(eltype(ψ)))
    tol = isnothing(tol) ? max(1e-12, 50ϵ) : tol
    maxiter = isnothing(maxiter) ? 500 : maxiter
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    return tol, maxiter
end

function _restore_canonical_qr!(ψ; tol, maxiter, accelerate)
    ϵ = eps(real(eltype(ψ)))
    tol, maxiter = _qr_target(ψ, tol, maxiter)
    seam = argmin(length.(ψ.λ))
    order = [seam+1:ψ.n; 1:seam]
    tensors = ψ.Γ[order]
    χ = length(ψ.λ[seam])
    seed = Matrix{eltype(ψ)}(I, χ, χ)
    seed ./= norm(seed)
    B = _periodic_right_orth(tensors, seed, tol, maxiter, accelerate)
    _check_small_transfer_uniqueness(B)
    centers = _periodic_left_centers(B, seed, tol, maxiter, accelerate)

    rotations = similar(centers)
    spectra = similar(ψ.λ)
    for i in eachindex(centers)
        F = svd!(centers[i])
        last(F.S) > ϵ * length(F.S) * first(F.S) ||
            throw(ArgumentError("periodic QR/LQ found a numerically rank-deficient center on bond $(order[i])"))
        rotations[i] = F.Vt
        spectra[i] = F.S ./ norm(F.S)
    end
    gauged = similar(B)
    for i in eachindex(gauged)
        previous = mod1(i - 1, ψ.n)
        gauged[i] = canonical_gauging(B[i], rotations[i]', rotations[previous])
        ρ = Matrix(Diagonal(spectra[previous] .^ 2))
        density = conj(apply_transfer(gauged[i], gauged[i], ρ; dir=:l))
        norm(density - Diagonal(spectra[i] .^ 2)) <= max(10tol, 100ϵ) ||
            error("periodic QR/LQ has an inconsistent Schmidt environment at site $(order[i])")
    end
    ψ.Γ[order] = gauged
    ψ.λ[order] = spectra
    return ψ
end

# Reuse the smallest virtual bond for the fixed-point solve. Then
# transport the answer to the original seam, where truncation must start.
function _streaming_seam_environment(tensors, dir; tol, maxiter, dense_map=nothing)
    χ = size(first(tensors), 1)
    seed = Matrix{eltype(first(tensors))}(I, χ, χ)
    if isnothing(dense_map)
        ρ = _cell_environment(tensors, dir; tol, maxiter, seed)
        return dir === :r ? ρ : conj.(ρ)
    end
    E = eigen(dense_map)
    ρ = reshape(E.vectors[:,argmax(real.(E.values))], size(seed))
    phase = tr(ρ)
    isfinite(phase) && !iszero(phase) || throw(ArgumentError("canonical!: invalid transfer fixed point for dir=$dir"))
    ρ = ρ / phase
    eltype(seed) <: Real && (ρ = real.(ρ))
    return dir === :r ? ρ : conj.(ρ)
end

function _seam_transfer_matrix(tensors)
    all(A -> size(A, 3) <= 4, tensors) && return gtrm(tensors, tensors)
    T = eltype(first(tensors))
    χ = size(first(tensors), 1)
    seed = zeros(T, χ, χ)
    E = Matrix{T}(undef, χ^2, χ^2)
    action = _chain_transfer_action(tensors, tensors; dir=:r)
    for i in eachindex(seed)
        seed[i] = 1
        E[:,i] = vec(action(seed))
        seed[i] = 0
    end
    return E
end

function _streaming_seam_sector(tensors)
    T = eltype(first(tensors))
    χ = size(first(tensors), 1)
    seed = zeros(T, χ, χ)
    weights = zeros(real(T), χ)
    offdiag = zero(real(T))
    action = _chain_transfer_action(tensors, tensors; dir=:r)
    for r in 1:χ
        seed[r,r] = 1
        density = action(seed)
        weights[r] = real(density[r,r])
        offdiag += max(0, real(tr(density)) - weights[r])
        seed[r,r] = 0
    end
    tol = _tolerance(sum(weights) + offdiag; zerotol=ZEROTOL, rtol=100*eps(Float64))
    count(>(tol), weights) > 1 && offdiag <= tol || return nothing
    choice = argmax(weights)
    return (; choice, scale=sqrt(weights[choice]), sectors=count(>(tol), weights))
end

function _streaming_split(tensors, left, right, weights, λ; maxdim, cutoff, renormalize)
    n = length(tensors)
    T = promote_type(eltype(first(tensors)), eltype(left), eltype(right))
    out = Vector{Array{T,3}}(undef, n)
    spectra = Vector{typeof(λ)}(undef, n)
    spectra[n] = λ

    # The grouped tensor after the seam transform is A₁…Aₙ. Untruncated
    # right LQ factors replace each physical suffix with a row isometry.
    # A cut M*Q then has the same nonzero singular values as M.
    carry = right .* transpose(weights)
    for i in n:-1:2
        a, d, b = size(tensors[i])
        block = reshape(reshape(tensors[i], a*d, b) * carry, a, :)
        F = lq!(block)
        carry = Matrix(F.L)
        out[i] = reshape(Matrix(F.Q), size(carry, 2), d, :)
    end
    a, d, b = size(first(tensors))
    block = reshape(reshape(first(tensors), a*d, b) * carry, a, :)
    block = (λ .* left) * block
    previous = λ
    for i in 1:n-1
        a, d = length(previous), size(tensors[i], 2)
        U, S, Vt = svd_trim(reshape(block, a*d, :); maxdim, svd_min=cutoff, renormalize)
        A = reshape(U, a, d, :)
        tensor_lmul!(_safe_reciprocal(previous; rtol=eps(Float64)), A)
        tensor_rmul!(A, S)
        out[i] = A
        spectra[i] = S

        Q = out[i+1]
        dnext, b = size(Q,2), size(Q,3)
        block = Vt * reshape(Q, size(Q,1), :)
        if i < n - 1
            block .*= S
        else
            out[n] = reshape(block, length(S), dnext, b)
        end
        previous = S
    end
    return out, spectra
end

function _schmidt_canonical_streaming(tensors, S;
        maxdim, cutoff, renormalize, noninjective, symmetry_break, tol, maxiter)
    n = length(tensors)
    if n == 1
        return schmidt_canonical(tensors, S; maxdim, cutoff, renormalize, noninjective,
                                 symmetry_break, tol, maxiter)
    end
    S_in = _sanitize_schmidt_values(S)
    seam_idx = argmin([size(A, 3) for A in tensors])
    order = [seam_idx+1:n; 1:seam_idx]
    χ = size(tensors[seam_idx], 3)
    ordered = tensors[order]
    check = noninjective !== :ignore || symmetry_break !== :none
    map = χ <= 8 ? _seam_transfer_matrix(ordered) : nothing
    degeneracy = check && !isnothing(map) ? _transfer_degeneracy(eigvals(map)) : nothing
    # At large χ, inspect the cheap common diagonal case. An explicit
    # selection request also checks the seam of more general cell layouts.
    sector = nothing
    if check && isnothing(map) &&
            (symmetry_break === :auto || all(A -> !isnothing(_simple_sector_selection(A)), tensors))
        sector = _streaming_seam_sector(tensors)
        isnothing(sector) || (degeneracy = _transfer_degeneracy(sector.weights))
    end
    if !isnothing(degeneracy) && degeneracy.degenerate
        _handle_noninjective!(degeneracy, noninjective)
        if symmetry_break === :auto
            isnothing(sector) && (sector = _streaming_seam_sector(tensors))
            if !isnothing(sector)
                nsel = sector.choice:sector.choice
                restricted = copy(tensors)
                restricted[1] = tensors[1][nsel, :, :] / sector.scale
                restricted[end] = tensors[end][:, :, nsel]
                if noninjective === :warn
                    @warn "Detected likely non-injective/degenerate transfer spectrum; symmetry_break=:auto selected virtual sector $(sector.choice) of $(sector.sectors)."
                end
                return _schmidt_canonical_streaming(restricted, [1.0];
                    maxdim, cutoff, renormalize, noninjective=:ignore, symmetry_break=:none, tol, maxiter)
            end
        end
        if noninjective === :warn
            @warn "Detected likely non-injective/degenerate transfer spectrum; proceeding without symmetry-sector selection. Full non-injective block canonical decomposition is not implemented."
        end
    end

    R = _streaming_seam_environment(ordered, :r; tol, maxiter, dense_map=map)
    if seam_idx != n
        head = view(tensors, 1:seam_idx)
        R = apply_chain_transfer(head, head, R)
    end
    scale = norm(R)
    isfinite(scale) && scale > 0 || throw(ArgumentError("canonical!: invalid right transfer environment"))
    R = Hermitian((R + R') / (2*scale))

    Sinv = _safe_reciprocal(S_in; atol=ZEROTOL, rtol=0.0)
    if all(isone, S_in)
        weighted, ordered_weighted = tensors, ordered
        left_map = isnothing(map) ? nothing : transpose(map)
    else
        weighted = copy(tensors)
        weighted[1] = copy(first(tensors))
        weighted[n] = copy(last(tensors))
        tensor_lmul!(S_in, weighted[1])
        tensor_rmul!(weighted[n], Sinv)
        ordered_weighted = weighted[order]
        left_map = isnothing(map) ? nothing : transpose(_seam_transfer_matrix(ordered_weighted))
    end
    L = _streaming_seam_environment(ordered_weighted, :l; tol, maxiter, dense_map=left_map)
    if seam_idx != n
        tail = view(weighted, seam_idx+1:n)
        L = conj.(apply_chain_transfer(tail, tail, conj.(L); dir=:l))
    end
    scale = norm(L)
    isfinite(scale) && scale > 0 || throw(ArgumentError("canonical!: invalid left transfer environment"))
    L = Hermitian((L + L') / (2*scale))

    left, right, weights, λ = _canonical_seam(R, L, S_in; maxdim, cutoff, renormalize)
    return _streaming_split(tensors, left, Sinv .* right, weights, λ;
                            maxdim, cutoff, renormalize)
end
#---------------------------------------------------------------------------------------------------
#=
export canonical_trim
function canonical_trim(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}};
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true,  
)
    n = length(Ts)
    T = tensor_group(Ts)
    T_RC = right_canonical(T)
    T_BRC = block_trim(T_RC)
    A, λ = schmidt_canonical(T_BRC; renormalize)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, n; maxdim, svd_min=cutoff, renormalize)
end
=#
