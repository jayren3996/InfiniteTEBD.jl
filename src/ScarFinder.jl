#---------------------------------------------------------------------------------------------------
# ScarFinder
#---------------------------------------------------------------------------------------------------
export operator_span, energy_density, energy_span
export scarfinder_step!, scarfinder!, minimize_on_trajectory!
export uniform_layers, truncate_unitcell!, fix_energy!

"""
    operator_span(ψ, O)

Infer how many sites the local operator `O` acts on from the local dimension of `ψ`.

This assumes a uniform local Hilbert-space dimension across the unit cell. An
`ArgumentError` is thrown if `size(O, 1)` is not an exact power of the local dimension.

Parameters:
- `ψ`
  Reference state used only to infer the local Hilbert-space dimension.
- `O`
  Square dense local operator.

Returns:
- Number of sites on which `O` acts.

Example:
```julia
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
span = operator_span(psi, kron([1 0; 0 -1], [1 0; 0 -1]))
```
"""
function operator_span(ψ::iMPS, O::AbstractMatrix)
    size(O, 1) == size(O, 2) || throw(ArgumentError("operator must be square"))
    d = size(ψ.Γ[1], 2)
    dim = size(O, 1)
    dim > 1 || throw(ArgumentError("scalar operators do not define a nonzero local support"))
    d > 1 || throw(ArgumentError("operator dimension $dim is incompatible with local dimension $d"))
    span = 0
    block = 1
    while block < dim
        block *= d
        span += 1
    end
    block == dim || throw(ArgumentError("operator dimension $dim is incompatible with local dimension $d"))
    return span
end

function _validate_local_span(ψ::iMPS, span::Integer; name::AbstractString="span")
    span >= 1 || throw(ArgumentError("$name must be positive"))
    span <= ψ.n || throw(ArgumentError("$name must not exceed the unit-cell length $(ψ.n)"))
    return nothing
end

_can_group_unitcell(ψ::iMPS) = ψ.n <= 6 && _fits_grouped_cell(ψ)

"""
    uniform_layers(ψ, h; span=operator_span(ψ, h))

Build commuting layers for every translation of a local density `h` on the
periodic unit cell. Gates in one layer have disjoint supports. If the cell
length is divisible by `span`, this uses exactly `span` layers. For other
cell lengths it uses the minimum number of layers with disjoint supports.
Supply custom layers to `trotter_gates` for overlapping terms known to commute.
"""
function uniform_layers(ψ::iMPS, h::AbstractMatrix; span::Integer=operator_span(ψ, h))
    _validate_local_span(ψ, span)
    d = size(ψ.Γ[1], 2)
    all(A -> size(A, 2) == d, ψ.Γ) || throw(ArgumentError("uniform_layers requires equal site dimensions"))
    operator_span(ψ, h) == span || throw(ArgumentError("h and span specify different supports"))
    blocks, extra = divrem(ψ.n, span)
    per_block, longer_blocks = divrem(extra, blocks)
    width = span + per_block
    layers = [Tuple{typeof(h), Int, Int}[] for _ in 1:cld(ψ.n, blocks)]
    i = 1
    # Balance the cell into blocks of width >= span. Starts at the same
    # offset in successive blocks are separated by >= span, including across
    # the seam. Each layer can hold at most `blocks` disjoint terms, so the
    # number of layers reaches the lower bound ceil(n / blocks).
    for b in 1:blocks
        for k in 1:(width + (b <= longer_blocks))
            push!(layers[k], (h, i, mod1(i + span - 1, ψ.n)))
            i += 1
        end
    end
    return layers
end

"""
    _evolve_uniform!(ψ, G; span, maxdim=MAXDIM)

Apply the same local gate `G` at every translation of a periodic unit cell.

Parameters:
- `ψ`
  State updated in place.
- `G`
  Local gate to apply uniformly.

Keyword arguments:
- `span`
  Number of sites acted on by `G`.
- `maxdim=MAXDIM`
  Maximum temporary bond dimension used by each local update.

Returns:
- The same object `ψ`, mutated in place.

Notes:
- This is an internal helper used by the ScarFinder routines.
"""
function _evolve_uniform!(ψ::iMPS, G::AbstractMatrix; span::Integer, maxdim::Integer=MAXDIM)
    _validate_local_span(ψ, span)
    offset = span - 1
    # Optimization: for translationally invariant states, apply once and copy.
    # This reduces O(n) gate applications to O(1) when all sites are identical.
    # Note: truncation may break exact translational invariance, so this is
    # guarded by an approximate equality check.
    # isapprox on arrays of unequal size throws DimensionMismatch rather than
    # returning false, and bond dimensions routinely differ across bonds after
    # truncation — guard each comparison with a size check so non-uniform
    # states take the generic path instead of crashing.
    is_uniform = span == 1 &&
                 ψ.n > 1 &&
                 all(size(ψ.λ[1]) == size(ψ.λ[i]) && ψ.λ[1] ≈ ψ.λ[i] for i in 2:ψ.n) &&
                 all(size(ψ.Γ[1]) == size(ψ.Γ[i]) && ψ.Γ[1] ≈ ψ.Γ[i] for i in 2:ψ.n)
    if is_uniform
        applygate!(ψ, G, 1, mod(1 + offset - 1, ψ.n) + 1; maxdim)
        for i in 2:ψ.n
            ψ.Γ[i] = copy(ψ.Γ[1])
            ψ.λ[i] = copy(ψ.λ[1])
        end
    else
        @inbounds for i in 1:ψ.n
            applygate!(ψ, G, i, mod(i + offset - 1, ψ.n) + 1; maxdim)
        end
    end
    return ψ
end

"""
    truncate_unitcell!(ψ, χ; cutoff=SVDTOL, recanonicalize=false)

Compress a dense iMPS unit cell to at most `χ`
on each bond.

Parameters:
- `ψ`
  State updated in place.
- `χ`
  Target bond dimension after projection.

Keyword arguments:
- `cutoff=SVDTOL`
  Singular-value cutoff used during truncation.
- `recanonicalize=false`
  On the large-cell path, restore the global canonical gauge after the
  sweep with [`restore_canonical!`](@ref) and `method=:auto`. The grouped small-cell path
  already includes canonicalization.

Returns:
- The same object `ψ`, mutated in place.

Notes:
- Small cells with equal physical dimensions are grouped, compressed, and
  canonicalized. Large and heterogeneous cells use a local SVD sweep with
  bounded memory. Wide local cuts first factor the right tensor by LQ,
  which preserves their nonzero singular spectra and avoids fusing two
  physical legs. Gauge restoration also works site by site and requires an
  injective state on the retained bonds.
- Both paths split the cell bond by bond in a fixed order, so the bonds are
  not treated alike. On a cell that repeats a smaller cell this can break the
  repetition (a spurious cell doubling). [`truncate!`](@ref) instead keeps the
  `χ` largest Schmidt values of the canonical form on every bond.
"""
function truncate_unitcell!(ψ::iMPS, χ::Integer; cutoff::Real=SVDTOL,
                            recanonicalize::Bool=false)
    _validate_truncation_args(χ, 1, 0.0, cutoff)
    d = size(ψ.Γ[1], 2)
    if ψ.n == 1
        # canonical! routes through the vector schmidt_canonical wrapper,
        # whose isone(n) path finishes with the right-canonical trace
        # normalization. The previous code called the scalar kernel — which
        # already returns the stored-convention tensor B = Γ·λ — and then
        # applied tensor_lmul!(λ, Γ) on top, silently storing λ·B: an extra
        # diag(λ) on every bond of the infinite chain, not a gauge change.
        canonical!(ψ; maxdim=χ, cutoff, renormalize=true)
        return ψ
    end

    # Size guard: avoid exponential memory blowup from grouping the entire unit cell.
    # The grouped tensor has shape (χ, d^n, χ), so its physical dimension d^n
    # grows exponentially with the unit-cell length n.
    can_group = _can_group_unitcell(ψ)
    if !can_group || !all(A -> size(A, 2) == d, ψ.Γ)
        if !can_group
            @warn "Large unit cell detected (n=$(ψ.n)). " *
                  "Using a site-by-site compression sweep." maxlog=1
        end
        # The sweep replaces tensors rather than changing their storage.
        # Keep its result separate until the requested gauge solve succeeds.
        trial = recanonicalize ? iMPS(copy(ψ.Γ), copy(ψ.λ), ψ.n) : ψ
        # This is the identity-gate sweep without building, permuting, or
        # multiplying by an explicit identity on the two physical legs.
        for i in 1:ψ.n
            j = mod1(i + 1, ψ.n)
            λl = trial.λ[mod1(i - 1, ψ.n)]
            left, spectrum, right = _compression_cut(trial.Γ[i], trial.Γ[j], λl, χ, cutoff)
            tensor_lmul!(_safe_reciprocal(λl; atol=ZEROTOL, rtol=eps(Float64)), left)
            tensor_rmul!(left, spectrum)
            trial.Γ[i], trial.Γ[j] = left, right
            trial.λ[i] = spectrum
        end
        if recanonicalize
            _restore_scarfinder_gauge!(trial)
            ψ.Γ .= trial.Γ
            ψ.λ .= trial.λ
        end
        return ψ
    end

    Γ = tensor_group(ψ.Γ)
    A, λ = schmidt_canonical(Γ, ψ.λ[end]; maxdim=χ, cutoff, renormalize=true)
    tensor_lmul!(λ, A)
    Γs, λs = tensor_decomp!(A, λ, ψ.n; maxdim=χ, svd_min=cutoff, renormalize=true)
    push!(λs, λ)
    ψ.Γ .= Γs
    ψ.λ .= λs
    canonical!(ψ; maxdim=χ, cutoff, renormalize=true)
    return ψ
end

function _compression_cut(A, B, λ, maxdim, cutoff)
    l, d1, m = size(A)
    _, d2, r = size(B)
    if cutoff > 0 && d2*r >= max(64, 4m) &&
            cutoff > eps(real(eltype(A))) * norm(A) * norm(B) * maximum(abs, λ)
        # B=LQ and QQ†=I. The weighted cut is (Λ A L)Q and its
        # nonzero singular values are those of the narrower Λ A L.
        _finite_entries(B, "right site tensor")
        factor = lq(reshape(B, m, d2*r))
        core = reshape(A, l*d1, m) * Matrix(factor.L)
        tensor_lmul!(λ, reshape(core, l, d1, :))
        U, S, V = svd_trim(core; maxdim, svd_min=cutoff, renormalize=true)
        return reshape(U, l, d1, :), S, reshape(V * Matrix(factor.Q), :, d2, r)
    end
    # Below the roundoff scale, the full SVD may keep additional modes in
    # the nominal null space. Keep the requested direct-cut behavior there.
    block = tensor_group_2(A, B)
    tensor_lmul!(λ, block)
    return tensor_svd(reshape(block, l, d1, d2, r);
        maxdim, svd_min=cutoff, renormalize=true)
end

function _restore_scarfinder_gauge!(ψ::DenseIMPS)
    tolerance = 10 * _qr_target(ψ, nothing, nothing)[1]
    _check_restored_gauge(ψ, tolerance) || restore_canonical!(ψ; method=:auto)
    return ψ
end

const _truncate_unitcell! = truncate_unitcell!

"""
    _warn_scarfinder_nstep(nstep, mode)

Emit a one-time warning when `nstep == 1` in a ScarFinder real-time stage.

This is an internal helper used to flag the degenerate case where one
ScarFinder iteration contains only a single microscopic evolution step before
projection. That setting is still allowed for backwards compatibility, but it
is usually too aggressive to represent the intended coarse-grained ScarFinder
flow.
"""
function _warn_scarfinder_nstep(nstep::Integer, mode::Symbol)
    nstep >= 1 || throw(ArgumentError("nstep must be positive"))
    if nstep == 1
        label = mode === :hamiltonian ? "Hamiltonian-based" : "Gate-based"
        @warn "$label ScarFinder is using nstep = 1. This is usually too small: one ScarFinder iteration should normally contain multiple microscopic evolution steps before projection. Consider choosing nstep > 1 to avoid an overly aggressive truncation cycle." maxlog=1
    end
    return nothing
end

"""
    energy_density(ψ, h; span=operator_span(ψ, h))

Return the unit-cell averaged expectation value of the local operator `h`.

This is the average of `expect(ψ, h, i, i + span - 1)` over all starting sites
in the unit cell.

Parameters:
- `ψ`
  State in which the expectation value is measured.
- `h`
  Local operator interpreted as a Hamiltonian density or other local observable.

Keyword arguments:
- `span=operator_span(ψ, h)`
  Number of sites acted on by `h`.

Returns:
- Unit-cell averaged expectation value of `h`.

Notes:
- This is the quantity used by ScarFinder when `target` energy fixing is
  enabled.
"""
function energy_density(ψ::iMPS, h::AbstractMatrix; span::Integer=operator_span(ψ, h))
    _validate_local_span(ψ, span)
    d = size(ψ.Γ[1], 2)
    if all(A -> size(A, 2) == d, ψ.Γ)
        return _energy_density_internal(ψ, convert_operator(h, d, span), span)
    end
    E = 0.0
    for i in 1:ψ.n
        E += expect(ψ, h, i, mod1(i + span - 1, ψ.n))
    end
    return E / ψ.n
end

function _energy_density_internal(ψ, O, span)
    E = 0.0
    for i in 1:ψ.n
        inds = _gate_indices(ψ, i, mod1(i + span - 1, ψ.n))
        E += _expect_internal(ψ, O, inds)
    end
    return E / ψ.n
end

"""
    energy_span(n, d, h; dτ=0.1, Nτ=1000, maxdim=32)

Estimate the lowest- and highest-energy density reachable by imaginary-time iTEBD
using the local operator `h`.

This helper is useful for choosing a target energy when using `scarfinder!`.
The result is `(Emin, Emax, (ψmin, ψmax))`.

Parameters:
- `n`
  Unit-cell length of the trial states used in the estimate.
- `d`
  Local Hilbert-space dimension.
- `h`
  Local Hamiltonian density.

Keyword arguments:
- `dτ=0.1`
  Imaginary-time step used to build `exp(∓dτ * h)`.
- `Nτ=1000`
  Number of imaginary-time steps used for the low- and high-energy estimates.
- `maxdim=32`
  Maximum temporary bond dimension used during those imaginary-time updates.

Returns:
- `(Emin, Emax, (ψmin, ψmax))` where `Emin` and `Emax` are the estimated energy
  densities and `ψmin`, `ψmax` are the corresponding trial states.

Notes:
- This is a heuristic helper, not a rigorous variational bound.
"""
function energy_span(
    n::Integer,
    d::Integer,
    h::AbstractMatrix;
    dτ::Real=0.1,
    Nτ::Integer=1000,
    maxdim::Integer=32
)
    ψmin = rand_iMPS(ComplexF64, n, d, 1)
    Gmin = exp(-dτ * h)
    span = operator_span(ψmin, h)
    for _ in 1:Nτ
        _evolve_uniform!(ψmin, Gmin; span, maxdim)
    end
    canonical!(ψmin)
    Emin = energy_density(ψmin, h; span)

    ψmax = rand_iMPS(ComplexF64, n, d, 1)
    Gmax = exp(dτ * h)
    for _ in 1:Nτ
        _evolve_uniform!(ψmax, Gmax; span, maxdim)
    end
    canonical!(ψmax)
    Emax = energy_density(ψmax, h; span)

    return Emin, Emax, (ψmin, ψmax)
end

"""
    fix_energy!(ψ, h, χ; target, span=operator_span(ψ, h), tol=1e-6, α=0.1, maxstep=50, recanonicalize=true)

Move the energy density toward `target` by normalized imaginary-time sweeps.
Each trial uses at most `χ` on the bonds it updates. Rejected trials are
discarded, so `ψ` retains the best state found. `α` sets the initial step
size relative to the initial energy error, and `maxstep` bounds the number
of trial sweeps. On small grouped cells, start from a canonical state.
Those cells recanonicalize entangled trials before measurement. On large
cells the default checks the initial gauge and restores it as needed before
the initial energy measurement. It also restores entangled trials before
measuring them. With `recanonicalize=false`, large cells use the stored
gauge for both measurements.

Parameters:
- `ψ`
  State updated in place.
- `h`
  Local Hamiltonian density used to measure the energy drift.
- `χ`
  Bond dimension used during the correction steps.

Keyword arguments:
- `span`
  Number of sites acted on by `h`.
- `target`
  Target energy density.
- `tol=1e-6`
  Desired absolute tolerance on the energy density.
- `α=0.1`
  Step-size parameter for the imaginary-time correction.
- `maxstep=50`
  Maximum number of substeps in the correction loop.
- `recanonicalize=true`
  Ensure the initial state and entangled trials are in canonical gauge on
  large cells before measuring their energy.

Returns:
- `(converged, residual, steps)`, where `residual` is the measured energy
  density minus `target` in the state left in `ψ`.
"""
function fix_energy!(
    ψ::iMPS,
    h::AbstractMatrix,
    χ::Integer;
    span::Integer=operator_span(ψ, h),
    target::Real,
    tol::Real=1e-6,
    α::Real=0.1,
    maxstep::Integer=50,
    recanonicalize::Bool=true,
)
    ishermitian(h) || throw(ArgumentError("fix_energy! requires a Hermitian density h"))
    isfinite(target) || throw(ArgumentError("target must be finite"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    isfinite(α) && α >= 0 || throw(ArgumentError("α must be finite and nonnegative"))
    maxstep >= 0 || throw(ArgumentError("maxstep must be nonnegative"))
    _validate_truncation_args(χ, 1, 0.0, SVDTOL)
    _validate_local_span(ψ, span)
    d = size(ψ.Γ[1], 2)
    O = all(A -> size(A, 2) == d, ψ.Γ) ? convert_operator(h, d, span) : nothing
    can_group = !isnothing(O) && _can_group_unitcell(ψ)
    recanonicalize && !can_group && _restore_scarfinder_gauge!(ψ)
    measure(state) = isnothing(O) ? energy_density(state, h; span) : _energy_density_internal(state, O, span)
    residual = measure(ψ) - target
    steps = 0
    if abs(residual) <= tol || maxstep == 0 || iszero(α)
        return (converged=abs(residual) <= tol, residual, steps)
    end

    eh = eigen(Hermitian(h))
    trial = deepcopy(ψ)
    dτ = α * residual
    vectors = convert(Matrix{promote_type(eltype(eh.vectors), typeof(float(dτ)))}, eh.vectors)
    weighted_vectors = similar(vectors)
    G = similar(vectors)
    while steps < maxstep && isfinite(dτ) && !iszero(dτ)
        # Shift the exponent by a scalar so the normalized sweep cannot
        # overflow just because the zero of energy was changed.
        reference = dτ > 0 ? first(eh.values) : last(eh.values)
        weights = exp.(-dτ .* (eh.values .- reference))
        weighted_vectors .= vectors .* transpose(weights)
        mul!(G, weighted_vectors, vectors')
        _evolve_uniform!(trial, G; span, maxdim=χ)
        if any(λ -> length(λ) > 1, trial.λ)
            if can_group
                canonical!(trial; maxdim=χ, noninjective=:ignore)
            elseif recanonicalize
                _restore_scarfinder_gauge!(trial)
            end
        end
        attempted = measure(trial) - target
        steps += 1
        if abs(attempted) < abs(residual)
            _copy_imps_state!(ψ, trial)
            abs(attempted) <= tol && return (converged=true, residual=attempted, steps)
            # Predict the next signed step from the measured local slope,
            # limiting growth in case truncation changes that slope.
            dτ *= clamp(attempted / (residual - attempted), -2, 2)
            residual = attempted
        else
            _copy_imps_state!(trial, ψ)
            dτ /= 2
        end
    end
    return (converged=false, residual, steps)
end

function _energy_fix!(ψ::iMPS, h::AbstractMatrix, χ::Integer; target::Real, kwargs...)
    result = fix_energy!(ψ, h, χ; target, kwargs...)
    result.converged || @warn "ScarFinder energy correction did not converge" target result.residual result.steps maxlog=1
    return ψ
end

"""
    minimize_on_trajectory!(f, step!, ψ, samples)

Apply `step!(trial)` in place `samples` times on a copy of `ψ`, and evaluate the
ordered real objective `f(trial)` at the initial state and after every step.
`f` should only read the state. The return value of `step!` is ignored. Keep the
first point that attains the minimum and update `ψ` in place to that sampled
state.

Return `(step, value)` as a named tuple. The initial point has `step=0` and
`value` is the objective measured at the selected point. The scan never
replays `step!` to recover the selected state.
"""
function minimize_on_trajectory!(f, step!, ψ::iMPS, samples::Integer)
    samples >= 0 || throw(ArgumentError("samples must be nonnegative"))
    ψtrial = deepcopy(ψ)
    best_value = f(ψtrial)
    best_step = 0
    best_state = nothing
    for step in 1:samples
        step!(ψtrial)
        value = f(ψtrial)
        if value < best_value
            best_value = value
            best_step = step
            best_state = deepcopy(ψtrial)
        end
    end
    # Retain the tensors of the sampled state, including its gauge.
    isnothing(best_state) || _copy_imps_state!(ψ, best_state)
    return (step=best_step, value=best_value)
end

function _minimize_on_trajectory!(f, step!, ψ::iMPS, samples::Integer)
    minimize_on_trajectory!(f, step!, ψ, samples)
    return ψ
end

"""
    _refinement_objective(ψ)

Worst-case bond entropy of the iMPS unit cell, used as the objective for the
ScarFinder refinement scan. Minimizing this picks the trajectory point with
the lowest peak entanglement across all bonds, which is the right criterion
for a translationally-invariant scar state: a low maximum across bonds means
all bonds are low. Previously the scan minimized `ent_S(ψ, ψ.n)` (only the
wraparound bond), which for `n > 1` is one bond out of many and can prefer
states where entanglement has merely migrated to other bonds.
"""
_refinement_objective(ψ::iMPS) = maximum(ent_S(ψ, i) for i in 1:ψ.n)

function _copy_imps_state!(dst::iMPS, src::iMPS)
    dst.n == src.n ||
        throw(DimensionMismatch("cannot copy iMPS states with different unit-cell lengths"))
    resize!(dst.Γ, length(src.Γ))
    resize!(dst.λ, length(src.λ))
    for i in eachindex(src.Γ)
        dst.Γ[i] = copy(src.Γ[i])
    end
    for i in eachindex(src.λ)
        dst.λ[i] = copy(src.λ[i])
    end
    return dst
end

function _finish_scarfinder_step!(ψ, h, χ; span, target, tol, α, maxstep, cutoff, recanonicalize)
    truncate_unitcell!(ψ, χ; cutoff, recanonicalize)
    isnothing(target) || _energy_fix!(ψ, h, χ; span, target, tol, α, maxstep, recanonicalize)
    return ψ
end

const _ScarFinderGateList = Union{AbstractVector,Tuple}

function _scarfinder_gate_step!(
    ψ, gates, h, χ, mode::Symbol;
    nstep::Integer=10,
    maxdim::Integer=MAXDIM,
    hspan::Integer=isnothing(h) ? 0 : operator_span(ψ, h),
    target::Union{Real,Nothing}=nothing,
    tol::Real=1e-6,
    α::Real=0.1,
    maxstep::Integer=50,
    cutoff::Real=SVDTOL,
    recanonicalize::Bool=true,
)
    _validate_truncation_args(χ, 1, 0.0, cutoff)
    if !isnothing(target)
        isnothing(h) && throw(ArgumentError("supply a Hamiltonian density h when setting target"))
        _validate_local_span(ψ, hspan; name="hspan")
    end
    _warn_scarfinder_nstep(nstep, mode)
    evolve!(ψ, gates, nstep; maxdim)
    return _finish_scarfinder_step!(ψ, h, χ;
        span=hspan, target, tol, α, maxstep, cutoff, recanonicalize)
end

"""
    scarfinder_step!(ψ, gates, χ; nstep=10, maxdim=MAXDIM, cutoff=SVDTOL, recanonicalize=true)
    scarfinder_step!(ψ, gates, h, χ; nstep=10, maxdim=MAXDIM, cutoff=SVDTOL, recanonicalize=true, target=nothing, ...)

Apply `nstep` sweeps of a prepared vector or tuple of `(G, i, j)` updates
using [`evolve!`](@ref), then compress to `χ`. The sequence may contain both
ordinary matrices, [`EndpointGate`](@ref)s and [`ControlledGate`](@ref)s.
Supply a separate local density
`h` and `target` to correct the energy. The correction also accepts `hspan`,
`tol`, `α`, and `maxstep`. On large cells, restore the canonical gauge after
compression and during energy-fixing trials by default.
"""
function scarfinder_step!(ψ::DenseIMPS, gates::_ScarFinderGateList, χ::Integer; kwargs...)
    return _scarfinder_gate_step!(ψ, gates, nothing, χ, :gate; kwargs...)
end

function scarfinder_step!(ψ::DenseIMPS, gates::_ScarFinderGateList, h::AbstractMatrix, χ::Integer; kwargs...)
    return _scarfinder_gate_step!(ψ, gates, h, χ, :gate; kwargs...)
end

"""
    scarfinder_step!(ψ, h, dt, χ; keywords...)

Perform one hybrid ScarFinder step:

1. real-time evolution under `h`,
2. truncation back to bond dimension `χ`,
3. optional imaginary-time correction to match a target energy density.

Keyword arguments:
- `nstep=10`: number of real-time TEBD steps before truncation.
- `maxdim=MAXDIM`: temporary bond dimension used during the real-time evolution.
- `span=operator_span(ψ, h)`: number of sites acted on by `h`.
- `trotter=nothing`: use the original uniform sweep. Pass `:second`, `:fourth`,
  or `:fourth_opt` to use Trotter gates.
- `layers=nothing`: with a Trotter scheme, generate disjoint layers from `h`
  by default. Pass commuting layers explicitly to override this grouping.
- `target=nothing`: target energy density. If not `nothing`, an energy correction is applied.
- `tol=1e-6`, `α=0.1`, `maxstep=50`: parameters for the energy correction.
- `cutoff=SVDTOL`: truncation cutoff used when compressing back to bond dimension `χ`.

When `dt` is a microscopic gate time but each ScarFinder iteration is intended to
represent a larger physical interval `Δt`, set `nstep ≈ Δt / dt`. For example,
`dt = 0.01` and `Δt = 0.1` should typically use `nstep = 10`.

As a practical rule, `nstep = 1` is usually too small for ScarFinder because it
reduces one iteration to a single microscopic evolution step before projection.
The implementation therefore warns when `nstep == 1`, even though that setting
is still accepted for backwards compatibility.

Parameters:
- `ψ`
  Trial state updated in place.
- `h`
  Local Hamiltonian density used both for real-time evolution and, if enabled,
  energy fixing.
- `dt`
  Microscopic real-time step used to build `exp(-1im * dt * h)`.
- `χ`
  Target bond dimension after the projection step.

Keyword arguments:
- `nstep=10`
  Number of microscopic evolution steps before the projection back to `χ`.
- `maxdim=MAXDIM`
  Maximum temporary bond dimension during the real-time evolution stage.
- `span=operator_span(ψ, h)`
  Number of sites acted on by `h`.
- `target=nothing`
  Target energy density. If `nothing`, no energy-fixing step is performed.
- `tol=1e-6`, `α=0.1`, `maxstep=50`
  Parameters controlling the energy-fixing loop.
- `cutoff=SVDTOL`
  Singular-value cutoff used when truncating back to bond dimension `χ`.
- `recanonicalize=true`
  Restore the canonical gauge after local compression sweeps and during energy
  correction on large cells.

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder_step!(
    ψ::iMPS,
    h::AbstractMatrix,
    dt::Real,
    χ::Integer;
    nstep::Integer=10,
    maxdim::Integer=MAXDIM,
    span::Integer=operator_span(ψ, h),
    target::Union{Real,Nothing}=nothing,
    tol::Real=1e-6,
    α::Real=0.1,
    maxstep::Integer=50,
    cutoff::Real=SVDTOL,
    recanonicalize::Bool=true,
    trotter::Union{Nothing,Symbol}=nothing,
    layers=nothing,
)
    if !isnothing(trotter)
        resolved_layers = isnothing(layers) ? uniform_layers(ψ, h; span) : layers
        gates = trotter_gates(resolved_layers, dt; trotter)
        return _scarfinder_gate_step!(ψ, gates, h, χ, :hamiltonian;
            nstep, maxdim, hspan=span, target, tol, α, maxstep, cutoff, recanonicalize)
    end
    isnothing(layers) || throw(ArgumentError("layers requires a trotter scheme"))
    _warn_scarfinder_nstep(nstep, :hamiltonian)
    G = exp(-1im * dt * h)
    for _ in 1:nstep
        _evolve_uniform!(ψ, G; span, maxdim)
    end
    return _finish_scarfinder_step!(ψ, h, χ; span, target, tol, α, maxstep, cutoff, recanonicalize)
end

"""
    scarfinder_step!(ψ, G, χ; keywords...)

Perform one gate-based ScarFinder step.

This variant is useful when the driving object is already given as a local gate,
for example a projected Floquet-like step such as `G = P * U`. The routine:

1. applies the gate `G`,
2. truncates the state back to bond dimension `χ`.

Unlike the Hamiltonian-based method, this variant does not perform energy fixing.

Keyword arguments:
- `nstep=10`: number of times to apply `G` before truncation.
- `maxdim=MAXDIM`: temporary bond dimension used during the gate evolution.
- `span=operator_span(ψ, G)`: number of sites acted on by `G`.
- `cutoff=SVDTOL`: truncation cutoff used when compressing back to bond dimension `χ`.

This gate-only variant does not perform energy fixing. If the intended workflow
uses a custom gate for evolution but still needs Hamiltonian-based energy
correction, prefer `scarfinder_step!(ψ, G, h, χ; ...)`.

As a practical rule, `nstep = 1` is usually too small for ScarFinder because it
collapses one iteration into a single microscopic gate application followed by
projection. The implementation therefore warns when `nstep == 1`, even though
that setting is still accepted for backwards compatibility.

Parameters:
- `ψ`
  Trial state updated in place.
- `G`
  Local gate applied during the real-time stage.
- `χ`
  Target bond dimension after truncation.

Keyword arguments:
- `nstep=10`
  Number of gate applications before truncation.
- `maxdim=MAXDIM`
  Maximum temporary bond dimension during those gate applications.
- `span=operator_span(ψ, G)`
  Number of sites acted on by `G`.
- `cutoff=SVDTOL`
  Singular-value cutoff used when truncating back to bond dimension `χ`.
- `recanonicalize=true`
  Restore the canonical gauge after local compression sweeps.

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder_step!(
    ψ::iMPS,
    G::AbstractMatrix,
    χ::Integer;
    nstep::Integer=10,
    maxdim::Integer=MAXDIM,
    span::Integer=operator_span(ψ, G),
    cutoff::Real=SVDTOL,
    recanonicalize::Bool=true
)
    _warn_scarfinder_nstep(nstep, :gate)
    for _ in 1:nstep
        _evolve_uniform!(ψ, G; span, maxdim)
    end
    return truncate_unitcell!(ψ, χ; cutoff, recanonicalize)
end

"""
    scarfinder_step!(ψ, G, h, χ; keywords...)

Perform one gate-based ScarFinder step with Hamiltonian-based energy fixing.

This variant is designed for cases where the update rule is given by a local gate
`G`, but the target constraint should still be imposed with respect to a local
Hamiltonian `h`. A typical example is a projected update such as `G = P * U`
in the PXP model.

The routine:

1. applies the gate `G`,
2. truncates the state back to bond dimension `χ`,
3. optionally applies the same energy-fixing step used by the Hamiltonian-based
   ScarFinder, now measured with `h`.

Keyword arguments:
- `nstep=10`: number of times to apply `G` before truncation.
- `maxdim=MAXDIM`: temporary bond dimension used during the gate evolution.
- `span=operator_span(ψ, G)`: number of sites acted on by `G`.
- `hspan=operator_span(ψ, h)`: number of sites acted on by `h`.
- `target=nothing`: target energy density. If not `nothing`, an energy correction is applied.
- `tol=1e-6`, `α=0.1`, `maxstep=50`: parameters for the energy correction.
- `cutoff=SVDTOL`: truncation cutoff used when compressing back to bond dimension `χ`.

For constrained models such as PXP, it is common to choose `G` as a projected
update that restores the physical subspace after truncation, while `h` remains
the unprojected local Hamiltonian density used for energy fixing. These two
objects should generally not be conflated.

As with the Hamiltonian-based interface, if `G` is built from a microscopic time
step `dt` but one ScarFinder iteration should represent a larger interval `Δt`,
set `nstep ≈ Δt / dt`.

As a practical rule, `nstep = 1` is usually too small for ScarFinder because it
reduces each iteration to a single microscopic gate application before
projection. The implementation therefore warns when `nstep == 1`, even though
that setting is still accepted for backwards compatibility.

Parameters:
- `ψ`
  Trial state updated in place.
- `G`
  Local gate used for the real-time evolution stage.
- `h`
  Local Hamiltonian density used only for energy fixing.
- `χ`
  Target bond dimension after the projection step.

Keyword arguments:
- `nstep=10`
  Number of gate applications before truncation.
- `maxdim=MAXDIM`
  Maximum temporary bond dimension during the gate evolution.
- `span=operator_span(ψ, G)`
  Number of sites acted on by `G`.
- `hspan=operator_span(ψ, h)`
  Number of sites acted on by `h` when measuring and correcting the energy.
- `target=nothing`
  Target energy density. If `nothing`, no energy correction is applied.
- `tol=1e-6`, `α=0.1`, `maxstep=50`
  Parameters controlling the energy-fixing loop.
- `cutoff=SVDTOL`
  Singular-value cutoff used during projection back to bond dimension `χ`.
- `recanonicalize=true`
  Restore the canonical gauge after local compression sweeps and during energy
  correction on large cells.

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder_step!(
    ψ::iMPS,
    G::AbstractMatrix,
    h::AbstractMatrix,
    χ::Integer;
    nstep::Integer=10,
    maxdim::Integer=MAXDIM,
    span::Integer=operator_span(ψ, G),
    hspan::Integer=operator_span(ψ, h),
    target::Union{Real,Nothing}=nothing,
    tol::Real=1e-6,
    α::Real=0.1,
    maxstep::Integer=50,
    cutoff::Real=SVDTOL,
    recanonicalize::Bool=true
)
    _warn_scarfinder_nstep(nstep, :gate)
    for _ in 1:nstep
        _evolve_uniform!(ψ, G; span, maxdim)
    end
    return _finish_scarfinder_step!(ψ, h, χ; span=hspan, target, tol, α, maxstep, cutoff, recanonicalize)
end

"""
    scarfinder!(ψ, h, dt, χ, N; keywords...)

Run `N` hybrid ScarFinder iterations in place.

By default the final state is additionally refined by scanning a shorter-time
trajectory and selecting the minimum-entanglement point, which matches the workflow
used in the reference `ScarFinder` scripts.

Keyword arguments:
- `refine=true`: enable the minimum-entanglement trajectory scan.
- `refine_dt=dt/10`: time step used during the refinement scan.
- `refine_step=100`: number of trial points in the refinement scan.

All keyword arguments accepted by `scarfinder_step!` are also supported.

The return value is `ψ`, mutated in place.

Parameters:
- `ψ`
  Trial state updated in place.
- `h`
  Local Hamiltonian density.
- `dt`
  Microscopic real-time step used in each hybrid ScarFinder iteration.
- `χ`
  Target bond dimension of the projected manifold.
- `N`
  Number of ScarFinder iterations.

Keyword arguments:
- `refine=true`
  Whether to perform the post-processing scan for a minimum-entanglement point.
- `refine_dt=dt / 10`
  Microscopic step used during that refinement scan.
- `refine_step=100`
  Number of trial points considered in the refinement scan.
- `trotter=nothing`, `layers=nothing`
  Opt in to a Trotter formula for both the main evolution and refinement.
  Use `layers` to override the automatically grouped uniform density.
- `kwargs...`
  Additional keyword arguments forwarded to [`scarfinder_step!`](@ref).

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder!(
    ψ::iMPS,
    h::AbstractMatrix,
    dt::Real,
    χ::Integer,
    N::Integer;
    refine::Bool=true,
    refine_dt::Real=dt / 10,
    refine_step::Integer=100,
    trotter::Union{Nothing,Symbol}=nothing,
    layers=nothing,
    span::Integer=operator_span(ψ, h),
    kwargs...
)
    if !isnothing(trotter)
        resolved_layers = isnothing(layers) ? uniform_layers(ψ, h; span) : layers
        gates = trotter_gates(resolved_layers, dt; trotter)
        refine_gates = refine ? trotter_gates(resolved_layers, refine_dt; trotter) : nothing
        for _ in 1:N
            _scarfinder_gate_step!(ψ, gates, h, χ, :hamiltonian; hspan=span, kwargs...)
        end
        if refine
            step! = ψ0 -> _scarfinder_gate_step!(ψ0, refine_gates, h, χ, :hamiltonian;
                hspan=span, kwargs...)
            minimize_on_trajectory!(_refinement_objective, step!, ψ, refine_step)
        end
        return ψ
    end
    isnothing(layers) || throw(ArgumentError("layers requires a trotter scheme"))
    # Build the real-time gate once per scarfinder! call instead of per
    # scarfinder_step! call. Previously exp(-1im*dt*h) was a dense matrix
    # exponential redone on every iteration (N times for the main loop plus
    # refine_step times for the refinement scan), which dominated the cost
    # for large local Hilbert spaces. Dispatching to the mixed
    # scarfinder_step!(ψ, G, h, χ; ...) interface keeps the energy-fixing
    # path identical because G = exp(-1im*dt*h) has the same support as h.
    G_main = exp(-1im * dt * h)
    for _ in 1:N
        scarfinder_step!(ψ, G_main, h, χ; span, kwargs...)
    end
    if refine
        G_refine = exp(-1im * refine_dt * h)
        step! = ψ0 -> scarfinder_step!(ψ0, G_refine, h, χ; span, kwargs...)
        minimize_on_trajectory!(_refinement_objective, step!, ψ, refine_step)
    end
    return ψ
end

"""
    scarfinder!(ψ, gates, χ, N; refine=true, refine_step=100, refine_gates=gates, ...)
    scarfinder!(ψ, gates, h, χ, N; refine=true, refine_step=100, refine_gates=gates, ...)

Run `N` ScarFinder iterations using a prepared sequence of `(G, i, j)` updates.
By default refinement reuses that sequence. Pass `refine_gates` to scan with
a different prepared sequence. Other keywords go to [`scarfinder_step!`](@ref).
An independent Hamiltonian density `h` enables energy fixing with `target`.
"""
function scarfinder!(ψ::DenseIMPS, gates::_ScarFinderGateList,
                     χ::Integer, N::Integer; kwargs...)
    return scarfinder!(ψ, gates, nothing, χ, N; kwargs...)
end

function scarfinder!(ψ::DenseIMPS, gates::_ScarFinderGateList,
                     h::Union{Nothing,AbstractMatrix}, χ::Integer, N::Integer;
                     refine::Bool=true, refine_step::Integer=100,
                     refine_gates::_ScarFinderGateList=gates, kwargs...)
    N >= 0 || throw(ArgumentError("N must be nonnegative"))
    (!refine || refine_step >= 0) || throw(ArgumentError("refine_step must be nonnegative"))
    for _ in 1:N
        _scarfinder_gate_step!(ψ, gates, h, χ, :gate; kwargs...)
    end
    if refine
        step! = state -> _scarfinder_gate_step!(state, refine_gates, h, χ, :gate; kwargs...)
        minimize_on_trajectory!(_refinement_objective, step!, ψ, refine_step)
    end
    return ψ
end

"""
    scarfinder!(ψ, G, χ, N; keywords...)

Run `N` gate-based ScarFinder iterations in place.

This is the high-level companion of `scarfinder_step!(ψ, G, χ; ...)` and is
intended for cases where the update rule is already provided as a local gate,
for example `G = P * U` in constrained or projected dynamics.

Keyword arguments:
- `refine=true`: enable the minimum-entanglement trajectory scan.
- `refine_step=100`: number of trial points in the refinement scan.

All keyword arguments accepted by `scarfinder_step!(ψ, G, χ; ...)` are also supported.
The return value is `ψ`, mutated in place.

Parameters:
- `ψ`
  Trial state updated in place.
- `G`
  Local gate used in the real-time stage.
- `χ`
  Target bond dimension after each projection.
- `N`
  Number of ScarFinder iterations.

Keyword arguments:
- `refine=true`
  Whether to perform the minimum-entanglement refinement scan.
- `refine_step=100`
  Number of trial points used in that scan.
- `kwargs...`
  Additional keyword arguments forwarded to `scarfinder_step!(ψ, G, χ; ...)`.

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder!(
    ψ::iMPS,
    G::AbstractMatrix,
    χ::Integer,
    N::Integer;
    refine::Bool=true,
    refine_step::Integer=100,
    kwargs...
)
    for _ in 1:N
        scarfinder_step!(ψ, G, χ; kwargs...)
    end
    if refine
        step! = ψ0 -> scarfinder_step!(ψ0, G, χ; kwargs...)
        minimize_on_trajectory!(_refinement_objective, step!, ψ, refine_step)
    end
    return ψ
end

"""
    scarfinder!(ψ, G, h, χ, N; keywords...)

Run `N` gate-based ScarFinder iterations in place while using `h` for energy fixing.

This is the high-level interface for projected or constrained dynamics where the
update rule is already encoded in a gate `G`, but the energy target should still
be measured using a Hamiltonian density `h`.

Keyword arguments:
- `refine=true`: enable the minimum-entanglement trajectory scan.
- `refine_step=100`: number of trial points in the refinement scan.

All keyword arguments accepted by `scarfinder_step!(ψ, G, h, χ; ...)` are also supported.
The return value is `ψ`, mutated in place.

Parameters:
- `ψ`
  Trial state updated in place.
- `G`
  Local gate used for the real-time evolution stage.
- `h`
  Local Hamiltonian density used for energy fixing.
- `χ`
  Target bond dimension after each projection.
- `N`
  Number of ScarFinder iterations.

Keyword arguments:
- `refine=true`
  Whether to perform the minimum-entanglement refinement scan.
- `refine_step=100`
  Number of trial points used in that scan.
- `kwargs...`
  Additional keyword arguments forwarded to
  `scarfinder_step!(ψ, G, h, χ; ...)`.

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder!(
    ψ::iMPS,
    G::AbstractMatrix,
    h::AbstractMatrix,
    χ::Integer,
    N::Integer;
    refine::Bool=true,
    refine_step::Integer=100,
    kwargs...
)
    for _ in 1:N
        scarfinder_step!(ψ, G, h, χ; kwargs...)
    end
    if refine
        step! = ψ0 -> scarfinder_step!(ψ0, G, h, χ; kwargs...)
        minimize_on_trajectory!(_refinement_objective, step!, ψ, refine_step)
    end
    return ψ
end
