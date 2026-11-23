#---------------------------------------------------------------------------------------------------
# ScarFinder
#---------------------------------------------------------------------------------------------------
export operator_span, energy_density, energy_span
export scarfinder_step!, scarfinder!, minimize_on_trajectory!
export uniform_layers, project!, truncate_unitcell!, fix_energy!

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
    _warn_scarfinder_nstep(nstep)

Throw for `nstep < 1` and warn once for `nstep == 1`, where one ScarFinder
iteration contains a single microscopic step before projection.
"""
function _warn_scarfinder_nstep(nstep::Integer)
    nstep >= 1 || throw(ArgumentError("nstep must be positive"))
    nstep == 1 && @warn "ScarFinder is using nstep = 1. This is usually too small: one ScarFinder iteration should normally contain multiple microscopic evolution steps before projection." maxlog=1
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

#---------------------------------------------------------------------------------------------------
# ScarFinder: evolve off the manifold, project back onto it
#---------------------------------------------------------------------------------------------------
# One iteration is `evolve!(ψ, U, nstep)` followed by `project!`. `U` is anything `evolve!`
# accepts (a gate sequence, a vector of layer MPOs, …); `project!` is defined per manifold.

_uniform_terms(ψ::iMPS, h::AbstractMatrix) = (s = operator_span(ψ, h); [(h, i, i + s - 1) for i in 1:ψ.n])

# A single local gate is applied at every translation of the cell, left to right.
_evolution(ψ::iMPS, G::AbstractMatrix) =
    (s = operator_span(ψ, G); [(G, i, mod1(i + s - 1, ψ.n)) for i in 1:ψ.n])
_evolution(ψ::iMPS, U) = U

function _hamiltonian_evolution(ψ::iMPS, h::AbstractMatrix, dt::Real; trotter, layers)
    if isnothing(trotter)
        isnothing(layers) || throw(ArgumentError("layers requires a trotter scheme"))
        return _evolution(ψ, exp(-1im * dt * h))
    end
    return trotter_gates(isnothing(layers) ? uniform_layers(ψ, h) : layers, dt; trotter)
end

_term_period(ψ::iMPS, U) = U isa AbstractVector{<:iMPO} && !isempty(U) ? first(U).n : ψ.n

# Validate everything `project!` needs before any state changes.
function _projection_setup(ψ::iMPS, χ::Integer; cutoff, terms, target, tol, alpha, maxstep, n)
    ψ isa DenseIMPS || throw(ArgumentError("ScarFinder supports dense iMPS states"))
    _validate_mpo_truncation(χ, cutoff, 0.0)
    n > 0 || throw(ArgumentError("unit-cell length must be positive"))
    prep = isnothing(terms) ? nothing : _prepare_imaginary(terms; n, d=size(ψ.Γ[1], 2))
    isnothing(target) && return nothing
    isnothing(prep) && throw(ArgumentError("supply `terms` (or a density `h`) when setting `target`"))
    _validate_energy_fix_args(ψ, χ, prep; target, tol, alpha, maxstep, cutoff)
    eltype(ψ) <: Real && throw(ArgumentError("energy correction needs a complex iMPS; use mps_promote_type"))
    return prep
end

function _project!(ψ::iMPS, χ::Integer, prep; cutoff, target, tol, alpha, maxstep)
    before = iMPS(copy(ψ.Γ), copy(ψ.λ), ψ.n)
    truncate!(ψ, χ; cutoff)
    loss = -log(min(per_site_fidelity(before, ψ), 1.0))
    isnothing(prep) && return (; loss, energy_error=nothing, converged=true)
    r = _energy_fix_core!(ψ, prep, χ; target, tol, alpha, maxstep, cutoff)
    return (; loss, energy_error=r.residual, converged=r.converged)
end

"""
    project!(ψ, χ; cutoff=SVDTOL, terms=nothing, target=nothing, tol=1e-6, alpha=0.1, maxstep=50, n=ψ.n)

Project `ψ` onto the manifold of bond dimension `χ` at energy density `target`:
[`truncate!`](@ref) every bond to `χ`, then, if `target` is given, correct the
energy with [`energy_fix!`](@ref) using the terms `(h, i, j)` (period `n`).

Returns `(loss, energy_error, converged)`. `loss = −log F_site` between the
input and the truncated state, measured before energy correction.
`energy_error` is `E(ψ) − target` after correction (`nothing` without
`target`); `converged` tells whether it met `tol` (`true` without `target`).

This is the projection step of [`scarfinder_step!`](@ref). Another variational
manifold can take part in ScarFinder by defining its own `project!` (and an
`evolve!` for its evolution type) with the same return fields.
"""
function project!(ψ::iMPS, χ::Integer; cutoff::Real=SVDTOL, terms=nothing,
                  target::Union{Nothing,Real}=nothing, tol::Real=1e-6, alpha::Real=0.1,
                  maxstep::Integer=50, n::Integer=ψ.n)
    prep = _projection_setup(ψ, χ; cutoff, terms, target, tol, alpha, maxstep, n)
    return _project!(ψ, χ, prep; cutoff, target, tol, alpha, maxstep)
end

"""
    scarfinder_step!(ψ, U, χ; nstep=10, maxdim=MAXDIM, cutoff=SVDTOL, terms=nothing, target=nothing, tol=1e-6, alpha=0.1, maxstep=50, n)
    scarfinder_step!(ψ, U, h, χ; …)
    scarfinder_step!(ψ, h, dt, χ; trotter=nothing, layers=nothing, …)

One ScarFinder iteration: `evolve!(ψ, U, nstep; maxdim, cutoff)`, then
[`project!`](@ref)`(ψ, χ; …)`. Returns the named tuple of `project!`.

`U` is anything [`evolve!`](@ref) accepts: a sequence of `(G, i, j)` gates
(matrices, [`EndpointGate`](@ref)s, [`ControlledGate`](@ref)s), a vector of
layer MPOs from [`trotter_mpos`](@ref), or a single local gate `G`, applied at
every translation of the cell. A local density `h` in place of `terms` gives
the terms `(h, i, i + span − 1)` for every site of the cell. The form
`(h, dt)` evolves with `exp(−i dt h)` at every translation (`trotter=nothing`)
or with `trotter_gates(layers, dt; trotter)`, and corrects the energy with `h`.

The term period `n` defaults to the cell length of the first MPO, or `ψ.n`.
All inputs are validated before `ψ` changes. `nstep = 1` warns: one iteration
should contain several microscopic steps.
"""
function scarfinder_step!(ψ::iMPS, U, χ::Integer; nstep::Integer=10, maxdim::Integer=MAXDIM,
                          cutoff::Real=SVDTOL, terms=nothing,
                          target::Union{Nothing,Real}=nothing, tol::Real=1e-6, alpha::Real=0.1,
                          maxstep::Integer=50, n::Integer=_term_period(ψ, U))
    prep = _projection_setup(ψ, χ; cutoff, terms, target, tol, alpha, maxstep, n)
    _warn_scarfinder_nstep(nstep)
    evolve!(ψ, _evolution(ψ, U), nstep; maxdim, cutoff)
    return _project!(ψ, χ, prep; cutoff, target, tol, alpha, maxstep)
end

scarfinder_step!(ψ::iMPS, U, h::AbstractMatrix, χ::Integer; kwargs...) =
    scarfinder_step!(ψ, U, χ; terms=_uniform_terms(ψ, h), n=ψ.n, kwargs...)

scarfinder_step!(ψ::iMPS, h::AbstractMatrix, dt::Real, χ::Integer; trotter=nothing, layers=nothing,
                 kwargs...) =
    scarfinder_step!(ψ, _hamiltonian_evolution(ψ, h, dt; trotter, layers), h, χ; kwargs...)

"""
    scarfinder!(ψ, U, χ, N; refine=true, refine_step=100, refine_U=U, history=nothing, observe=nothing, kwargs...)
    scarfinder!(ψ, U, h, χ, N; …)
    scarfinder!(ψ, h, dt, χ, N; refine_dt=dt/10, trotter=nothing, layers=nothing, …)

Run `N` [`scarfinder_step!`](@ref) iterations; other keywords go to the step.
With `refine=true`, [`minimize_on_trajectory!`](@ref) can then replace `ψ` with
the state of lowest maximal bond entropy over `refine_step` further steps with
`refine_U` (for the `(h, dt)` form: time step `refine_dt`). Use `refine=false`
whenever the caller selects its own representative.

`observe(ψ, k, r)` receives `k = 0` and `r = nothing` at the initial state, then
each iteration's candidate and result; `ψ` is the live state, copy it to keep
it. `history` collects the `N` results. Refinement contributes to neither.
Returns `ψ`.
"""
function scarfinder!(ψ::iMPS, U, χ::Integer, N::Integer; refine::Bool=true,
                     refine_step::Integer=100, refine_U=U,
                     history::Union{Nothing,AbstractVector}=nothing, observe=nothing, kwargs...)
    N >= 0 || throw(ArgumentError("N must be nonnegative"))
    (!refine || refine_step >= 0) || throw(ArgumentError("refine_step must be nonnegative"))
    isnothing(observe) || observe(ψ, 0, nothing)
    for k in 1:N
        r = scarfinder_step!(ψ, U, χ; kwargs...)
        isnothing(history) || push!(history, r)
        isnothing(observe) || observe(ψ, k, r)
    end
    if refine
        step! = state -> scarfinder_step!(state, refine_U, χ; kwargs...)
        minimize_on_trajectory!(_refinement_objective, step!, ψ, refine_step)
    end
    return ψ
end

scarfinder!(ψ::iMPS, U, h::AbstractMatrix, χ::Integer, N::Integer; kwargs...) =
    scarfinder!(ψ, U, χ, N; terms=_uniform_terms(ψ, h), n=ψ.n, kwargs...)

function scarfinder!(ψ::iMPS, h::AbstractMatrix, dt::Real, χ::Integer, N::Integer;
                     refine::Bool=true, refine_dt::Real=dt / 10, trotter=nothing, layers=nothing,
                     kwargs...)
    U = _hamiltonian_evolution(ψ, h, dt; trotter, layers)
    refine_U = refine ? _hamiltonian_evolution(ψ, h, refine_dt; trotter, layers) : U
    return scarfinder!(ψ, U, h, χ, N; refine, refine_U, kwargs...)
end

#---------------------------------------------------------------------------------------------------
# Deprecated
#---------------------------------------------------------------------------------------------------
function truncate_unitcell!(ψ::iMPS, χ::Integer; cutoff::Real=SVDTOL, recanonicalize::Bool=false)
    Base.depwarn("`truncate_unitcell!(ψ, χ)` is deprecated, use `truncate!(ψ, χ)`", :truncate_unitcell!)
    return truncate!(ψ, χ; cutoff)
end

function fix_energy!(ψ::iMPS, h::AbstractMatrix, χ::Integer; target::Real, tol::Real=1e-6,
                     α::Real=0.1, maxstep::Integer=50, kwargs...)
    Base.depwarn("`fix_energy!(ψ, h, χ)` is deprecated, use `energy_fix!(ψ, terms, χ)`", :fix_energy!)
    return energy_fix!(ψ, _uniform_terms(ψ, h), χ; target, tol, alpha=α, maxstep)
end
