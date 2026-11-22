#---------------------------------------------------------------------------------------------------
# ScarFinder with layer MPOs: one step, the multi-iteration driver, and a prepared plan
#---------------------------------------------------------------------------------------------------
export ScarFinderPlan

"""
    ScarFinderPlan(layers, dt; n, gate_map=identity, constraint=nothing, terms=nothing, trotter=:second, d=nothing, cutoff=MPO_CUTOFF)

Read-only preparation of a layer-MPO ScarFinder for one Hamiltonian, shared by
any number of seeds (and target energies):

- the layers, validated once (supports, local dimension, at most one gate per
  cell bond, commutation with periodic translates, `gate_map(I)` commuting with
  every term, commutation of the mapped gates);
- the real-time MPOs `trotter_mpos(layers, dt; n, gate_map, trotter)` and their
  merged step boundary;
- the spectral bounds of the terms for the imaginary-time correction, which
  uses the **same** `gate_map` (so `gate_map = G -> P3 * G` folds the projector
  on both time paths);
- for `constraint` (a [`Blockade`](@ref) or a projector [`iMPO`](@ref)), the
  projector MPO used by the optional post-SVD repair.

`n` (the cell of the layers) must be given: supports such as `(2, 4)` do not
determine it. The energy is measured with the terms of `layers`; if `terms` is
given it must list the same terms (it is checked, not used instead).

Trial imaginary-time gates depend on `dτ` and are built per trial exactly as
[`energy_fix!`](@ref) builds them, so a plan changes no number: a step with a
plan equals the same step through `scarfinder_step!(ψ, plan.mpos, χ; terms=layers, n, gate_map, constraint, …)`.
Each task owns its state and RNG; the plan holds no mutable workspace (do not
modify its arrays).

Use with [`scarfinder_step!`](@ref)`(ψ, plan, χ; …)` and
[`scarfinder!`](@ref)`(ψ, plan, χ, N; …)`.
"""
struct ScarFinderPlan
    layers::Vector{Vector{Any}}
    dt::Float64
    n::Int
    d::Int
    trotter::Symbol
    gate_map::Any
    mpos::Vector
    merged::Union{Nothing,iMPO}
    prep::_ImagPrep
    constraint::Any
    projector::Union{Nothing,iMPO}
end

function ScarFinderPlan(layers, dt::Real; n::Integer, gate_map=identity, constraint=nothing,
                        terms=nothing, trotter::Symbol=:second, d::Union{Nothing,Integer}=nothing,
                        cutoff::Real=MPO_CUTOFF)
    ls = [collect(Any, l) for l in layers]
    isempty(ls) && throw(ArgumentError("ScarFinderPlan needs at least one layer"))
    all(l -> !isempty(l) && first(l) isa Tuple, ls) ||
        throw(ArgumentError("layers must be a list of layers of (h, i, j) terms"))
    dd = _uniform_local_dim(reduce(vcat, ls), n, d)
    prep = _prepare_imaginary(ls; n, d=dd, gate_map)
    mpos = trotter_mpos(ls, dt; trotter, n, cutoff, check=true, d=dd, gate_map)
    merged = length(mpos) >= 2 ? _merged_boundary(mpos[end], mpos[1]) : nothing
    if !isnothing(terms)
        given = [(Matrix(h), Int(i), Int(j)) for (h, i, j) in collect(terms)]
        mine = [(Matrix(h), Int(i), Int(j)) for (h, i, j) in prep.terms]
        (length(given) == length(mine) && all(t -> t in mine, given) && all(t -> t in given, mine)) ||
            throw(ArgumentError("`terms` differ from the terms of `layers`; the plan measures " *
                                "the energy with the layers' terms"))
    end
    projector = _constraint_projector(constraint, n, dd)
    return ScarFinderPlan(ls, Float64(dt), Int(n), dd, trotter, gate_map, mpos, merged, prep,
                          constraint, projector)
end

function Base.show(io::IO, p::ScarFinderPlan)
    print(io, "ScarFinderPlan(", length(p.layers), " layers, dt=", p.dt, ", n=", p.n, ", d=", p.d,
          ", ", length(p.mpos), " MPOs", isnothing(p.merged) ? "" : " + merged boundary",
          p.gate_map === identity ? "" : ", gate_map", isnothing(p.constraint) ? "" : ", constraint", ")")
end

_constraint_projector(::Nothing, n, d) = nothing
_constraint_projector(b::Blockade, n, d) = iMPO(b, n; d)
_constraint_projector(P::iMPO, n, d) = P
_constraint_projector(c, n, d) = throw(ArgumentError(
    "constraint must be a projector iMPO or a Blockade, got $(typeof(c))"))

# Options shared by both entry points, validated before the state is touched.
function _validate_step_options(ψ, χ; maxdim, cutoff, nstep, evolve_truncerr, target, terms_given,
                                constraint, projected_energy, repair, max_project, leak_tol)
    _validate_mpo_truncation(χ, cutoff, 0.0)
    _validate_mpo_truncation(maxdim, cutoff, evolve_truncerr)
    nstep >= 0 || throw(ArgumentError("nstep must be non-negative"))
    !isnothing(target) && !terms_given && throw(ArgumentError("supply `terms` when setting `target`"))
    isnothing(constraint) || _check_constraint(ψ, constraint)
    if projected_energy
        isnothing(constraint) && throw(ArgumentError("projected_energy=true needs a `constraint`"))
        terms_given || throw(ArgumentError("projected_energy=true needs `terms`"))
    end
    if repair
        isnothing(constraint) && throw(ArgumentError("repair=true needs a `constraint`"))
        max_project >= 0 || throw(ArgumentError("max_project must be non-negative"))
        isfinite(leak_tol) && leak_tol >= 0 || throw(ArgumentError("leak_tol must be finite and non-negative"))
    end
    return nothing
end

# One iteration on validated inputs. `merged` is the precomputed boundary (or nothing);
# `prep` the imaginary-time preparation (or nothing without terms).
function _mpo_sf_step!(ψ, mpos, merged, χ, prep, constraint, projector;
                       nstep, maxdim, cutoff, target, tol, alpha, maxstep, structured,
                       evolve_truncerr, projected_energy, repair, max_project, leak_tol,
                       max_rejections=nothing)
    _warn_scarfinder_nstep(nstep, :gate)
    boundary = nstep >= 2 ? merged : nothing
    _, est = _evolve_sequence!(ψ, mpos, boundary, nstep; maxdim, cutoff, truncerr=evolve_truncerr,
                               renormalize=true, return_stats=true, structured)
    leak(state) = isnothing(constraint) ? nothing : constraint_leakage(state, constraint)
    leakage_evolved = leak(ψ)
    before = iMPS(copy(ψ.Γ), copy(ψ.λ), ψ.n)                    # the evolved state b
    if nstep >= 1 && !isempty(mpos)
        # evolve! returns an exactly canonical state: truncate its Schmidt spectra directly
        _truncate_canonical!(ψ; maxdim=χ, cutoff, truncerr=0.0, renormalize=true, recanonicalize=true)
    else
        truncate!(ψ, χ; cutoff)
    end
    loss_svd = -log(min(per_site_fidelity(before, ψ), 1.0))
    leakage_projected = leak(ψ)
    # Optional bounded repair: P, then truncation back to χ, until leak_tol or the budget.
    project_steps, project_status, leakage_repaired = 0, :off, leakage_projected
    if repair
        current = leakage_projected
        project_status = current <= leak_tol ? :clean : :budget
        while current > leak_tol && project_steps < max_project
            try
                apply!(ψ, projector; maxdim=χ, cutoff)
            catch err
                err isa MPOAnnihilationError || rethrow()
                project_status = :annihilated                 # ψ is unchanged (transactional)
                break
            end
            project_steps += 1
            new = leak(ψ)
            if new <= leak_tol
                current, project_status = new, :converged
                break
            elseif !(new < current)
                current, project_status = new, :stalled
                break
            end
            current = new
        end
        leakage_repaired = current
    end
    loss = project_steps == 0 ? loss_svd : -log(min(per_site_fidelity(before, ψ), 1.0))
    energy_error, converged, steps, leakage_fixed = nothing, true, 0, nothing
    energy_initial_error, energy_fix_status, rejected = nothing, :off, 0
    if !isnothing(target)
        res = _energy_fix_core!(ψ, prep, χ; target, tol, alpha, maxstep, cutoff, structured,
                                max_rejections)
        energy_error, converged, steps = res.residual, res.converged, res.steps
        energy_initial_error, energy_fix_status, rejected = res.initial_residual, res.status, res.rejected
        leakage_fixed = leak(ψ)
    end
    energy_projected = energy_projected_error = energy_projected_status = nothing
    if projected_energy
        # An optional diagnostic of a state that is already final: it never throws for a
        # property of the state or a solver failure, it reports them.
        energy_projected, energy_projected_status = try
            projected_energy_density(ψ, constraint, prep.terms; n=prep.n), :ok
        catch err
            err isa MPOAnnihilationError ? (NaN, :annihilated) :       # P ψ = 0: undefined
            err isa CanonicalFormError ? (NaN, :unconverged) : rethrow()
        end
        isnothing(target) || (energy_projected_error = energy_projected - target)
    end
    return (loss=loss, energy_error=energy_error, converged=converged,
            loss_svd=loss_svd,
            evolve_max_discarded=est.max_discarded_weight,
            evolve_discarded_sum=sum((u.max_discarded_weight for u in est.updates); init=0.0),
            evolve_saturated=est.num_saturated,
            evolve_at_cap=est.num_at_cap,
            evolve_structured=est.num_structured,
            leakage_evolved=leakage_evolved, leakage_projected=leakage_projected,
            leakage_repaired=leakage_repaired, leakage_fixed=leakage_fixed,
            project_steps=project_steps, project_status=project_status,
            project_converged=repair ? leakage_repaired <= leak_tol : nothing,
            energy_initial_error=energy_initial_error, energy_fix_status=energy_fix_status,
            energy_fix_steps=steps, energy_fix_rejected=rejected,
            energy_projected=energy_projected, energy_projected_error=energy_projected_error,
            energy_projected_status=energy_projected_status)
end

"""
    scarfinder_step!(ψ, mpos::Vector{<:iMPO}, χ; nstep=10, maxdim=MAXDIM, cutoff=SVDTOL, terms=nothing, target=nothing, tol=1e-6, alpha=0.1, maxstep=50, n=first(mpos).n, gate_map=identity, constraint=nothing, repair=false, max_project=3, leak_tol=1e-10, structured=true, evolve_truncerr=0.0, projected_energy=false)
    scarfinder_step!(ψ, plan::ScarFinderPlan, χ; nstep=10, maxdim=MAXDIM, cutoff=SVDTOL, target=nothing, tol=1e-6, alpha=0.1, maxstep=50, repair=false, max_project=3, leak_tol=1e-10, structured=true, evolve_truncerr=0.0, projected_energy=false)

One ScarFinder iteration with layer MPOs:

1. evolve `nstep` Trotter steps with `mpos` (half steps merged); keep the
   evolved state `b`;
2. truncate every bond to `χ` Schmidt values (`a₀`; the evolved state is
   canonical, so its spectra are cut directly, as [`truncate!`](@ref) would);
3. only with `repair=true` (needs `constraint`): while the leakage exceeds
   `leak_tol` and fewer than `max_project` passes were made, apply the full
   constraint projector and truncate back to `χ` (`a₁, a₂, …`);
4. if `target` is given, correct the energy with imaginary-time layers built from
   `terms` and mapped by `gate_map` ([`energy_fix!`](@ref)), starting from the
   residual of the (repaired) state of step 3.

With a [`ScarFinderPlan`](@ref) the real-time MPOs, the merged boundary, the
validated layers, `gate_map` (the same on both time paths) and the constraint
come from the plan, and the numbers are the same as through the vector method
given the same inputs. With the vector method, `mpos` fixes the real-time folding
and `gate_map` only the imaginary-time correction: pass the same map you gave
`trotter_mpos` (e.g. `G -> P3 * G`); `identity` (the default) leaves the
correction unfolded, as in earlier versions.

`constraint=Blockade()` alone is a **diagnostic** (leakages, optional projected
energy); it changes the state only with `repair=true`.

The repair is a bounded heuristic, not an optimal constrained truncation. A pass
removes the illegal component, but the following truncation can reintroduce
leakage and moves the state; later passes can lower the leakage while slightly
increasing the loss, and passes need not reach `leak_tol` within the budget. Its
outcome is reported, never enforced.

Returns a named tuple. The first three fields are unchanged from earlier
versions, so `loss, err = scarfinder_step!(…)` still works:

- `loss`: `−log F_site(b, a)`, with `a` the candidate **after repair and before
  energy correction** (`a = a₀` without repair). Ranking quantity.
- `energy_error`: `E(ψ) − target` of the returned state after step 4
  (`nothing` without `target`).
- `converged`: whether the energy correction met `tol` (`true` without
  `target`).

Projection and repair:

- `loss_svd`: `−log F_site(b, a₀)`, the loss right after the initial SVD.
- `leakage_evolved`, `leakage_projected`, `leakage_repaired`, `leakage_fixed`:
  per-site leakage ([`constraint_leakage`](@ref)) of `b`, of `a₀`, of the repaired
  `a` (= `leakage_projected` without repair) and after energy correction
  (`nothing` without `constraint`, or `leakage_fixed` without `target`).
- `project_steps`: repair passes made after the initial SVD (0 if `a₀` already met
  `leak_tol`).
- `project_status`: `:off` (no repair), `:clean` (no pass needed), `:converged`,
  `:budget` (`max_project` passes used, `leak_tol` not reached), `:stalled` (a pass
  did not lower the leakage; the state of that pass is kept) or `:annihilated`
  (the projector maps the state to zero; the state before that pass is kept).
- `project_converged`: `leakage_repaired ≤ leak_tol` (`nothing` without repair).

Energy correction:

- `energy_initial_error`: residual at the start of the correction, measured on
  the repaired state.
- `energy_fix_status`: `:off`, `:converged`, `:maxstep`, `:stalled`, `:not_run`
  (see [`energy_fix!`](@ref)); `energy_fix_steps`: trials; `energy_fix_rejected`:
  rejected trials (each restarted from the last accepted state). The opt-in
  `max_rejections` (default off) ends a correction after that many consecutive
  rejections with `:stalled`.
- `energy_projected`, `energy_projected_error`, `energy_projected_status`: with
  `projected_energy=true`, the energy density of `Pψ/‖Pψ‖` for the returned
  candidate, `− target`, and `:ok`/`:annihilated`/`:unconverged` (see
  [`projected_energy_density`](@ref)). `energy_error` refers to the whole state,
  which is what the correction controls.

Evolution window (over its `apply!` calls):

- `evolve_max_discarded` (**maximum** discarded weight over applications and
  bonds), `evolve_discarded_sum` (**sum** of the per-application maxima),
  `evolve_saturated` (bonds whose requested dimension exceeded `maxdim`),
  `evolve_at_cap` (bonds whose retained dimension equals `maxdim`),
  `evolve_structured` (applications that used the controlled-layer update).
  Trust a loss only when `evolve_discarded_sum ≪ loss`.

Options: `structured=false` forces the generic MPO update everywhere;
`evolve_truncerr` (opt-in) lets real-time bonds shrink below `maxdim` when the
Schmidt tail is small.

All static inputs — MPOs against the state, constraint, `target`, `terms` (sizes,
supports, at most one gate per bond and layer, commutation, `gate_map`
compatibility), energy-correction, repair and truncation parameters — are
validated before `ψ` is modified.
"""
function scarfinder_step!(ψ::DenseIMPS, mpos::AbstractVector{<:iMPO}, χ::Integer;
                          nstep::Integer=10, maxdim::Integer=MAXDIM, cutoff::Real=SVDTOL,
                          terms=nothing, target::Union{Nothing,Real}=nothing,
                          tol::Real=1e-6, alpha::Real=0.1, maxstep::Integer=50,
                          n::Integer=isempty(mpos) ? ψ.n : first(mpos).n,
                          gate_map=identity, constraint=nothing, structured::Bool=true,
                          evolve_truncerr::Real=0.0, projected_energy::Bool=false,
                          repair::Bool=false, max_project::Integer=3, leak_tol::Real=1e-10,
                          max_rejections::Union{Nothing,Integer}=nothing)
    _validate_step_options(ψ, χ; maxdim, cutoff, nstep, evolve_truncerr, target,
                           terms_given=!isnothing(terms), constraint, projected_energy, repair,
                           max_project, leak_tol)
    foreach(W -> _check_mpo_state(ψ, W), mpos)
    d = size(ψ.Γ[1], 2)
    prep = isnothing(terms) ? nothing : _prepare_imaginary(terms; n, d, gate_map)
    if !isnothing(prep)
        if isnothing(target)
            ψ.n % n == 0 || throw(ArgumentError("ψ.n = $(ψ.n) must be a multiple of the term period n = $n"))
        else
            _validate_energy_fix_args(ψ, χ, prep; target, tol, alpha, maxstep, cutoff, max_rejections)
        end
    end
    projector = repair ? _constraint_projector(constraint, n, d) : nothing
    merged = length(mpos) >= 2 && nstep >= 2 ? _merged_boundary(mpos[end], mpos[1]) : nothing
    return _mpo_sf_step!(ψ, mpos, merged, χ, prep, constraint, projector;
                         nstep, maxdim, cutoff, target, tol, alpha, maxstep, structured,
                         evolve_truncerr, projected_energy, repair, max_project, leak_tol,
                         max_rejections)
end

function scarfinder_step!(ψ::DenseIMPS, plan::ScarFinderPlan, χ::Integer;
                          nstep::Integer=10, maxdim::Integer=MAXDIM, cutoff::Real=SVDTOL,
                          target::Union{Nothing,Real}=nothing, tol::Real=1e-6, alpha::Real=0.1,
                          maxstep::Integer=50, structured::Bool=true, evolve_truncerr::Real=0.0,
                          projected_energy::Bool=false, repair::Bool=false,
                          max_project::Integer=3, leak_tol::Real=1e-10,
                          max_rejections::Union{Nothing,Integer}=nothing)
    _validate_step_options(ψ, χ; maxdim, cutoff, nstep, evolve_truncerr, target, terms_given=true,
                           constraint=plan.constraint, projected_energy, repair, max_project, leak_tol)
    foreach(W -> _check_mpo_state(ψ, W), plan.mpos)
    if isnothing(target)
        ψ.n % plan.n == 0 || throw(ArgumentError("ψ.n = $(ψ.n) must be a multiple of the plan cell n = $(plan.n)"))
    else
        _validate_energy_fix_args(ψ, χ, plan.prep; target, tol, alpha, maxstep, cutoff, max_rejections)
    end
    return _mpo_sf_step!(ψ, plan.mpos, plan.merged, χ, plan.prep, plan.constraint,
                         repair ? plan.projector : nothing;
                         nstep, maxdim, cutoff, target, tol, alpha, maxstep, structured,
                         evolve_truncerr, projected_energy, repair, max_project, leak_tol,
                         max_rejections)
end

# A matrix density `h` belongs to the gate-list interface; MPO steps take `terms`.
function scarfinder_step!(ψ::DenseIMPS, mpos::AbstractVector{<:iMPO}, h::AbstractMatrix, χ::Integer;
                          kwargs...)
    throw(ArgumentError("with MPOs, pass the Hamiltonian as `terms=[(h, i, j), …]`, " *
                        "not as a matrix argument"))
end

"""
    scarfinder!(ψ, mpos::Vector{<:iMPO}, χ, N; refine=true, refine_step=100, refine_mpos=mpos, history=nothing, observe=nothing, kwargs...)
    scarfinder!(ψ, plan::ScarFinderPlan, χ, N; refine=true, refine_step=100, history=nothing, observe=nothing, kwargs...)

`N` ScarFinder iterations with layer MPOs: exactly `N` calls of
[`scarfinder_step!`](@ref) with the same options (`nstep`, `maxdim`, `terms`,
`target`, `gate_map`, `constraint`, `repair`, `projected_energy`, …), followed,
if `refine=true`, by the minimum-entanglement scan of
[`minimize_on_trajectory!`](@ref) over `refine_step` further steps (with
`refine_mpos`, or the plan), which can **replace** `ψ` by a scanned state. With
`refine=false` the result is identical to an explicit loop over
`scarfinder_step!`; pass `refine=false` whenever the caller selects its own
representative.

Observation: `observe(ψ, k, r)` is called with `k = 0` and `r = nothing` before
the first iteration, then after each main iteration `k = 1, …, N` with the
returned candidate (after repair and energy correction) and that iteration's
named tuple `r`. `ψ` is the live state: the observer must not modify it and must
copy it (`deepcopy(ψ)`) to keep a snapshot. Refinement samples are not observed
and are not in `history`. `history` (a vector) collects the `N` named tuples.
Returns `ψ`.
"""
function scarfinder!(ψ::DenseIMPS, mpos::AbstractVector{<:iMPO}, χ::Integer, N::Integer;
                     refine::Bool=true, refine_step::Integer=100,
                     refine_mpos::AbstractVector{<:iMPO}=mpos,
                     history::Union{Nothing,AbstractVector}=nothing, observe=nothing, kwargs...)
    return _sf_driver!(ψ, st -> scarfinder_step!(st, mpos, χ; kwargs...),
                       st -> scarfinder_step!(st, refine_mpos, χ; kwargs...), N;
                       refine, refine_step, history, observe)
end

function scarfinder!(ψ::DenseIMPS, plan::ScarFinderPlan, χ::Integer, N::Integer;
                     refine::Bool=true, refine_step::Integer=100,
                     history::Union{Nothing,AbstractVector}=nothing, observe=nothing, kwargs...)
    step = st -> scarfinder_step!(st, plan, χ; kwargs...)
    return _sf_driver!(ψ, step, step, N; refine, refine_step, history, observe)
end

function _sf_driver!(ψ, step, refine_step!, N; refine, refine_step, history, observe)
    N >= 0 || throw(ArgumentError("N must be nonnegative"))
    (!refine || refine_step >= 0) || throw(ArgumentError("refine_step must be nonnegative"))
    isnothing(observe) || observe(ψ, 0, nothing)
    for k in 1:N
        r = step(ψ)
        isnothing(history) || push!(history, r)
        isnothing(observe) || observe(ψ, k, r)
    end
    refine && minimize_on_trajectory!(_refinement_objective, refine_step!, ψ, refine_step)
    return ψ
end

function scarfinder!(ψ::DenseIMPS, mpos::AbstractVector{<:iMPO}, h::Union{Nothing,AbstractMatrix},
                     χ::Integer, N::Integer; kwargs...)
    isnothing(h) && return scarfinder!(ψ, mpos, χ, N; kwargs...)
    throw(ArgumentError("with MPOs, pass the Hamiltonian as `terms=[(h, i, j), …]`, " *
                        "not as a matrix argument"))
end

export release_recorder
"""
    observe, record = release_recorder(ψ0; dt, site_ops=(;), fidelity=true)

An observer for the MPO [`evolve!`](@ref) of a release from `ψ0`, and the
record it fills (a named tuple of vectors, one entry per sample):

- `step`, `t = step · dt` (physical time of the sampled state);
- `entropy[k][b]`: entanglement entropy of bond `b` (`ent_S`, natural log);
- `dims[k][b]`: retained bond dimension (`length(λ[b])`);
- `max_discarded`, `num_saturated` (requested dimension exceeded `maxdim`),
  `num_at_cap` (retained dimension equal to `maxdim`), `max_kept_dim`: over the
  applications since the previous sample (zero at `t = 0`);
- `ops[name][k][j]`: `⟨O⟩` on site `j` of the cell for each `name => O` in
  `site_ops` (e.g. `(; n = [0 0; 0 1])` gives `n_A`, `n_B` on a two-site cell);
- `return_fidelity[k]`: `per_site_fidelity(ψ0, ψ(t))` (amplitude per site; the
  squared overlap per site is its square).

It only reads the state. The project decides how to score entropies (maximum or
mean over bonds, time window) and what counts as saturation; the record keeps
the separate measurements "retained dimension reached the cap", "requested
dimension exceeded the cap" and the entropies themselves.

```julia
obs, rec = release_recorder(ψ0; dt=0.05, site_ops=(; n=N))
evolve!(deepcopy(ψ0), U, 400; maxdim=32, observe=obs, every=5)
```
"""
function release_recorder(ψ0::iMPS; dt::Real, site_ops=(;), fidelity::Bool=true)
    ref = deepcopy(ψ0)
    names = collect(keys(site_ops))
    record = (step=Int[], t=Float64[], entropy=Vector{Float64}[], dims=Vector{Int}[],
              max_discarded=Float64[], num_saturated=Int[], num_at_cap=Int[], max_kept_dim=Int[],
              ops=Dict{Symbol,Vector{Vector{Float64}}}(k => Vector{Float64}[] for k in names),
              return_fidelity=Float64[])
    observe = function (φ, k, stats)
        push!(record.step, k)
        push!(record.t, k * dt)
        push!(record.entropy, [ent_S(φ, b) for b in 1:φ.n])
        push!(record.dims, length.(φ.λ))
        push!(record.max_discarded, isnothing(stats) ? 0.0 : stats.max_discarded_weight)
        push!(record.num_saturated, isnothing(stats) ? 0 : stats.num_saturated)
        push!(record.num_at_cap, isnothing(stats) ? 0 : stats.num_at_cap)
        push!(record.max_kept_dim, isnothing(stats) ? maximum(length, φ.λ) : stats.max_kept_dim)
        for name in names
            O = site_ops[name]
            push!(record.ops[name], [real(expect(φ, O, j, j)) for j in 1:φ.n])
        end
        fidelity && push!(record.return_fidelity, per_site_fidelity(ref, φ))
        return nothing
    end
    return observe, record
end
