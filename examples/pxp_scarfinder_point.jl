# One grid point of a staggered-PXP ScarFinder phase diagram with the package primitives.
#
#   H = Σ P X P + Σ_A (μ+δ) n + Σ_B (μ−δ) n,   two-site cell, A = site 1.
#
# Per seed: random χ = 2 tensors → energy correction to e →
# 30 coarse iterations (ΔT = 1) → ~400 fine iterations (ΔT = 0.1), recording every returned
# candidate. The representative, orbit classes and scores are the project's decisions; this
# script only shows where they plug in. A release of a chosen state is recorded at the end.
#
#   OPENBLAS_NUM_THREADS=1 julia --project=. examples/pxp_scarfinder_point.jl
#   (SF_SMALL=1 for a quick smoke run)
using InfiniteTEBD, LinearAlgebra, Random, Printf

const SMALL = get(ENV, "SF_SMALL", "0") == "1"
const N = ComplexF64[0 0; 0 1]
const X = ComplexF64[0 1; 1 0]
const P = ComplexF64[1 0; 0 0]
const I2 = Matrix{ComplexF64}(I, 2, 2)
const P3 = Diagonal(ComplexF64[(k & (k >> 1)) == 0 for k in 0:7])   # three-site blockade
fold(G) = P3 * G                                                       # every gate is P3 * U
h(c) = kron(P, X, P) + c * kron(I2, N, I2)

μ, δ, e = 0.9, 0.0, 0.25
point_id = 1                                    # the project's identifier of (μ, δ, e)
layers = [[(h(μ + δ), 2, 4)], [(h(μ - δ), 1, 3)]]     # centres on site 1 (A), then site 2 (B)

U = trotter_mpos(layers, 0.05; n=2, gate_map=fold)

# Each seed's stream depends only on (grid point, seed id), not on scheduling. The
# historical scripts drew complex Gaussian tensors; `rand_iMPS(rng, …)` draws uniform ones.
seed_state(seed_id) = begin
    rng = Xoshiro(hash((point_id, seed_id)))
    iMPS(ComplexF64, [randn(rng, ComplexF64, 2, 2, 2) for _ in 1:2]; renormalize=true)
end

options = (maxdim=16, cutoff=1e-12, terms=layers, target=e, n=2)

function run_seed(seed_id; coarse=30, fine=400)
    ψ = seed_state(seed_id)
    init = energy_fix!(ψ, layers, 2; target=e, n=2)
    # Fine-stage record: the candidate returned by each iteration (after correction).
    times, states, nA, nB, diag = Float64[], Any[], Float64[], Float64[], Any[]
    record = (φ, k, r) -> begin
        k == 0 && return
        push!(times, 0.1k)
        push!(states, deepcopy(φ))         # φ is the live state: copy to keep it
        push!(nA, real(expect(φ, N, 1, 1)))
        push!(nB, real(expect(φ, N, 2, 2)))
        push!(diag, r)
    end
    scarfinder!(ψ, U, 2, coarse; refine=false, nstep=20, options...)
    scarfinder!(ψ, U, 2, fine; refine=false, nstep=2, observe=record, options...)
    unmet_energy = count(r -> !r.converged, diag)
    return (; seed_id, ψ, times, states, nA, nB, diag, init, unmet_energy)
end

nseeds = SMALL ? 2 : 8
fine = SMALL ? 20 : 400
results = [run_seed(id; coarse=SMALL ? 3 : 30, fine) for id in 1:nseeds]
for r in results
    @printf("seed %d  initial correction converged %s  unmet energy %d\n",
            r.seed_id, r.init.converged, r.unmet_energy)
end

# ---- project decisions (not in the package): primitive period, representative, classes ----
# Example placeholder: the n_A maximum over the recorded window of the first seed.
best = results[1]
k = argmax(best.nA)
representative = best.states[k]
# Orbit comparisons use per_site_fidelity (an amplitude: the old squared per-site overlap is
# its square) and symmetry images such as translate(ψ, 1) (which maps δ → −δ) or conj(ψ)
# (time reversal of a real H); whether an image counts as the same class is decided here.

# ---- release under the full H, recorded without changing the integrator ----
obs, rec = release_recorder(representative; dt=0.05, site_ops=(; n=N))
evolve!(deepcopy(representative), U, SMALL ? 40 : 400; maxdim=16, observe=obs, every=5)
window = findall(t -> 10 <= t <= 20, rec.t)
@printf("release: %d samples, final t = %.2f, max bond entropy %.3f, bonds at cap in %d samples\n",
        length(rec.t), rec.t[end], maximum(maximum, rec.entropy), count(>(0), rec.num_at_cap))
isempty(window) || @printf("mean of the larger bond entropy over t ∈ [10, 20]: %.4f\n",
                           sum(maximum(rec.entropy[i]) for i in window) / length(window))
