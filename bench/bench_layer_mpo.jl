# Time per second-order Trotter step for the staggered PXP chain:
#   two-site cell + layer MPOs (apply! + canonical truncation)
#   vs four-site cell + three-site gates (evolve! with trotter layers).
#
# Run: OPENBLAS_NUM_THREADS=1 julia --project=. bench/bench_layer_mpo.jl [chi ...]
# Report only; nothing is asserted.
using InfiniteTEBD, LinearAlgebra, Printf

const X = ComplexF64[0 1; 1 0]
const P = ComplexF64[1 0; 0 0]
const N = ComplexF64[0 0; 0 1]
const I2 = Matrix{ComplexF64}(I, 2, 2)
h(c) = kron(P, X, P) + c * kron(I2, N, I2)
const μ, δ, dt = 0.9, 0.3, 0.05
layers2 = [[(h(μ + δ), 2, 4)], [(h(μ - δ), 1, 3)]]
layers4 = [[(h(μ + δ), 2, 4), (h(μ + δ), 4, 2)], [(h(μ - δ), 1, 3), (h(μ - δ), 3, 1)]]

tile(ψ, m) = iMPS(repeat([copy(A) for A in ψ.Γ], m), repeat([copy(l) for l in ψ.λ], m), ψ.n * m)

# A blockaded state that already saturates χ: evolve a projected product state.
function saturated_state(χ)
    θ = 0.6
    ψ = product_iMPS(ComplexF64, [[cos(θ), sin(θ)], [cos(θ), sin(θ)]])
    apply!(ψ, layer_mpo([(I - kron(N, N), 1, 2), (I - kron(N, N), 2, 3)], 2))
    U = trotter_mpos(layers2, dt; n=2)
    while minimum(length.(ψ.λ)) < χ
        evolve!(ψ, U, 5; maxdim=χ, cutoff=1e-14)
    end
    return ψ
end

# Minimum over `reps` timed runs (the machine may be shared), after one warm-up.
function bench(χ; steps=10, reps=3)
    ψ0 = saturated_state(χ)
    U = trotter_mpos(layers2, dt; n=2)
    φ0 = tile(ψ0, 2)
    evolve!(deepcopy(ψ0), U, 2; maxdim=χ, cutoff=1e-14)
    evolve!(deepcopy(φ0), layers4, dt, 2; maxdim=χ, cutoff=1e-14)
    t_mpo = minimum(@elapsed(evolve!(deepcopy(ψ0), U, steps; maxdim=χ, cutoff=1e-14)) for _ in 1:reps)
    t_gate = minimum(@elapsed(evolve!(deepcopy(φ0), layers4, dt, steps; maxdim=χ, cutoff=1e-14)) for _ in 1:reps)
    return t_mpo / steps, t_gate / steps
end

if abspath(PROGRAM_FILE) == @__FILE__
    chis = isempty(ARGS) ? [16, 32, 64] : parse.(Int, ARGS)
    println("| χ | 2-site MPO path (ms/step) | 4-site gate path (ms/step) | ratio |")
    println("|---|---|---|---|")
    for χ in chis
        a, b = bench(χ)
        @printf("| %d | %.2f | %.2f | %.2f |\n", χ, 1e3a, 1e3b, a / b)
    end
end
