using Test
using LinearAlgebra
using InfiniteTEBD

# Staggered PXP chain, two-site cell (site 1 = A, site 2 = B):
#   H = ∑_j P_{j-1} X_j P_{j+1} + ∑_{j∈A} (μ+δ) n_j + ∑_{j∈B} (μ-δ) n_j.
# A term centred on c spans (c-1, c, c+1); with supports written i ≤ j the
# A term is (2, 4) and the B term is (1, 3).
module PXPMPO
using LinearAlgebra
using InfiniteTEBD
const X = ComplexF64[0 1; 1 0]
const P = ComplexF64[1 0; 0 0]
const N = ComplexF64[0 0; 0 1]
const I2 = Matrix{ComplexF64}(I, 2, 2)
const P3 = Diagonal(ComplexF64[!(((s >> 2) & 1 == 1 && (s >> 1) & 1 == 1) ||
                                 ((s >> 1) & 1 == 1 && s & 1 == 1)) for s in 0:7])
h(c) = kron(P, X, P) + c * kron(I2, N, I2)
const μ, δ = 0.9, 0.3
layers2() = [[(h(μ + δ), 2, 4)], [(h(μ - δ), 1, 3)]]
layers4_gates() = [[(h(μ + δ), 2, 4), (h(μ + δ), 4, 2)], [(h(μ - δ), 1, 3), (h(μ - δ), 3, 1)]]
layers4_mpo() = [[(h(μ + δ), 2, 4), (h(μ + δ), 4, 6)], [(h(μ - δ), 1, 3), (h(μ - δ), 3, 5)]]
blockade(n) = layer_mpo([(I - kron(N, N), k, k + 1) for k in 1:n], n)
end
using .PXPMPO: PXPMPO

_tile(ψ, m) = iMPS(repeat([copy(A) for A in ψ.Γ], m), repeat([copy(l) for l in ψ.λ], m), ψ.n * m)

function _projected_product(θ=0.35)
    ψ = product_iMPS(ComplexF64, [[cos(θ), sin(θ)], [cos(θ), sin(θ)]])
    apply!(ψ, PXPMPO.blockade(2))
    return ψ
end

_forbidden(ψ) = maximum(expect(ψ, kron(PXPMPO.N, PXPMPO.N), j, j + 1) for j in 1:ψ.n)

@testset "PXP layer MPOs have bond dimension 2" begin
    for dt in (0.005, 0.1), c in (0.5, 1.0), fold in (false, true)
        map = fold ? (G -> PXPMPO.P3 * G) : identity
        U = trotter_mpos(PXPMPO.layers2(), dt; n=2, gate_map=map)
        @test all(W -> bonddims(W) == [2, 2], U)
        for (i, j) in ((2, 4), (1, 3))
            G = map(exp(-1im * dt * PXPMPO.h(c)))
            @test bonddims(layer_mpo([(G, i, j)], 2)) == [2, 2]
        end
    end
end

@testset "projector MPO gives a normalized blockaded state" begin
    ψ = _projected_product()
    @test all(l -> norm(l) ≈ 1, ψ.λ)
    @test inner_product(ψ) ≈ 1 atol=1e-12
    @test _forbidden(ψ) < 1e-14
    @test length.(ψ.λ) == [2, 2]
end

@testset "two-site MPO path vs four-site gate path" begin
    dt, steps, maxdim = 0.005, 200, 64
    terms = reduce(vcat, PXPMPO.layers2())
    ψ2 = _projected_product()
    ψ4 = _tile(ψ2, 2)
    canonical!(ψ4)
    evolve!(ψ2, trotter_mpos(PXPMPO.layers2(), dt; n=2), steps; maxdim, cutoff=1e-14)
    evolve!(ψ4, PXPMPO.layers4_gates(), dt, steps; maxdim, cutoff=1e-14)
    canonical!(ψ4; maxdim, cutoff=1e-14)
    for (a, b) in ((1, 1), (2, 2))
        @test abs(expect(ψ2, PXPMPO.N, a, a) - expect(ψ4, PXPMPO.N, b, b)) < 1e-9
        @test abs(expect(ψ2, PXPMPO.N, a, a) - expect(ψ4, PXPMPO.N, b + 2, b + 2)) < 1e-9
    end
    @test abs(energy_density(ψ2, terms) - energy_density(ψ4, terms; n=2)) < 1e-9
end

@testset "real-time energy drift stays at the Trotter level" begin
    terms = reduce(vcat, PXPMPO.layers2())
    drift = map((0.05, 0.025)) do dt
        ψ = _projected_product()
        e0 = energy_density(ψ, terms)
        U = trotter_mpos(PXPMPO.layers2(), dt; n=2)
        worst = 0.0
        for _ in 1:round(Int, 2 / dt / 10)
            evolve!(ψ, U, 10; maxdim=128, cutoff=1e-14)
            worst = max(worst, abs(energy_density(ψ, terms) - e0))
        end
        worst
    end
    @test drift[1] < 0.2 * 0.05^2
    @test 3 < drift[1] / drift[2] < 5    # second order: ∝ dt²
end

@testset "P3-folded gates keep the blockade" begin
    ψ = _projected_product()
    U = trotter_mpos(PXPMPO.layers2(), 0.05; n=2, gate_map=G -> PXPMPO.P3 * G)
    for _ in 1:10
        evolve!(ψ, U, 4; maxdim=32, cutoff=1e-12)
        @test _forbidden(ψ) < 1e-12
    end
end

@testset "imaginary-time layers lower the energy monotonically" begin
    terms = reduce(vcat, PXPMPO.layers2())
    ψ = _projected_product()
    U = trotter_mpos(PXPMPO.layers2(), 0.02; evolution=:imaginary, n=2)
    energies = [energy_density(ψ, terms)]
    for _ in 1:40
        evolve!(ψ, U, 1; maxdim=32, cutoff=1e-12)
        push!(energies, energy_density(ψ, terms))
    end
    @test all(diff(energies) .<= 1e-12)
    @test energies[end] < energies[1] - 0.1
end

@testset "energy_fix! with site-dependent terms" begin
    terms = reduce(vcat, PXPMPO.layers2())
    ψ = _projected_product()
    target = energy_density(ψ, terms) - 0.05
    r = energy_fix!(ψ, terms, 4; target, tol=1e-8, maxstep=100)
    @test r.converged
    @test abs(energy_density(ψ, terms) - target) <= 1e-8
    @test r.residual ≈ energy_density(ψ, terms) - target
    # non-convergence is reported, not only warned about
    φ = _projected_product()
    r = energy_fix!(φ, PXPMPO.layers2(), 4; target=-100.0, maxstep=3)
    @test !r.converged && r.steps == 3 && r.residual > 1
end

@testset "two-site symmetry: no cell doubling" begin
    dt, χ = 0.05, 2
    terms2 = reduce(vcat, PXPMPO.layers2())
    terms4 = reduce(vcat, PXPMPO.layers4_mpo())
    U2 = trotter_mpos(PXPMPO.layers2(), dt; n=2)
    U4 = trotter_mpos(PXPMPO.layers4_mpo(), dt; n=4)
    ψ2 = _projected_product()
    e = energy_density(ψ2, terms2)
    ψ4 = _tile(ψ2, 2)
    canonical!(ψ4)
    losses = Float64[]
    for _ in 1:15
        r2 = scarfinder_step!(ψ2, U2, χ; nstep=10, maxdim=16, terms=terms2, target=e, tol=1e-10)
        r4 = scarfinder_step!(ψ4, U4, χ; nstep=10, maxdim=16, terms=terms4, target=e, tol=1e-10)
        push!(losses, r2.loss)
        @test r2.converged && r4.converged
        @test abs(r2.loss - r4.loss) < 1e-10
    end
    @test all(isfinite, losses) && all(>=(0), losses)
    @test abs(per_site_fidelity(_tile(ψ2, 2), ψ4) - 1) < 1e-10
    @test ψ4.λ[1] ≈ ψ4.λ[3] atol=1e-10
    @test ψ4.λ[2] ≈ ψ4.λ[4] atol=1e-10
    @test abs(energy_density(ψ2, terms2) - e) < 1e-9
end

@testset "constraint_leakage matches the hard-core transfer matrix" begin
    for θ in (0.2, 0.6)
        c2, s2 = cos(θ)^2, sin(θ)^2
        ψ = product_iMPS(ComplexF64, [[cos(θ), sin(θ)], [cos(θ), sin(θ)]])
        # ⟨ψ|∏(1 − n n)|ψ⟩ per site = largest eigenvalue of [c² s²; c² 0]
        λ = (c2 + sqrt(c2^2 + 4c2 * s2)) / 2
        @test constraint_leakage(ψ, PXPMPO.blockade(2)) ≈ -log(λ) / 2 rtol=1e-10
        @test constraint_leakage(_projected_product(θ), PXPMPO.blockade(2)) < 1e-14
    end
end

@testset "ScarFinder diagnostics" begin
    terms = reduce(vcat, PXPMPO.layers2())
    U = trotter_mpos(PXPMPO.layers2(), 0.05; n=2, gate_map=G -> PXPMPO.P3 * G)
    run(maxdim) = begin
        ψ = _projected_product()
        e = energy_density(ψ, terms)
        scarfinder_step!(ψ, U, 2; nstep=10, maxdim, terms, target=e, tol=1e-10,
                         constraint=PXPMPO.blockade(2))
    end
    # maxdim = χ: the window already projects, so the loss is ≈ 0 and unresolved
    r2 = run(2)
    @test r2.loss < 1e-12
    @test r2.evolve_saturated > 0
    @test r2.evolve_discarded_sum > 1e-9 > r2.loss
    # a resolved window: accumulated discarded weight far below the loss
    r16 = run(16)
    @test r16.evolve_discarded_sum < 1e-6 * r16.loss
    @test r16.evolve_max_discarded <= r16.evolve_discarded_sum
    # P3-folded evolution keeps the blockade; the projection leaks; energy fixing does not repair it
    @test r16.leakage_evolved < 1e-20
    @test r16.leakage_projected > 1e-14
    @test r16.leakage_fixed > 1e-14
    @test r16.converged && r16.energy_fix_steps > 0
    # positional destructuring of the first fields is unchanged
    loss, err = run(16)
    @test loss == r16.loss
end
