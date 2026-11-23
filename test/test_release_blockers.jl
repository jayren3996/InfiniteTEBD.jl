using Test
using LinearAlgebra
using Random
using InfiniteTEBD

# Regressions for the v0.3.0-rc1 review: ScarFinder entry points, structure/tensor
# consistency of the fast path, annihilation versus small finite weights, retry safety.
module RBPXP
using LinearAlgebra
using InfiniteTEBD
const X = ComplexF64[0 1; 1 0]
const P = ComplexF64[1 0; 0 0]
const N = ComplexF64[0 0; 0 1]
const I2 = Matrix{ComplexF64}(I, 2, 2)
const P3 = Diagonal(ComplexF64[(k & (k >> 1)) == 0 for k in 0:7])
h(c) = kron(P, X, P) + c * kron(I2, N, I2)
layers() = [[(h(1.2), 2, 4)], [(h(0.6), 1, 3)]]
terms() = reduce(vcat, layers())
site(w) = iMPO([reshape(ComplexF64.(w), 1, size(w)..., 1)])
occ(ψ, j) = real(expect(ψ, N, j, j))
seed(φ) = apply!(product_iMPS(ComplexF64, [[cos(0.6), sin(0.6) * cis(t)] for t in (φ, π - φ)]),
                 iMPO(Blockade(), 2))
end
using .RBPXP: RBPXP

const _U = trotter_mpos(RBPXP.layers(), 0.05; n=2, gate_map=G -> RBPXP.P3 * G)

@testset "scarfinder! with MPOs matches an explicit scarfinder_step! loop" begin
    e = energy_density(RBPXP.seed(0.45), RBPXP.terms())
    options = [(; nstep=6, maxdim=12),
               (; nstep=6, maxdim=12, terms=RBPXP.terms()),
               (; nstep=6, maxdim=12, terms=RBPXP.terms(), target=e, tol=1e-10)]
    for kw in options
        a, b = RBPXP.seed(0.45), RBPXP.seed(0.45)
        hist = Any[]
        scarfinder!(a, _U, 2, 4; refine=false, history=hist, kw...)
        loop = [scarfinder_step!(b, _U, 2; kw...) for _ in 1:4]
        @test abs(per_site_fidelity(a, b) - 1) < 1e-14
        @test all(abs(RBPXP.occ(a, j) - RBPXP.occ(b, j)) < 1e-13 for j in 1:2)
        @test abs(energy_density(a, RBPXP.terms()) - energy_density(b, RBPXP.terms())) < 1e-13
        @test length(hist) == 4 && [r.loss for r in hist] == [r.loss for r in loop]
        @test length.(a.λ) == length.(b.λ) == [2, 2]
    end
    # refinement runs the same step
    ψ = RBPXP.seed(0.45)
    @test scarfinder!(ψ, _U, 2, 2; refine=true, refine_step=3, nstep=4, maxdim=12) === ψ
    # a matrix density h stands for the terms (h, i, i + 1) on every site of the cell
    hX = kron(RBPXP.X, RBPXP.X)
    a, b = RBPXP.seed(0.45), RBPXP.seed(0.45)
    t = energy_density(a, [(hX, 1, 2), (hX, 2, 3)]; n=2) + 0.01
    ra = scarfinder_step!(a, _U, hX, 2; nstep=4, maxdim=12, target=t)
    rb = scarfinder_step!(b, _U, 2; nstep=4, maxdim=12, target=t, terms=[(hX, 1, 2), (hX, 2, 3)], n=2)
    @test ra == rb && a.Γ == b.Γ
end

@testset "fast path describes the stored operator" begin
    zero2 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    same(W, ψ) = begin
        a, b = deepcopy(ψ), deepcopy(ψ)
        sa = apply!(a, W; return_stats=true)[2]
        sb = apply!(b, W; return_stats=true, structured=false)[2]
        (sa.path, abs(per_site_fidelity(a, b) - 1), maximum(abs(RBPXP.occ(a, j) - RBPXP.occ(b, j)) for j in 1:2))
    end
    # a coarse construction cutoff changes the stored operator; the structure follows it
    for cutoff in (0.0, 1e-14, 0.25, 0.9)
        W = layer_mpo([(exp(-0.7im * kron(RBPXP.P, RBPXP.X, RBPXP.P)), 1, 3)], 2; cutoff)
        path, δF, δn = same(W, zero2)
        @test δF < 1e-13 && δn < 1e-13
    end
    W = layer_mpo([(exp(-0.7im * kron(RBPXP.P, RBPXP.X, RBPXP.P)), 1, 3)], 2; cutoff=0.25)
    @test bonddims(W) == [1, 1]
    @test abs(RBPXP.occ(apply!(deepcopy(zero2), W), 2) - 0.0265276190940486) < 1e-12
    # in-place edit through W[i]: the fast path is disabled and the edit takes effect
    W = layer_mpo([(kron(RBPXP.I2, RBPXP.I2, RBPXP.I2), 1, 3)], 2)
    @test !isnothing(W.structure)
    W[2][:, :, :, :] .= reshape(RBPXP.X, 1, 2, 2, 1) .* W[2][1, 1, 1, 1]
    path, δF, δn = same(W, zero2)
    @test path === :generic && δF < 1e-14 && δn < 1e-14
    @test abs(RBPXP.occ(apply!(deepcopy(zero2), W), 2) - 1) < 1e-14
    @test isnothing(InfiniteTEBD._valid_structure(copy(W)))
    # replacing a tensor object is caught too
    W2 = copy(_U[1])
    @test !isnothing(InfiniteTEBD._valid_structure(W2))
    W2.W[1] = 2 .* W2.W[1]
    @test isnothing(InfiniteTEBD._valid_structure(W2))
    @test isnothing(InfiniteTEBD._merged_boundary(W2, _U[1]).structure)
    # ordinary PXP layers keep the fast path, including merged half steps
    _, st = evolve!(RBPXP.seed(0.45), _U, 5; maxdim=12, return_stats=true)
    @test st.num_structured == length(st.updates) == 11
end

@testset "annihilation versus small finite weights" begin
    # identity on long cells (the (√2)ⁿ bound used to reject n ≥ 78)
    for n in (2, 76, 78, 80, 120)
        ψ = product_iMPS(ComplexF64, [[1, 0] for _ in 1:n])
        @test abs(inner_product(apply!(deepcopy(ψ), RBPXP.site(RBPXP.I2))) - 1) < 1e-12
        @test constraint_leakage(ψ, RBPXP.site(RBPXP.I2)) == 0
    end
    # a rare but finite weight under a 0/1 projector, for any cell length
    for n in (1, 2, 8)
        ψ = product_iMPS(ComplexF64, [[1e-6, sqrt(1 - 1e-12)] for _ in 1:n])
        @test abs(constraint_leakage(ψ, RBPXP.site(RBPXP.P)) - 13.815510557964274) < 1e-9
    end
    # true annihilation is still detected (exact zero and a nilpotent cell)
    full = product_iMPS(ComplexF64, [[0, 1], [0, 1]])
    @test_throws MPOAnnihilationError apply!(deepcopy(full), RBPXP.site(RBPXP.P))
    @test_throws MPOAnnihilationError apply!(deepcopy(full), _U[2])
    @test_throws MPOAnnihilationError apply!(deepcopy(full), _U[2]; structured=false)
    @test constraint_leakage(full, iMPO(Blockade(), 2)) == Inf
    @test constraint_leakage(full, Blockade()) == Inf
    tiled = product_iMPS(ComplexF64, repeat([[0, 1], [0, 1]], 54))
    @test constraint_leakage(tiled, Blockade()) == Inf
    # analytic regression: |1⟩ ⊗ (δ|0⟩ + √(1-δ²)|1⟩), δ = 1e-3; P ψ ∝ |1 0 1 0 …⟩,
    # leakage −log(δ²)/4 per site; unchanged up to 120 sites
    δ = 1e-3
    exact = -log(δ^2) / 4
    @test exact ≈ 3.45387763949 atol = 1e-11
    for n in (2, 40, 100, 108, 112, 120)
        s = product_iMPS(ComplexF64, repeat([ComplexF64[0, 1], ComplexF64[δ, sqrt(1 - δ^2)]], n ÷ 2))
        @test abs(constraint_leakage(s, Blockade()) - exact) < 1e-13
    end
end

@testset "scarfinder_step!: retry safety" begin
    # invalid term sizes are rejected before the state changes, with or without target
    for kw in ((;), (; target=0.1))
        ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
        saved = deepcopy(ψ)
        W = RBPXP.site(RBPXP.X)
        @test_throws ArgumentError scarfinder_step!(ψ, [W], 2; nstep=3, maxdim=4,
            terms=[(Matrix(1.0I, 3, 3), 1, 1)], kw...)
        @test ψ.Γ == saved.Γ && ψ.λ == saved.λ
    end
end
