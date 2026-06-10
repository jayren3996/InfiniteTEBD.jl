using Test
using InfiniteTEBD
using LinearAlgebra

# Regression tests for the 2026-06-09 systematic review fixes
# (docs/superpowers/2026-06-09-review-findings.md). Each testset pins one
# previously-confirmed bug or contract.

const _RF_Z = [1.0 0.0; 0.0 -1.0]
const _RF_X = [0.0 1.0; 1.0 0.0]
const _RF_I2 = Matrix{Float64}(I, 2, 2)

@testset "operator convention: leftmost kron factor acts on leftmost site" begin
    # |0⟩ on site 1, |1⟩ on site 2
    psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    # expect must place the first kron factor on site i
    @test expect(psi, kron(_RF_Z, _RF_I2), 1, 2) ≈ 1.0 atol = 1e-10
    @test expect(psi, kron(_RF_I2, _RF_Z), 1, 2) ≈ -1.0 atol = 1e-10

    # applygate! must agree with expect: kron(X, I) on (1,2) flips site 1
    psi2 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    applygate!(psi2, kron(_RF_X, _RF_I2), 1, 2)
    @test expect(psi2, _RF_Z, 1, 1) ≈ -1.0 atol = 1e-8
    @test expect(psi2, _RF_Z, 2, 2) ≈ 1.0 atol = 1e-8

    # three-site support: kron(Z, I, I) on (1,3) of |011⟩ measures site 1
    psi3 = product_iMPS(ComplexF64, [[1, 0], [0, 1], [0, 1]])
    @test expect(psi3, kron(_RF_Z, _RF_I2, _RF_I2), 1, 3) ≈ 1.0 atol = 1e-10
    @test expect(psi3, kron(_RF_I2, _RF_I2, _RF_Z), 1, 3) ≈ -1.0 atol = 1e-10

    # convert_operator reverses site order and is an involution
    G = kron(_RF_Z, _RF_X)
    @test InfiniteTEBD.convert_operator(G, 2, 2) == kron(_RF_X, _RF_Z)
    @test InfiniteTEBD.convert_operator(InfiniteTEBD.convert_operator(G, 2, 2), 2, 2) == G
end

@testset "expect wraps site indices periodically" begin
    psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    # support (n, n+1) crosses the unit-cell seam; previously a BoundsError
    @test expect(psi, kron(_RF_Z, _RF_Z), 2, 3) ≈ -1.0 atol = 1e-10
    @test expect(psi, _RF_Z, 3, 3) ≈ 1.0 atol = 1e-10  # site 3 ≡ site 1
end

@testset "complex gate on real-eltype state throws a descriptive error" begin
    h = kron(_RF_Z, _RF_Z) .+ 0.5 .* (kron(_RF_X, _RF_I2) .+ kron(_RF_I2, _RF_X))
    layers = [[(h, 1, 2)], [(h, 2, 1)]]

    # the all-defaults first-user path: untyped rand_iMPS is Float64
    psi = rand_iMPS(2, 2, 4)
    err = try
        evolve!(psi, layers, 0.1, 1)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("mps_promote_type", err.msg)

    # the state must be untouched by the failed call
    @test eltype(psi) === Float64

    # direct applygate! path throws the same error before mutating
    psi2 = rand_iMPS(2, 2, 4)
    G = exp(-0.1im .* Matrix{ComplexF64}(kron(_RF_X, _RF_X)))
    before = copy(psi2.Γ[1])
    @test_throws ArgumentError applygate!(psi2, G, 1, 2)
    @test psi2.Γ[1] == before

    # the documented remedy works end to end
    psic = mps_promote_type(ComplexF64, rand_iMPS(2, 2, 4))
    evolve!(psic, layers, 0.1, 2)
    @test abs(inner_product(psic) - 1) < 1e-6

    # real gates on real states still work (imaginary time)
    psir = rand_iMPS(2, 2, 4)
    evolve!(psir, layers, 0.1, 1; evolution=:imaginary, recanonicalize=true)
    @test eltype(psir) === Float64
end

@testset "canonical!(renormalize=false) keeps the Schmidt-spectrum convention" begin
    for n in (1, 2)
        psi = rand_iMPS(ComplexF64, n, 2, 4)   # canonical and normalized
        lam_before = [copy(l) for l in psi.λ]
        canonical!(psi; renormalize=false)
        for i in 1:n
            # previously came out scaled by exactly 1/√D
            @test norm(psi.λ[i]) ≈ 1.0 atol = 1e-8
            @test psi.λ[i] ≈ lam_before[i] atol = 1e-6
        end
        # expectation values must not be rescaled by the gauge artifact
        @test abs(expect(psi, _RF_I2, 1, 1) - 1.0) < 1e-8
    end
end

@testset "_truncate_unitcell! n=1 stores B, not λ·B" begin
    psi = rand_iMPS(ComplexF64, 1, 2, 4)
    z_before = expect(psi, _RF_Z, 1, 1)
    InfiniteTEBD._truncate_unitcell!(psi, 4)   # unchanged χ: must be a no-op
    @test expect(psi, _RF_Z, 1, 1) ≈ z_before atol = 1e-6

    # stored tensor must satisfy the right-canonical condition Σ_s B_s B_s† = I
    B = psi.Γ[1]
    acc = zeros(ComplexF64, size(B, 1), size(B, 1))
    for s in 1:size(B, 2)
        Bs = B[:, s, :]
        acc .+= Bs * Bs'
    end
    @test norm(acc - I) < 1e-8
end

@testset "_evolve_uniform! takes the slow path on non-uniform bond dims" begin
    psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    applygate!(psi, exp(-0.3im .* Matrix(kron(_RF_X, _RF_X))), 1, 2; maxdim=4)
    @test length(psi.λ[1]) != length(psi.λ[2])  # the crashing precondition
    # previously threw DimensionMismatch from isapprox on unequal lengths
    InfiniteTEBD._evolve_uniform!(psi, exp(-0.1im .* _RF_X); span=1)
    @test abs(inner_product(psi) - 1) < 1e-6
end

@testset "single-layer :second schedule applies one stage per macro step" begin
    # previously all steps merged into one stage [(1, steps)], so evolve!
    # built exp(-i·steps·dt·h) and skipped every intermediate truncation
    @test InfiniteTEBD._trotter_stage_schedule(1, :second, 3) ==
          [(1, 1.0), (1, 1.0), (1, 1.0)]
    # multi-layer Strang boundary merging is unchanged
    @test InfiniteTEBD._trotter_stage_schedule(2, :second, 2) ==
          [(1, 0.5), (2, 1.0), (1, 1.0), (2, 1.0), (1, 0.5)]

    # behavioral check: single-layer evolution with non-commuting terms in
    # one layer (violating the contract) must still step-and-truncate, which
    # for this small case matches the two-layer reference closely
    h = kron(_RF_Z, _RF_Z) .+ 0.3 .* (kron(_RF_X, _RF_I2) .+ kron(_RF_I2, _RF_X))
    psi_a = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    evolve!(psi_a, [[(h, 1, 2), (h, 2, 1)]], 0.05, 20; maxdim=8)
    psi_b = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    evolve!(psi_b, [[(h, 1, 2)], [(h, 2, 1)]], 0.05, 20; maxdim=8)
    @test expect(psi_a, _RF_Z, 1, 1) ≈ expect(psi_b, _RF_Z, 1, 1) atol = 0.05
end

@testset "iterative svd_trim works for maxdim > 30" begin
    A = randn(120, 120)
    U, S, V = InfiniteTEBD.svd_trim(A; maxdim=50, use_iterative=true)
    @test length(S) == 50
    @test S ≈ svd(A).S[1:50] rtol = 1e-8
    @test norm(U * Diagonal(S) * V - A) <= norm(A)  # sane factorization
end

@testset "single-site renormalize skips unitary gates, stays exact otherwise" begin
    # unitary gates: norm preserved without the per-gate global eigensolve
    psi = rand_iMPS(ComplexF64, 2, 2, 8)
    G = exp(-0.05im .* _RF_X)
    for _ in 1:25
        applygate!(psi, G, 1, 1; renormalize=true)
        applygate!(psi, G, 2, 2; renormalize=true)
    end
    @test abs(inner_product(psi) - 1) < 1e-10

    # non-unitary gates keep the exact global renormalization
    psi2 = rand_iMPS(ComplexF64, 2, 2, 8)
    Gn = exp(-0.05 .* Matrix(_RF_X .+ 0.3 .* _RF_Z))
    for _ in 1:50
        applygate!(psi2, Gn, 1, 1; renormalize=true)
        applygate!(psi2, Gn, 2, 2; renormalize=true)
    end
    @test abs(inner_product(psi2) - 1) < 1e-8
end

@testset "conj copies storage for real element types" begin
    psi = rand_iMPS(2, 2, 3)
    psic = conj(psi)
    orig = psi.Γ[1][1, 1, 1]
    psic.Γ[1][1, 1, 1] = orig + 1
    @test psi.Γ[1][1, 1, 1] == orig
end
