# Tests for src/ITensorsInterop.jl — the imps2mps boundary contract.
#
# Contract under test (see the imps2mps docstring):
# - the window length L must be a multiple of the unit-cell length ψ.n
#   (ArgumentError otherwise);
# - left edge: the wraparound Schmidt vector ψ.λ[end] is capped into the
#   first tensor, which carries a dimension-1 "Link,l=0" marker index;
# - right edge: the last tensor keeps a dangling full-bond-dimension
#   "Link,l=L" index (the exact purification of the right environment);
# - for a canonical iMPS the window satisfies inner(mps, mps) ≈ 1, and
#   observables away from the capped left edge match the infinite state.
#
# Everything lives in a module so the file is self-contained and does not
# leak names into Main (test_aklt_integration.jl imports AKLT_TENSOR there).
#
# NOTE: InfiniteTEBD and ITensorMPS both export `expect` — qualify both.
module TestITensorsInterop

using Test
using InfiniteTEBD
using ITensors
using ITensorMPS
using LinearAlgebra
using Random

# Deterministic spin-1 AKLT tensor (χ = 2, d = 3).
const AKLT = begin
    t = zeros(ComplexF64, 2, 3, 2)
    t[1, 1, 2] = +sqrt(2 / 3)
    t[1, 2, 1] = -sqrt(1 / 3)
    t[2, 2, 2] = +sqrt(1 / 3)
    t[2, 3, 1] = -sqrt(2 / 3)
    t
end

# Deterministic symmetry-breaking perturbation: the pure AKLT state has
# ⟨Sz⟩ = 0 on every site, which would make the bulk-match test vacuous.
# The perturbed state stays injective and gapped, so the left-edge cap
# artifact decays ≈ 0.34^j into the bulk (measured), reaching ~1e-9 by
# site 19 of an L = 24 window.
const PERT = begin
    t = zeros(ComplexF64, 2, 3, 2)
    t[1, 1, 1] = 0.10
    t[2, 2, 1] = 0.05im
    t[1, 3, 2] = -0.07
    t[2, 1, 2] = 0.03
    t
end

const SZ1 = ComplexF64[1 0 0; 0 0 0; 0 0 -1]   # spin-1 Sz, same basis as ITensors "S=1"
const SZHALF = ComplexF64[0.5 0; 0 -0.5]       # spin-1/2 Sz

# ⟨Sz_c Sz_{c+1}⟩ on the iMPS for the unit-cell pair starting at cell site c
# (the wraparound pair (n, 1) is handled by InfiniteTEBD.expect).
pair_ref(ψ, O, c) = InfiniteTEBD.expect(ψ, kron(O, O), c, mod1(c + 1, ψ.n))

@testset "IMPS2MPS_WINDOW_LENGTH_VALIDATION" begin
    ψ = iMPS([AKLT, AKLT])                       # n = 2
    sites = siteinds("S=1", 8)
    # Windows covering partial unit cells glue mismatched Schmidt bases.
    for badL in (1, 3, 5, 7)
        @test_throws ArgumentError InfiniteTEBD.imps2mps(ψ, sites; L = badL)
    end
    @test_throws ArgumentError InfiniteTEBD.imps2mps(ψ, sites; L = -2)
    @test_throws ArgumentError InfiniteTEBD.imps2mps(ψ, sites; L = 10)  # L > length(sites)
    @test length(InfiniteTEBD.imps2mps(ψ, sites; L = 0)) == 0
    for okL in (2, 4, 8)
        @test length(InfiniteTEBD.imps2mps(ψ, sites; L = okL)) == okL
    end
    # Physical-dimension mismatches are still rejected.
    @test_throws ArgumentError InfiniteTEBD.imps2mps(ψ, siteinds("S=1/2", 4))
end

@testset "IMPS2MPS_CANONICAL_INNER_IS_ONE" begin
    # Before the fix the wraparound link was reused as the left edge, giving
    # inner(mps, mps) == χ (= 2 for AKLT) instead of 1.
    ψa = iMPS([AKLT, AKLT])
    mps = InfiniteTEBD.imps2mps(ψa, siteinds("S=1", 12))
    @test abs(inner(mps, mps) - 1) < 1e-8
    @test norm(mps) ≈ 1 atol = 1e-8

    Random.seed!(20260609)
    ψr = rand_iMPS(ComplexF64, 2, 2, 4)
    canonical!(ψr)
    mpsr = InfiniteTEBD.imps2mps(ψr, siteinds("S=1/2", 16))
    @test abs(inner(mpsr, mpsr) - 1) < 1e-8

    ψp = iMPS([AKLT .+ PERT, AKLT .- 0.5 .* PERT])
    mpsp = InfiniteTEBD.imps2mps(ψp, siteinds("S=1", 8))
    @test abs(inner(mpsp, mpsp) - 1) < 1e-8
end

@testset "IMPS2MPS_EDGE_INDEX_STRUCTURE" begin
    ψ = iMPS([AKLT, AKLT])
    sites = siteinds("S=1", 8)
    mps = InfiniteTEBD.imps2mps(ψ, sites)
    L = length(mps)

    # Site-index detection must find the physical indices, not boundary
    # links (before the fix siteind(mps, 1) returned a Link index).
    @test siteind(mps, 1) == sites[1]
    @test siteind(mps, L) == sites[L]
    @test [siteind(mps, j) for j in 1:L] == collect(sites)

    # Left edge: dimension-1 marker index tagged "Link,l=0".
    lb = uniqueind(mps[1], mps[2]; tags = "Link")
    @test dim(lb) == 1
    @test hastags(lb, "l=0")

    # Right edge: dangling purification leg of the full bond dimension.
    rb = uniqueind(mps[L], mps[L - 1]; tags = "Link")
    @test dim(rb) == length(ψ.λ[end])
    @test hastags(rb, "l=$L")

    # n = 1, L = 1: the single tensor has three distinct legs
    # (site, dim-1 left marker, dangling right) — no self-contraction.
    ψ1 = product_iMPS(ComplexF64, [[1, 0]])
    m1 = InfiniteTEBD.imps2mps(ψ1, [Index(2, "site=1")]; L = 1)
    @test length(inds(m1[1])) == 3
    @test length(unique(inds(m1[1]))) == 3
    @test abs(inner(m1, m1) - 1) < 1e-8
end

@testset "IMPS2MPS_ITENSORMPS_EXPECT_MATCHES_IMPS" begin
    # Pure AKLT: ⟨Sz⟩ = 0; matches at machine precision on every site.
    ψa = iMPS([AKLT, AKLT])
    La = 12
    mps = InfiniteTEBD.imps2mps(ψa, siteinds("S=1", La))
    ez = ITensorMPS.expect(mps, "Sz")
    @test ez isa AbstractVector{<:Real} && length(ez) == La
    refa = [InfiniteTEBD.expect(ψa, SZ1, c, c) for c in 1:2]
    @test maximum(abs(ez[j] - refa[mod1(j, 2)]) for j in 1:La) < 1e-12

    # Symmetry-broken deterministic state: nonzero ⟨Sz⟩. Bulk sites away
    # from the capped left edge match the infinite-state values.
    ψp = iMPS([AKLT .+ PERT, AKLT .- 0.5 .* PERT])
    Lp = 24
    mpsp = InfiniteTEBD.imps2mps(ψp, siteinds("S=1", Lp))
    ezp = ITensorMPS.expect(mpsp, "Sz")
    refp = [InfiniteTEBD.expect(ψp, SZ1, c, c) for c in 1:2]
    @test any(abs(r) > 1e-4 for r in refp)       # the test is non-vacuous
    @test maximum(abs(ezp[j] - refp[mod1(j, 2)]) for j in 19:Lp) < 1e-6

    # Seeded random canonical state (χ = 4): same bulk-match contract.
    Random.seed!(20260609)
    ψr = rand_iMPS(ComplexF64, 2, 2, 4)
    canonical!(ψr)
    Lr = 16
    mpsr = InfiniteTEBD.imps2mps(ψr, siteinds("S=1/2", Lr))
    ezr = ITensorMPS.expect(mpsr, "Sz")
    refr = [InfiniteTEBD.expect(ψr, SZHALF, c, c) for c in 1:2]
    @test maximum(abs(ezr[j] - refr[mod1(j, 2)]) for j in 11:Lr) < 1e-6
end

@testset "IMPS2MPS_CORRELATION_MATRIX_RUNS_AND_MATCHES" begin
    # The docstring-advertised call (threw before the fix).
    ψa = iMPS([AKLT, AKLT])
    La = 12
    mps = InfiniteTEBD.imps2mps(ψa, siteinds("S=1", La))
    C = ITensorMPS.correlation_matrix(mps, "Sz", "Sz")
    @test size(C) == (La, La)
    # AKLT nearest-neighbour ⟨Sz Sz⟩ = -4/9 exactly; for Sz on the pure
    # AKLT state the cap artifact vanishes, so every pair matches.
    @test maximum(abs(C[i, i + 1] + 4 / 9) for i in 1:(La - 1)) < 1e-12
    @test maximum(abs(C[i, i + 1] - pair_ref(ψa, SZ1, mod1(i, 2))) for i in 1:(La - 1)) <
        1e-12

    # Perturbed deterministic state: bulk pairs match the iMPS values.
    ψp = iMPS([AKLT .+ PERT, AKLT .- 0.5 .* PERT])
    Lp = 24
    mpsp = InfiniteTEBD.imps2mps(ψp, siteinds("S=1", Lp))
    Cp = ITensorMPS.correlation_matrix(mpsp, "Sz", "Sz")
    @test size(Cp) == (Lp, Lp)
    @test maximum(abs(Cp[i, i + 1] - pair_ref(ψp, SZ1, mod1(i, 2))) for i in 19:(Lp - 1)) <
        1e-6
end

@testset "IMPS2MPS_ORTHOGONALIZE_INTERIOR" begin
    ψa = iMPS([AKLT, AKLT])
    La = 12
    mps = InfiniteTEBD.imps2mps(ψa, siteinds("S=1", La))
    ez = ITensorMPS.expect(mps, "Sz")
    for k in (2, La ÷ 2, La - 1)
        m = copy(mps)
        orthogonalize!(m, k)
        @test isortho(m)
        @test ITensorMPS.orthocenter(m) == k
        @test abs(inner(m, m) - 1) < 1e-8
        @test maximum(abs.(ITensorMPS.expect(m, "Sz") .- ez)) < 1e-10
    end
end

end # module TestITensorsInterop
