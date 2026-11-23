using Test
using LinearAlgebra
using Random
using InfiniteTEBD
if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: right_canonical_error

# Layer-MPO ScarFinder, observation, reproducible seeds, and translation.
module SFW
using LinearAlgebra
using Random
using InfiniteTEBD
const N = ComplexF64[0 0; 0 1]
const X = ComplexF64[0 1; 1 0]
const P = ComplexF64[1 0; 0 0]
const I2 = Matrix{ComplexF64}(I, 2, 2)
const P3 = Diagonal(ComplexF64[(k & (k >> 1)) == 0 for k in 0:7])
fold(G) = P3 * G
h(c) = kron(P, X, P) + c * kron(I2, N, I2)
stag(μ, δ) = [[(h(μ + δ), 2, 4)], [(h(μ - δ), 1, 3)]]      # centres: site 1 (A), site 2 (B)
const LAYERS = stag(0.9, 0.0)
const U = trotter_mpos(LAYERS, 0.05; n=2, gate_map=fold)
const B = Blockade()
const PM = iMPO(B, 2)
gauss_seq(id) = (rng = MersenneTwister(80100 + id);
                 iMPS(ComplexF64, [randn(rng, ComplexF64, 2, 2, 2) for _ in 1:2]; renormalize=true))
example() = apply!(product_iMPS(ComplexF64, [[cos(0.6), sin(0.6) * cis(0.3)], [cos(0.5), sin(0.5)]]), PM)
occ(ψ, j) = real(expect(ψ, N, j, j))
energy(ψ, layers=LAYERS) = energy_density(ψ, reduce(vcat, layers))
end
using .SFW: SFW

@testset "static validation before mutation" begin
    q = kron(SFW.N, SFW.I2, SFW.N)
    ψ0 = SFW.gauss_seq(3)
    # two commuting three-site diagonal terms in one layer crossing the same bond (review reproducer)
    bad = [[(q, 1, 3), (q, 2, 4)]]
    ψ = deepcopy(ψ0)
    @test_throws ArgumentError scarfinder_step!(ψ, SFW.U, 2; nstep=2, maxdim=16, terms=bad, target=0.317)
    @test ψ.Γ == ψ0.Γ && ψ.λ == ψ0.λ
    @test_throws ArgumentError energy_fix!(deepcopy(ψ0), bad, 2; target=0.3)
    ψ = deepcopy(ψ0)
    @test_throws ArgumentError scarfinder_step!(ψ, SFW.U, 2; nstep=2, maxdim=16,
        terms=[[(kron(SFW.I2, SFW.I2), 1, 3)]], target=0.3)
    @test ψ.Γ == ψ0.Γ && ψ.λ == ψ0.λ
end

@testset "MPO ScarFinder matches the physical output of the explicit PXP loop" begin
    ψ = SFW.gauss_seq(3)
    explicit = deepcopy(ψ)
    e = 0.25
    energy_trials = 0
    for nstep in (20, 2, 2)
        r = scarfinder_step!(ψ, SFW.U, 2; nstep, maxdim=16, cutoff=1e-12,
                             terms=SFW.LAYERS, target=e, n=2)

        evolve!(explicit, SFW.U, nstep; maxdim=16, cutoff=1e-12)
        evolved = deepcopy(explicit)
        truncate!(explicit, 2)
        loss = -log(min(per_site_fidelity(evolved, explicit), 1.0))
        correction = energy_fix!(explicit, SFW.LAYERS, 2; target=e, n=2)

        energy_trials += correction.steps
        @test keys(correction) == (:converged, :residual, :steps)
        @test r.converged == correction.converged
        @test r.loss ≈ loss atol=1e-10
        @test r.energy_error ≈ correction.residual atol=1e-10
        @test length.(ψ.λ) == length.(explicit.λ)
        @test per_site_fidelity(ψ, explicit) ≈ 1 atol=1e-12
        @test right_canonical_error(ψ) < 1e-10
        for i in 1:ψ.n
            @test ψ.λ[i] ≈ explicit.λ[i] atol=1e-10
            @test SFW.occ(ψ, i) ≈ SFW.occ(explicit, i) atol=1e-10
        end
    end
    @test energy_trials > 0
end

@testset "MPO projection reuses only a well-separated normalized basis" begin
    id1 = iMPO([reshape(SFW.I2, 1, 2, 2, 1)])
    input = SFW.gauss_seq(3)
    evolved = evolve!(deepcopy(input), SFW.U, 10; maxdim=16)
    @test InfiniteTEBD._well_separated_schmidt_cut(evolved, 2, SVDTOL)
    @test InfiniteTEBD._isolated_transfer_fixed_point(evolved)
    InfiniteTEBD._truncate_canonical!(evolved; maxdim=2, cutoff=SVDTOL,
                                     truncerr=0.0, renormalize=true)
    scarfinder_step!(input, SFW.U, 2; nstep=10, maxdim=16)
    @test input.Γ == evolved.Γ && input.λ == evolved.λ

    ψ = product_iMPS(ComplexF64, [[1, 0]])
    evolved = evolve!(deepcopy(ψ), [id1], 2)
    @test scarfinder_step!(ψ, [id1], 1; nstep=2).loss === 0.0
    @test ψ.Γ == evolved.Γ && ψ.λ == evolved.λ

    # A rank-one cut in a degenerate GHZ Schmidt space keeps the explicit
    # projection's choice of basis.
    A = zeros(ComplexF64, 2, 2, 2)
    A[1, 1, 1] = A[2, 2, 2] = 1
    ψ = iMPS([A], [fill(inv(sqrt(2.0)), 2)], 1)
    evolved = evolve!(deepcopy(ψ), [id1], 2)
    @test !InfiniteTEBD._well_separated_schmidt_cut(evolved, 1, SVDTOL)
    expected = project!(evolved, 1)
    @test scarfinder_step!(ψ, [id1], 1; nstep=2) === expected
    @test ψ.Γ == evolved.Γ && ψ.λ == evolved.λ

    # Two sectors of different virtual dimensions keep different weights
    # in the controlled sweep and in a fresh canonicalization.
    center = zeros(ComplexF64, 3, 2, 3)
    center[1, 1, 1] = 1
    center[2:3, 1, 2:3] = Diagonal(sqrt.([0.99, 0.60]))
    center[2:3, 2, 2:3] = [0 sqrt(0.01); sqrt(0.40) 0]
    control = zeros(ComplexF64, 3, 2, 3)
    control[1, 1, 1] = control[2, 2, 2] = control[3, 2, 3] = 1
    weights = sqrt.([0.95, 0.05 * 0.40 / 0.41, 0.05 * 0.01 / 0.41])
    ψ = iMPS([center, control], [copy(weights), copy(weights)], 2)
    identity_layer = [layer_mpo([(Matrix{ComplexF64}(I, 8, 8), 2, 4)], 2)]
    evolved, stats = evolve!(deepcopy(ψ), identity_layer, 2; return_stats=true)
    @test stats.num_structured == 2
    @test InfiniteTEBD._well_separated_schmidt_cut(evolved, 1, SVDTOL)
    @test !InfiniteTEBD._isolated_transfer_fixed_point(evolved)
    expected = project!(evolved, 1)
    @test SFW.occ(evolved, 2) ≈ 1 atol=1e-12
    @test scarfinder_step!(ψ, identity_layer, 1; nstep=2) === expected
    @test ψ.Γ == evolved.Γ && ψ.λ == evolved.λ

    ψ = product_iMPS(ComplexF64, [[1, 0]])
    ψ.Γ[1] .*= 0.25
    expected = deepcopy(ψ)
    expected_result = project!(expected, 1)
    @test scarfinder_step!(ψ, typeof(id1)[], 1; nstep=2) === expected_result
    @test ψ.Γ == expected.Γ && ψ.λ == expected.λ
end

@testset "observing evolve! leaves the trajectory unchanged" begin
    ψ0 = truncate!(evolve!(SFW.gauss_seq(3), SFW.U, 3; maxdim=8), 2)
    for (merge, steps, every, cap) in ((true, 10, 3, 2), (true, 10, 1, 16), (false, 7, 2, 2))
        plain = evolve!(deepcopy(ψ0), SFW.U, steps; maxdim=cap, merge)
        samples = Any[]
        obs = (φ, k, st) -> push!(samples, (k, deepcopy(φ), st))
        watched = evolve!(deepcopy(ψ0), SFW.U, steps; maxdim=cap, merge, observe=obs, every)
        @test watched.Γ == plain.Γ && watched.λ == plain.λ
        ks = first.(samples)
        @test ks == unique(vcat(0, collect(every:every:steps), steps))
        @test isnothing(samples[1][3]) && samples[1][2].Γ == ψ0.Γ
        for (k, φ, st) in samples[2:end]
            prefix = evolve!(deepcopy(ψ0), SFW.U, k; maxdim=cap, merge)
            @test φ.Γ == prefix.Γ && φ.λ == prefix.λ
            @test st.max_kept_dim <= cap && st.num_at_cap >= 0
        end
    end
    # sampling by repeated one-step calls is a different integrator (no merged boundary)
    a = evolve!(deepcopy(ψ0), SFW.U, 10; maxdim=2)
    b = deepcopy(ψ0)
    for _ in 1:10
        evolve!(b, SFW.U, 1; maxdim=2)
    end
    @test abs(SFW.occ(a, 1) - SFW.occ(b, 1)) > 1e-8
    # release record
    obs, rec = release_recorder(ψ0; dt=0.05, site_ops=(; n=SFW.N))
    evolve!(deepcopy(ψ0), SFW.U, 20; maxdim=4, observe=obs, every=5)
    @test rec.step == [0, 5, 10, 15, 20] && rec.t ≈ [0, 0.25, 0.5, 0.75, 1.0]
    @test rec.return_fidelity[1] ≈ 1 && all(<=(1 + 1e-12), rec.return_fidelity)
    @test all(d -> all(<=(4), d), rec.dims) && length(rec.entropy[end]) == 2
    @test rec.ops[:n][1] ≈ [SFW.occ(ψ0, 1), SFW.occ(ψ0, 2)]
    @test any(>(0), rec.num_at_cap)
end

@testset "observing scarfinder!" begin
    ψ0 = SFW.example()
    e = SFW.energy(ψ0)
    kw = (; nstep=4, maxdim=12, terms=SFW.LAYERS, target=e)
    seen = Any[]
    hist = Any[]
    a = scarfinder!(deepcopy(ψ0), SFW.U, 2, 4; refine=false, history=hist, kw...,
                    observe=(ψ, k, r) -> push!(seen, (k, deepcopy(ψ), r)))
    b = deepcopy(ψ0)
    loop = [(scarfinder_step!(b, SFW.U, 2; kw...), deepcopy(b)) for _ in 1:4]
    @test a.Γ == b.Γ && first.(seen) == 0:4 && isnothing(seen[1][3])
    @test all(seen[k+1][2].Γ == loop[k][2].Γ && seen[k+1][3].loss == loop[k][1].loss for k in 1:4)
    @test length(hist) == 4
    # refinement is not observed and not in the history
    seen2 = Int[]
    hist2 = Any[]
    scarfinder!(deepcopy(ψ0), SFW.U, 2, 2; refine=true, refine_step=3, history=hist2, kw...,
                observe=(ψ, k, r) -> push!(seen2, k))
    @test seen2 == [0, 1, 2] && length(hist2) == 2
end

@testset "reproducible seeds" begin
    Random.seed!(7)
    a = rand_iMPS(ComplexF64, 2, 2, 3)
    b = rand_iMPS(Xoshiro(7), ComplexF64, 2, 2, 3)
    @test a.Γ == b.Γ && a.λ == b.λ                               # same distribution and stream
    c = rand_iMPS(Xoshiro(hash((17, 3))), ComplexF64, 2, 2, 2)
    d = rand_iMPS(Xoshiro(hash((17, 3))), ComplexF64, 2, 2, 2)
    f = rand_iMPS(Xoshiro(hash((17, 4))), ComplexF64, 2, 2, 2)
    @test c.Γ == d.Γ && c.Γ != f.Γ
    @test rand_iMPS(Xoshiro(1), Float64, 3, 2, 2) isa iMPS
end

@testset "translate" begin
    Random.seed!(3)
    ψ = rand_iMPS(ComplexF64, 3, 2, 3)
    truncate!(ψ, 3)
    ψ.Γ[2] = ψ.Γ[2][:, :, 1:2]; ψ.Γ[3] = ψ.Γ[3][1:2, :, :]; ψ.λ[2] = ψ.λ[2][1:2]
    canonical!(ψ)
    @test length.(ψ.λ) != fill(length(ψ.λ[1]), 3)                # unequal bonds
    for k in (-4, -2, -1, 0, 1, 2, 3, 4)
        t = translate(ψ, k)
        @test all(t.Γ[i] == ψ.Γ[mod1(i - k, 3)] && t.λ[i] == ψ.λ[mod1(i - k, 3)] for i in 1:3)
        @test translate(t, -k).Γ == ψ.Γ
    end
    @test translate(ψ, 3).Γ == ψ.Γ && translate(ψ, -3).λ == ψ.λ
    t = translate(ψ, 1)
    t.Γ[2][1] = 99
    t.λ[2][1] = 99
    @test ψ.Γ[1][1] != 99 && ψ.λ[1][1] != 99                       # independent arrays
    @test abs(per_site_fidelity(translate(translate(ψ, 1), 2), ψ) - 1) < 1e-12
    # staggered chain: a one-site translation maps δ to −δ
    φ = SFW.gauss_seq(3)
    μ, δ = 0.9, 0.3
    @test SFW.energy(translate(φ, 1), SFW.stag(μ, δ)) ≈ SFW.energy(φ, SFW.stag(μ, -δ)) atol = 1e-12
    @test SFW.energy(translate(φ, 1), SFW.stag(μ, δ)) - SFW.energy(φ, SFW.stag(μ, δ)) ≈
          -δ * (SFW.occ(φ, 1) - SFW.occ(φ, 2)) atol = 1e-12
end

@testset "annihilation screen with the Frobenius bound" begin
    id1 = iMPO([reshape(SFW.I2, 1, 2, 2, 1)])
    for n in (2, 78, 120)
        ψ = product_iMPS(ComplexF64, [[1, 0] for _ in 1:n])
        @test abs(inner_product(apply!(deepcopy(ψ), id1)) - 1) < 1e-12
    end
    # rescaled and poorly gauged inputs
    ψ = SFW.gauss_seq(2)
    raw = InfiniteTEBD.iMPS([1e3 .* ψ.Γ[1], 1e-2 .* ψ.Γ[2]], [ones(2), ones(2)], 2)
    out = apply!(deepcopy(raw), SFW.U[2]; maxdim=8)
    ref = apply!(deepcopy(ψ), SFW.U[2]; maxdim=8)
    @test abs(per_site_fidelity(out, ref) - 1) < 1e-12
    # true annihilation and small finite weights
    full = product_iMPS(ComplexF64, [[0, 1], [0, 1]])
    @test_throws MPOAnnihilationError apply!(deepcopy(full), SFW.U[2])
    @test_throws MPOAnnihilationError apply!(deepcopy(full), SFW.U[2]; structured=false)
    Pz = iMPO([reshape(SFW.P, 1, 2, 2, 1)])
    for n in (1, 2, 8)
        rare = product_iMPS(ComplexF64, [[1e-6, sqrt(1 - 1e-12)] for _ in 1:n])
        @test abs(constraint_leakage(rare, Pz) - 13.815510557964274) < 1e-9
        @test inner_product(apply!(deepcopy(rare), Pz)) ≈ 1
    end
end
