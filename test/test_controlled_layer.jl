using Test
using LinearAlgebra
using Random
using InfiniteTEBD

# Sector-resolved update for controlled-center layers (PXP-type gates on a two-site cell),
# checked against the generic layer update (`structured=false`) as the reference.
module CLPXP
using LinearAlgebra
using InfiniteTEBD
const X = ComplexF64[0 1; 1 0]
const P = ComplexF64[1 0; 0 0]
const N = ComplexF64[0 0; 0 1]
const I2 = Matrix{ComplexF64}(I, 2, 2)
const P3 = Diagonal(ComplexF64[!(((s >> 2) & 1 == 1 && (s >> 1) & 1 == 1) ||
                                 ((s >> 1) & 1 == 1 && s & 1 == 1)) for s in 0:7])
h(c) = kron(P, X, P) + c * kron(I2, N, I2)
layers() = [[(h(1.2), 2, 4)], [(h(0.6), 1, 3)]]
terms() = reduce(vcat, layers())
blockade() = layer_mpo([(I - kron(N, N), 1, 2), (I - kron(N, N), 2, 3)], 2)
dbl(ψ, j) = real(expect(ψ, kron(N, N), j, j + 1))
function projected(θ=0.35)
    ψ = product_iMPS(ComplexF64, [[cos(θ), sin(θ)], [cos(θ), sin(θ)]])
    apply!(ψ, blockade())
    return ψ
end
end
using .CLPXP: CLPXP

const _UF = trotter_mpos(CLPXP.layers(), 0.05; n=2, gate_map=G -> CLPXP.P3 * G)
const _UU = trotter_mpos(CLPXP.layers(), 0.05; n=2)
const _UI = trotter_mpos(CLPXP.layers(), 0.05; n=2, evolution=:imaginary)

function _midrun_candidate()
    ψ = CLPXP.projected()
    e = energy_density(ψ, CLPXP.terms())
    for _ in 1:6
        scarfinder_step!(ψ, _UF, 2; nstep=10, maxdim=16, terms=CLPXP.terms(), target=e, tol=1e-10,
                         structured=false)
    end
    evolve!(ψ, _UF, 5; maxdim=16, structured=false)
    return ψ
end

# Apply W both ways and compare everything a caller can observe.
function _check_same(ψ, W; maxdim=16, cutoff=1e-14, recanonicalize=true)
    a, b = deepcopy(ψ), deepcopy(ψ)
    _, sa = apply!(a, W; maxdim, cutoff, recanonicalize, return_stats=true, structured=false)
    _, sb = apply!(b, W; maxdim, cutoff, recanonicalize, return_stats=true)
    @test sa.path === :generic
    @test sb.path === :controlled
    if !recanonicalize
        canonical!(a; maxdim=10^4, cutoff=0.0)
        canonical!(b; maxdim=10^4, cutoff=0.0)
    end
    @test length.(a.λ) == length.(b.λ)
    @test all(norm(a.λ[i] - b.λ[i]) < 1e-11 for i in 1:2)
    @test abs(per_site_fidelity(a, b) - 1) < 1e-12
    @test abs(energy_density(a, CLPXP.terms()) - energy_density(b, CLPXP.terms())) < 1e-11
    @test all(abs(CLPXP.dbl(a, j) - CLPXP.dbl(b, j)) < 1e-11 for j in 1:2)
    @test sa.kept_dims == sb.kept_dims
    @test sa.num_saturated == sb.num_saturated
    @test abs(sa.max_discarded_weight - sb.max_discarded_weight) < 1e-14
end

@testset "controlled-center structure is detected" begin
    for U in (_UF, _UU, _UI), W in U
        @test W.structure isa InfiniteTEBD.ControlledCenter
    end
    @test _UF[1].structure.control == 2 && _UF[1].structure.center == 1
    @test _UF[2].structure.control == 1 && _UF[2].structure.center == 2
    # not controlled: a gate that is not diagonal in its outer sites, or a four-site cell
    rng = MersenneTwister(1)
    A = randn(rng, ComplexF64, 8, 8)
    @test isnothing(layer_mpo([(exp(-0.1im * (A + A')), 2, 4)], 2; check=false).structure)
    @test isnothing(layer_mpo([(exp(-0.1im * CLPXP.h(1.0)), 2, 4), (exp(-0.1im * CLPXP.h(1.0)), 4, 6)], 4).structure)
    @test isnothing(layer_mpo([(exp(-0.1im * CLPXP.h(1.0)), 2, 4)], 2; structured=false).structure)
    # products: the merged half steps keep it, a product of different layers does not
    merged = InfiniteTEBD._merged_boundary(_UF[3], _UF[1])
    @test merged.structure isa InfiniteTEBD.ControlledCenter
    @test isnothing(mpo_multiply(_UF[2], _UF[1]).structure)
end

@testset "structured update equals the generic update" begin
    Random.seed!(7)
    candidate = _midrun_candidate()
    inputs = [CLPXP.projected(), product_iMPS(ComplexF64, [[0, 1], [1, 0]]), candidate,
              rand_iMPS(ComplexF64, 2, 2, 8)]                 # the last one leaks out of the blockade
    for ψ in inputs, U in (_UF, _UU, _UI), W in U[1:2]
        _check_same(ψ, W)
    end
    # merged A half steps, and inputs that were never regauged (recanonicalize=false before)
    merged = InfiniteTEBD._merged_boundary(_UF[3], _UF[1])
    _check_same(candidate, merged)
    ψ = deepcopy(candidate)
    apply!(ψ, _UF[1]; maxdim=8, recanonicalize=false)
    _check_same(ψ, _UF[2]; maxdim=8)
    _check_same(ψ, _UF[2]; maxdim=8, recanonicalize=false)
    # a window: every application takes the structured path, same final state
    a, b = deepcopy(candidate), deepcopy(candidate)
    _, sa = evolve!(a, _UF, 10; maxdim=16, return_stats=true, structured=false)
    _, sb = evolve!(b, _UF, 10; maxdim=16, return_stats=true)
    @test sa.num_structured == 0 && sb.num_structured == length(sb.updates) == 21
    @test abs(per_site_fidelity(a, b) - 1) < 1e-12
    @test sa.num_saturated == sb.num_saturated
    @test [u.kept_dims for u in sa.updates] == [u.kept_dims for u in sb.updates]
end

@testset "fallback to the generic update" begin
    ψ = CLPXP.projected()
    W = _UF[2]
    path(state; kw...) = apply!(deepcopy(state), W; return_stats=true, kw...)[2].path
    @test path(ψ) === :controlled
    @test path(ψ; structured=false) === :generic
    @test path(ψ; renormalize=false) === :generic                   # outside the verified domain
    @test path(InfiniteTEBD.iMPS(repeat(ψ.Γ, 2), repeat(ψ.λ, 2), 4)) === :generic   # 4-site cell
    # forced non-convergence of the sector iteration: falls back and still agrees
    old = InfiniteTEBD._CONTROLLED_MAXITER[]
    try
        InfiniteTEBD._CONTROLLED_MAXITER[] = 0
        c = _midrun_candidate()
        a, b = deepcopy(c), deepcopy(c)
        _, sa = apply!(a, W; maxdim=16, return_stats=true)
        apply!(b, W; maxdim=16, structured=false)
        @test sa.path === :generic
        @test abs(per_site_fidelity(a, b) - 1) < 1e-14
    finally
        InfiniteTEBD._CONTROLLED_MAXITER[] = old
    end
    # annihilation is decided by the generic path and leaves the input unchanged
    full = product_iMPS(ComplexF64, [[0, 1], [0, 1]])          # |11⟩: every center is blocked
    saved = deepcopy(full)
    @test_throws MPOAnnihilationError apply!(full, _UF[2])
    @test full.Γ == saved.Γ && full.λ == saved.λ
end

@testset "ScarFinder: structured and generic runs agree" begin
    run(structured) = begin
        ψ = CLPXP.projected()
        e = energy_density(ψ, CLPXP.terms())
        out = [scarfinder_step!(ψ, _UF, 2; nstep=10, maxdim=16, terms=CLPXP.terms(), target=e,
                                tol=1e-10, constraint=CLPXP.blockade(), structured) for _ in 1:6]
        ψ, out
    end
    ψg, rg = run(false)
    ψs, rs = run(true)
    @test all(r.evolve_structured == 0 for r in rg)
    @test all(r.evolve_structured == 21 for r in rs)
    @test maximum(abs(a.loss - b.loss) for (a, b) in zip(rg, rs)) < 1e-11
    @test maximum(abs(a.leakage_projected - b.leakage_projected) for (a, b) in zip(rg, rs)) < 1e-11
    @test all(a.evolve_saturated == b.evolve_saturated for (a, b) in zip(rg, rs))
    @test all(a.energy_fix_steps == b.energy_fix_steps for (a, b) in zip(rg, rs))
    @test abs(per_site_fidelity(ψg, ψs) - 1) < 1e-12
    # opt-in adaptive window truncation shrinks bonds and stays close to the reference
    ψ = CLPXP.projected()
    e = energy_density(ψ, CLPXP.terms())
    r0 = scarfinder_step!(deepcopy(ψ), _UF, 2; nstep=10, maxdim=16, terms=CLPXP.terms(), target=e, tol=1e-10)
    r1 = scarfinder_step!(deepcopy(ψ), _UF, 2; nstep=10, maxdim=16, terms=CLPXP.terms(), target=e, tol=1e-10,
                          evolve_truncerr=1e-14)
    @test abs(r1.loss - r0.loss) < 1e-9
end
