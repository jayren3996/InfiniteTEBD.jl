using Test
using LinearAlgebra
using InfiniteTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end

@testset "SCARFINDER_FOURTH_ORDER_COMPOSES_WITH_PUBLIC_PRIMITIVES" begin
    P = ComplexF64[0 0; 0 1]
    X = ComplexF64[0 1; 1 0]
    h = kron(P, X, P)
    template = product_iMPS(ComplexF64, repeat([[0, 1], [1, 0]], 3))
    layers = uniform_layers(template, h)
    @test [[(i, j) for (_, i, j) in layer] for layer in layers] ==
          [[(1, 3), (4, 6)], [(2, 4), (5, 1)], [(3, 5), (6, 2)]]

    for n in (3, 4, 5, 7, 8, 9)
        ψ = product_iMPS(ComplexF64, fill([0, 1], n))
        groups = uniform_layers(ψ, h)
        @test length(groups) == cld(n, fld(n, 3))
        @test sort([i for group in groups for (_, i, _) in group]) == collect(1:n)
        for group in groups
            sites = [mod1(i + k, n) for (_, i, _) in group for k in 0:2]
            @test allunique(sites)
        end
    end

    dt, χ = 0.02, 2
    scheme_cases = ((:second, layers, nothing), (:fourth, layers, nothing),
        (:fourth_opt, [[(h, i, mod1(i + 2, 6)) for i in k:2:6] for k in 1:2], :explicit))
    for (scheme, groups, mode) in scheme_cases
        manual = deepcopy(template)
        gates = trotter_gates(groups, dt; trotter=scheme)
        evolve!(manual, gates, 2; maxdim=8)
        truncate_unitcell!(manual, χ)
        options = (; nstep=2, maxdim=8, trotter=scheme)
        if mode === :explicit
            options = (; options..., layers=groups)
        end
        step = deepcopy(template)
        run = deepcopy(template)
        @test scarfinder_step!(step, h, dt, χ; options...) === step
        @test scarfinder!(run, h, dt, χ, 1; refine=false, options...) === run
        @test abs(inner_product(manual, step)) ≈ 1.0 atol=1e-10
        @test abs(inner_product(manual, run)) ≈ 1.0 atol=1e-10
    end
    @test_throws ArgumentError scarfinder_step!(deepcopy(template), h, dt, χ; trotter=:fourth_opt)
    @test_throws ArgumentError scarfinder!(deepcopy(template), h, dt, χ, 1; layers)

    legacy = deepcopy(template)
    explicit_gate = deepcopy(template)
    scarfinder!(legacy, h, dt, χ, 2; nstep=2, maxdim=8, refine=false)
    scarfinder!(explicit_gate, exp(-im * dt * h), h, χ, 2; nstep=2, maxdim=8, refine=false)
    @test abs(inner_product(legacy, explicit_gate)) ≈ 1.0 atol=1e-10

    # Build an entangled state and verify refinement against explicitly
    # sampling backward using gates at refine_dt.
    ψ = deepcopy(template)
    evolve!(ψ, trotter_gates(layers, dt; trotter=:fourth), 2; maxdim=8)
    truncate_unitcell!(ψ, χ)
    refine_dt = -dt / 2
    trial = deepcopy(ψ)
    objective(state) = maximum(ent_S(state, i) for i in 1:state.n)
    initial_entropy = objective(trial)
    best_entropy = initial_entropy
    refine_gates = trotter_gates(layers, refine_dt; trotter=:fourth)
    for _ in 1:3
        evolve!(trial, refine_gates, 2; maxdim=8)
        truncate_unitcell!(trial, χ)
        best_entropy = min(best_entropy, objective(trial))
    end
    scarfinder!(ψ, h, dt, χ, 0; trotter=:fourth, refine_dt, refine_step=3, nstep=2, maxdim=8)
    @test best_entropy < initial_entropy
    @test objective(ψ) ≈ best_entropy atol=1e-10
end
using .TestUtils: bell_gate, pauli_matrices

@testset "OPERATOR_SPAN_VALIDATION" begin
    P = pauli_matrices()
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test operator_span(ψ, P.Z) == 1
    @test operator_span(ψ, kron(P.Z, P.Z)) == 2
    @test_throws ArgumentError operator_span(ψ, ones(ComplexF64, 1, 1))
    @test_throws ArgumentError operator_span(ψ, ones(ComplexF64, 2, 3))
    @test_throws ArgumentError operator_span(ψ, Matrix{ComplexF64}(I, 3, 3))

    ψ_scalar = product_iMPS(ComplexF64, [[1]])
    @test_throws ArgumentError operator_span(ψ_scalar, Matrix{ComplexF64}(I, 2, 2))
end

@testset "ENERGY_DENSITY_PRODUCT_STATE" begin
    P = pauli_matrices()
    ψ00 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    ψ01 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test energy_density(ψ00, P.Z) ≈ 1.0 atol=1e-12
    @test energy_density(ψ01, P.Z) ≈ 0.0 atol=1e-12
    @test energy_density(ψ01, kron(P.Z, P.Z)) ≈ -1.0 atol=1e-12
end

@testset "SCARFINDER_NSTEP_VALIDATION" begin
    @test_throws ArgumentError InfiniteTEBD._warn_scarfinder_nstep(0, :gate)
    @test InfiniteTEBD._warn_scarfinder_nstep(2, :gate) === nothing
end

@testset "SCARFINDER_UNIFORM_EVOLVE_VALIDATION" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    G = Matrix{ComplexF64}(I, 2, 2)

    @test_throws ArgumentError InfiniteTEBD._evolve_uniform!(ψ, G; span=0)
    @test_throws ArgumentError InfiniteTEBD._evolve_uniform!(ψ, Matrix{ComplexF64}(I, 8, 8); span=3)
    @test InfiniteTEBD._evolve_uniform!(ψ, G; span=1, maxdim=1) === ψ
end

@testset "SCARFINDER_TRUNCATE_ONE_SITE_UNITCELL" begin
    ψ = product_iMPS(ComplexF64, [[1, 0]])

    @test InfiniteTEBD._truncate_unitcell!(ψ, 1) === ψ
    @test ψ.n == 1
    @test length(ψ.λ[1]) == 1
end

@testset "SCARFINDER_UNIFORM_EVOLVE_SPAN2_MATCHES_EXPLICIT_SWEEP" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    G = bell_gate()

    ψ_uniform = deepcopy(ψ)
    ψ_explicit = deepcopy(ψ)

    InfiniteTEBD._evolve_uniform!(ψ_uniform, G; span=2, maxdim=4)
    applygate!(ψ_explicit, G, 1, 2; maxdim=4)
    applygate!(ψ_explicit, G, 2, 1; maxdim=4)

    @test size.(ψ_uniform.Γ) == size.(ψ_explicit.Γ)
    @test length.(ψ_uniform.λ) == length.(ψ_explicit.λ)
    @test ψ_uniform.λ[1] ≈ ψ_explicit.λ[1] atol=1e-12
    @test ψ_uniform.λ[2] ≈ ψ_explicit.λ[2] atol=1e-12
    if size.(ψ_uniform.Γ) == size.(ψ_explicit.Γ)
        @test InfiniteTEBD.inner_product(ψ_uniform, ψ_explicit) ≈ 1.0 atol=1e-12
    end
end

@testset "MINIMIZE_ON_TRAJECTORY_LANDS_AT_BEST_VALUE" begin
    # Build a synthetic step!/f pair where the minimum occurs strictly
    # mid-trajectory. The fixed-trajectory test verifies that the final
    # state matches the trial state at the minimum point, instead of
    # whatever a replayed step! would produce.
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    f = x -> -float(x.λ[1][1])         # decreasing in λ[1][1]
    counter = Ref(0)
    function step!(x)
        counter[] += 1
        # Mutate the Schmidt vector so that f(x) traces 0, -1, -2, -1, 0...
        # over four step!s. The minimum is therefore at sample index 3.
        idx = mod(counter[] - 1, 4) + 1
        x.λ[1][1] = (idx == 3 ? 2.0 : idx == 2 ? 1.0 : idx == 4 ? 1.0 : 0.0)
        return x
    end
    result = minimize_on_trajectory!(f, step!, ψ, 4)
    @test result == (step=3, value=-2.0)
    @test f(ψ) == -2.0
    @test ψ.λ[1][1] == 2.0
end

@testset "MINIMIZE_ON_TRAJECTORY_KEEPS_FIRST_TIED_POINT" begin
    ψ = product_iMPS(ComplexF64, [[1, 0]])
    X = ComplexF64[0 1; 1 0]
    P = ComplexF64[1 0; 0 0]
    step!(state) = applygate!(state, X, 1, 1)
    objective(state) = expect(state, P, 1, 1)
    @test minimize_on_trajectory!(objective, step!, ψ, 0) == (step=0, value=1.0)
    @test minimize_on_trajectory!(objective, step!, ψ, 4) == (step=1, value=0.0)
    @test objective(ψ) == 0.0
    @test minimize_on_trajectory!(objective, step!, ψ, 4) == (step=0, value=0.0)
    @test objective(ψ) == 0.0
    @test_throws ArgumentError minimize_on_trajectory!(objective, step!, ψ, -1)
end

@testset "SCARFINDER_PREPARED_GATE_SEQUENCES" begin
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    I3 = Matrix{ComplexF64}(I, 3, 3)
    ψ = iMPS(ComplexF64,
        [reshape(ComplexF64.(v), 1, length(v), 1) for v in [[1, 0], [1, 0, 0], [1, 0]]];
        renormalize=false)
    gate = EndpointGate(exp(-0.2im * kron(X, X)), (2, 2))
    P = Diagonal([1.0, 0.0])
    controlled = ControlledGate(P, Diagonal(ComplexF64[exp(0.1im), 1, 1]), P)
    gates = Any[(I3, 2, 2), (gate, 1, 3), (controlled, 1, 3)]
    manual = deepcopy(ψ)
    evolve!(manual, gates, 2; maxdim=6)
    truncate_unitcell!(manual, 2; recanonicalize=true)
    @test scarfinder_step!(ψ, gates, 2; nstep=2, maxdim=6) === ψ
    @test size.(ψ.Γ, 2) == [2, 3, 2]
    @test inner_product(ψ, manual) ≈ 1.0 atol=1e-10
    @test_throws ArgumentError scarfinder_step!(ψ, gates, 2; nstep=2, target=0.0)

    # Refinement can advance along a different, precomputed sequence.
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0], [1, 0]])
    main = ((EndpointGate(exp(-0.2im * kron(X, X)), (2, 2)), 1, 3),)
    back = ((EndpointGate(exp(0.2im * kron(X, X)), (2, 2)), 1, 3),)
    scarfinder!(ψ, main, 2, 1; nstep=2, maxdim=6, refine_gates=back, refine_step=1)
    @test maximum(ent_S(ψ, i) for i in 1:ψ.n) < 1e-10

    # The correction Hamiltonian is independent of the prepared driving gates.
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0], [1, 0]])
    gates = ((Z, 1, 1),)
    @test scarfinder!(ψ, gates, X, 1, 1; nstep=2, target=0.25,
        α=0.8, refine=false) === ψ
    @test energy_density(ψ, X) ≈ 0.25 atol=1e-6
end
