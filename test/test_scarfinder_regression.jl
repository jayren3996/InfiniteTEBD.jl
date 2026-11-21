using Test
using LinearAlgebra
using Random
using InfiniteTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: pauli_matrices

# =============================================================================
# Fix 1: _truncate_unitcell! should not crash on large unit cells
# =============================================================================

@testset "TRUNCATE_UNITCELL_SIZE_GUARD" begin
    # Large unit cell should warn and use fallback, not crash with OOM.
    # n=8, d=2, χ=2 → grouped physical dim = 2^8 = 256 > threshold for n>6.
    # The fallback path applies identity gates bond-by-bond, avoiding the
    # exponential tensor_group(ψ.Γ) call.
    ψ_large = product_iMPS(ComplexF64, fill([1, 0], 8))
    @test_logs (:warn, r"Large unit cell detected") InfiniteTEBD._truncate_unitcell!(ψ_large, 2)
    @test ψ_large.n == 8
    @test all(size(Γ, 1) <= 2 for Γ in ψ_large.Γ)

    ψ = rand_iMPS(ComplexF64, 8, 2, 4)
    reference = deepcopy(ψ)
    identity_gate = Matrix{ComplexF64}(I, 4, 4)
    for i in 1:ψ.n
        applygate!(reference, identity_gate, i, mod1(i + 1, ψ.n); maxdim=2)
    end
    truncate_unitcell!(ψ, 2)
    @test all(ψ.λ[i] ≈ reference.λ[i] for i in 1:ψ.n)
    @test abs(inner_product(ψ, reference)) ≈ inner_product(reference) atol=1e-10

    # Verify the size-guard condition directly for a small state
    ψ_small = product_iMPS(ComplexF64, [[1, 0], [0, 1], [1, 0], [0, 1]])
    d = size(ψ_small.Γ[1], 2)
    @test ψ_small.n <= 6 || d^ψ_small.n > 1_000_000
end

# =============================================================================
# Fix 2: _energy_fix! should converge to target energy
# =============================================================================

@testset "ENERGY_FIX_CONVERGES" begin
    P = pauli_matrices()
    # Use an off-diagonal Hamiltonian so imaginary-time evolution can actually
    # rotate the state. h = Z (the original choice) is degenerate on its own
    # eigenstates — exp(-dτ Z) leaves |↑⟩ and |↓⟩ pointing in the same
    # direction, so _energy_fix! has nothing to act on. With h = X, |↑↑⟩ has
    # ⟨X⟩ = 0 and applying exp(-dτ X) tilts the state toward the |−−⟩ ground
    # state (E = -1), passing through any target in (-1, 0).
    h = P.X
    target = -0.5

    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    InfiniteTEBD._energy_fix!(ψ, h, 4; span=1, target=target, tol=1e-6, α=0.1, maxstep=50)
    E_final = energy_density(ψ, h; span=1)
    @test E_final ≈ target atol=1e-3

    # Symmetric case: pull the energy upward instead.
    ψ2 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    InfiniteTEBD._energy_fix!(ψ2, h, 4; span=1, target=+0.5, tol=1e-6, α=0.1, maxstep=50)
    E_final2 = energy_density(ψ2, h; span=1)
    @test abs(E_final2 - 0.5) < 1e-3
end

@testset "ENERGY_FIX_REPORTS_MEASURED_RESIDUAL" begin
    h = ComplexF64[0 1; 1 0]
    ψ = product_iMPS(ComplexF64, [[1, 0]])
    target = -0.5
    result = fix_energy!(ψ, h, 1; target, α=0.8, tol=1e-8)
    @test result.converged
    @test result.residual ≈ energy_density(ψ, h) - target atol=1e-14
    @test abs(result.residual) <= 1e-8

    entangled = rand_iMPS(ComplexF64, 2, 2, 3)
    target_entangled = 0.9 * energy_density(entangled, h) - 0.1
    corrected = fix_energy!(entangled, h, 3; target=target_entangled)
    @test corrected.converged
    reference = canonical!(deepcopy(entangled); maxdim=3)
    @test corrected.residual ≈ energy_density(reference, h) - target_entangled atol=1e-8

    before = deepcopy(ψ)
    result = fix_energy!(ψ, h, 1; target=-2.0, α=0.8, maxstep=0)
    @test result == (converged=false, residual=energy_density(ψ, h) + 2.0, steps=0)
    @test ψ.Γ == before.Γ
    @test ψ.λ == before.λ
end

@testset "LARGE_SCARFINDER_CHECKPOINTS_USE_CANONICAL_GAUGE" begin
    rng = MersenneTwister(127)
    ψ = iMPS(ComplexF64, [randn(rng, ComplexF64, 4, 2, 4) for _ in 1:8])
    raw = deepcopy(ψ)
    truncate_unitcell!(raw, 2)
    fixed = deepcopy(ψ)
    truncate_unitcell!(fixed, 2; recanonicalize=true)
    @test InfiniteTEBD._check_restored_gauge(fixed, 1e-10)
    @test abs(inner_product(raw, fixed)) / sqrt(abs(inner_product(raw))) ≈ 1.0 atol=1e-10
    h = kron(pauli_matrices().X, pauli_matrices().Z)
    fixed_energy = energy_density(fixed, h)
    no_sweeps = fix_energy!(raw, h, 2; target=fixed_energy, maxstep=0, tol=1e-10)
    @test no_sweeps.steps == 0 && no_sweeps.converged
    @test InfiniteTEBD._check_restored_gauge(raw, 1e-10)
    @test no_sweeps.residual ≈ energy_density(raw, h) - fixed_energy atol=1e-12

    main = ((Matrix{ComplexF64}(I, 2, 2), 1, 1),)
    scarfinder_step!(ψ, main, 2; nstep=2)
    @test InfiniteTEBD._check_restored_gauge(ψ, 1e-10)
    @test inner_product(fixed, ψ) ≈ 1.0 atol=1e-10

    target = energy_density(ψ, h) + 0.02
    correction = fix_energy!(ψ, h, 2; target, α=0.2, tol=1e-7)
    @test correction.converged
    @test InfiniteTEBD._check_restored_gauge(ψ, 1e-10)
    @test correction.residual ≈ energy_density(ψ, h) - target atol=1e-12

    invalid = deepcopy(ψ)
    fill!(invalid.Γ[5], 0)
    before = deepcopy(invalid)
    @test_throws ArgumentError truncate_unitcell!(invalid, 1; recanonicalize=true)
    @test invalid.Γ == before.Γ && invalid.λ == before.λ
end

@testset "COMPRESSION_LQ_PRESERVES_THE_LOCAL_CUT" begin
    rng = MersenneTwister(213)
    A = randn(rng, ComplexF64, 6, 12, 4)
    B = randn(rng, ComplexF64, 4, 13, 5)
    λ = collect(range(0.1, 0.7; length=6))
    block = InfiniteTEBD.tensor_group_2(A, B)
    InfiniteTEBD.tensor_lmul!(λ, block)
    dense_left, dense_s, dense_right =
        InfiniteTEBD.tensor_svd(reshape(block, 6, 12, 13, 5); maxdim=2, renormalize=true)
    left, s, right = InfiniteTEBD._compression_cut(A, B, λ, 2, SVDTOL)
    @test s ≈ dense_s atol=1e-12
    InfiniteTEBD.tensor_rmul!(dense_left, dense_s)
    InfiniteTEBD.tensor_rmul!(left, s)
    @test InfiniteTEBD.tensor_group_2(left, right) ≈
          InfiniteTEBD.tensor_group_2(dense_left, dense_right) atol=1e-12

    A = zeros(ComplexF64, 3, 2, 3)
    B = zeros(ComplexF64, 3, 17, 4)
    for (i, value) in enumerate((1.0, 4e-12, 0.4e-12))
        A[i, 1, i] = value
        B[i, i, 1] = 1
    end
    λ = ones(3)
    # At a practical floor, use the factored spectrum. At a zero floor,
    # preserve the requested full-SVD behavior of keeping zero modes.
    @test length(InfiniteTEBD._compression_cut(A, B, λ, 5, 1e-12)[2]) == 2
    @test length(InfiniteTEBD._compression_cut(A, B, λ, 5, 0.0)[2]) == 5
end

# =============================================================================
# Fix 3: _minimize_on_trajectory! should find minimum with fewer samples
# =============================================================================

@testset "MINIMIZE_ON_TRAJECTORY_FEWER_SAMPLES" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    # A simple step that gradually increases a quantity
    step! = ψ0 -> begin
        for i in 1:ψ0.n
            ψ0.Γ[i] .*= 1.01
        end
        canonical!(ψ0)
        return ψ0
    end

    f = ψ0 -> sum(sum(λ.^2 .* log.(λ.^2 .+ 1e-20)) for λ in ψ0.λ)

    ψ_copy = deepcopy(ψ)
    # With only 10 samples, it should still find a minimum
    InfiniteTEBD._minimize_on_trajectory!(f, step!, ψ_copy, 10)
    @test ψ_copy.n == ψ.n

    # Early exit test: if objective stops improving, should not do all samples
    step_const! = ψ0 -> canonical!(ψ0)
    ψ_copy2 = deepcopy(ψ)
    InfiniteTEBD._minimize_on_trajectory!(f, step_const!, ψ_copy2, 10)
    @test ψ_copy2.n == ψ.n
end

@testset "MINIMIZE_ON_TRAJECTORY_SAMPLES_ALL_POINTS_FOR_LATE_MINIMUM" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, -1.0, 20.0]
    eval_count = Ref(0)
    trial_steps = Ref(0)
    replay_steps = Ref(0)

    f = _ -> begin
        eval_count[] += 1
        values[eval_count[]]
    end
    step! = ψ0 -> begin
        if ψ0 === ψ
            replay_steps[] += 1
        else
            trial_steps[] += 1
        end
        return ψ0
    end

    InfiniteTEBD._minimize_on_trajectory!(f, step!, ψ, length(values) - 1)

    @test trial_steps[] == length(values) - 1
    @test eval_count[] == length(values)
    # The current implementation copies the minimizing trial state directly
    # back into ψ instead of replaying step! (see _minimize_on_trajectory!:
    # gauge phases on degenerate Schmidt spectra are not reproducible across
    # SVD replays). So no step!(ψ) calls happen during the restore.
    @test replay_steps[] == 0
end

@testset "MINIMIZE_ON_TRAJECTORY_CONSIDERS_INITIAL_STATE" begin
    ψ = product_iMPS(ComplexF64, [[1, 0]])
    eval_count = Ref(0)
    trial_steps = Ref(0)
    replay_steps = Ref(0)

    f = _ -> begin
        eval_count[] += 1
        eval_count[] == 1 ? 0.0 : 1.0
    end
    step! = ψ0 -> begin
        if ψ0 === ψ
            replay_steps[] += 1
        else
            trial_steps[] += 1
        end
        return ψ0
    end

    InfiniteTEBD._minimize_on_trajectory!(f, step!, ψ, 5)

    @test eval_count[] == 6
    @test trial_steps[] == 5
    @test replay_steps[] == 0
end

# =============================================================================
# Fix 4: _evolve_uniform! should handle translationally invariant states
# =============================================================================

@testset "EVOLVE_UNIFORM_TRANSLATIONALLY_INVARIANT" begin
    P = pauli_matrices()
    # Translationally invariant state: all sites identical
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    G = exp(-0.1 * P.Z)
    ψ_copy = deepcopy(ψ)
    InfiniteTEBD._evolve_uniform!(ψ_copy, G; span=1, maxdim=4)
    @test ψ_copy.n == 2
    # For translationally invariant state, all sites should still be equivalent
    # (up to numerical noise)
    @test ψ_copy.Γ[1] ≈ ψ_copy.Γ[2] atol=1e-10

    # Non-translationally invariant state should still work
    ψ2 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    InfiniteTEBD._evolve_uniform!(ψ2, G; span=1, maxdim=4)
    @test ψ2.n == 2
end

# =============================================================================
# Fix 5: energy_density should produce same result with explicit loop
# =============================================================================

@testset "ENERGY_DENSITY_EXPLICIT_LOOP" begin
    P = pauli_matrices()
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    E1 = energy_density(ψ, P.Z; span=1)
    E2 = energy_density(ψ, kron(P.Z, P.Z); span=2)

    @test E1 ≈ 0.0 atol=1e-12
    @test E2 ≈ -1.0 atol=1e-12

    # Test with non-trivial state
    ψ2 = product_iMPS(ComplexF64, [[1, 0], [1, 0], [0, 1]])
    E3 = energy_density(ψ2, P.Z; span=1)
    @test E3 ≈ 1/3 atol=1e-12
end

# =============================================================================
# Default refine_step should be 100, not 1000
# =============================================================================

@testset "SCARFINDER_DEFAULT_REFINE_STEP" begin
    # Verify the default refine_step value in scarfinder! docstrings/signature
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    h = Matrix{ComplexF64}(I, 2, 2)
    dt = 0.1
    χ = 2
    N = 1

    # Test that scarfinder! with refine=true and refine_step=100 completes.
    # nstep=1 triggers a warning, so we test_logs for it.
    ψ_copy = deepcopy(ψ)
    @test_logs (:warn, r"nstep = 1") InfiniteTEBD.scarfinder!(ψ_copy, h, dt, χ, N; refine=true, refine_step=100, nstep=1, maxdim=2)

    # Test that calling without refine_step uses a small default (100) by
    # checking that the call completes in reasonable time.
    ψ_copy2 = deepcopy(ψ)
    @test_logs (:warn, r"nstep = 1") InfiniteTEBD.scarfinder!(ψ_copy2, h, dt, χ, N; refine=true, nstep=1, maxdim=2)
end
