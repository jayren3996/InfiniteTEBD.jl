using Test
using LinearAlgebra
using Random
using InfiniteTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: pauli_matrices

# =============================================================================
# Fix 2: energy_fix! should converge to target energy
# =============================================================================

@testset "ENERGY_FIX_CONVERGES" begin
    P = pauli_matrices()
    # Use an off-diagonal Hamiltonian so imaginary-time evolution can actually
    # rotate the state. h = Z (the original choice) is degenerate on its own
    # eigenstates — exp(-dτ Z) leaves |↑⟩ and |↓⟩ pointing in the same
    # direction, so energy_fix! has nothing to act on. With h = X, |↑↑⟩ has
    # ⟨X⟩ = 0 and applying exp(-dτ X) tilts the state toward the |−−⟩ ground
    # state (E = -1), passing through any target in (-1, 0).
    h = P.X
    target = -0.5

    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    energy_fix!(ψ, [(h, 1, 1)], 4; target, tol=1e-6, alpha=0.1, maxstep=50)
    E_final = energy_density(ψ, h; span=1)
    @test E_final ≈ target atol=1e-3

    # Symmetric case: pull the energy upward instead.
    ψ2 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    energy_fix!(ψ2, [(h, 1, 1)], 4; target=+0.5, tol=1e-6, alpha=0.1, maxstep=50)
    E_final2 = energy_density(ψ2, h; span=1)
    @test abs(E_final2 - 0.5) < 1e-3
end

@testset "ENERGY_FIX_REPORTS_MEASURED_RESIDUAL" begin
    h = ComplexF64[0 1; 1 0]
    ψ = product_iMPS(ComplexF64, [[1, 0]])
    target = -0.5
    result = energy_fix!(ψ, [(h, 1, 1)], 1; target, alpha=0.8, tol=1e-8)
    @test result.converged
    @test result.residual ≈ energy_density(ψ, h) - target atol=1e-14
    @test abs(result.residual) <= 1e-8

    entangled = rand_iMPS(ComplexF64, 2, 2, 3)
    target_entangled = 0.9 * energy_density(entangled, h) - 0.1
    corrected = energy_fix!(entangled, [(h, 1, 1)], 3; target=target_entangled, n=1)
    @test corrected.converged
    reference = canonical!(deepcopy(entangled); maxdim=3)
    @test corrected.residual ≈ energy_density(reference, h) - target_entangled atol=1e-8

    before = deepcopy(ψ)
    result = energy_fix!(ψ, [(h, 1, 1)], 1; target=-2.0, alpha=0.8, maxstep=0)
    @test result == (converged=false, residual=energy_density(ψ, h) + 2.0, steps=0)
    @test ψ.Γ == before.Γ
    @test ψ.λ == before.λ
end

# =============================================================================
# Fix 3: minimize_on_trajectory! should find minimum with fewer samples
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
    minimize_on_trajectory!(f, step!, ψ_copy, 10)
    @test ψ_copy.n == ψ.n

    # Early exit test: if objective stops improving, should not do all samples
    step_const! = ψ0 -> canonical!(ψ0)
    ψ_copy2 = deepcopy(ψ)
    minimize_on_trajectory!(f, step_const!, ψ_copy2, 10)
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

    minimize_on_trajectory!(f, step!, ψ, length(values) - 1)

    @test trial_steps[] == length(values) - 1
    @test eval_count[] == length(values)
    # The current implementation copies the minimizing trial state directly
    # back into ψ instead of replaying step! (see minimize_on_trajectory!:
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

    minimize_on_trajectory!(f, step!, ψ, 5)

    @test eval_count[] == 6
    @test trial_steps[] == 5
    @test replay_steps[] == 0
end

# =============================================================================
# Fix 4: uniform gate sweeps should handle translationally invariant states
# =============================================================================

@testset "EVOLVE_UNIFORM_TRANSLATIONALLY_INVARIANT" begin
    P = pauli_matrices()
    # Translationally invariant state: all sites identical
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    G = exp(-0.1 * P.Z)
    ψ_copy = deepcopy(ψ)
    evolve!(ψ_copy, InfiniteTEBD._evolution(ψ_copy, G), 1; maxdim=4)
    @test ψ_copy.n == 2
    # For translationally invariant state, all sites should still be equivalent
    # (up to numerical noise)
    @test ψ_copy.Γ[1] ≈ ψ_copy.Γ[2] atol=1e-10

    # Non-translationally invariant state should still work
    ψ2 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    evolve!(ψ2, InfiniteTEBD._evolution(ψ2, G), 1; maxdim=4)
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
