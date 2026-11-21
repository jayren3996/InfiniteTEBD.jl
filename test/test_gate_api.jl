using Test
using LinearAlgebra
using Random
using InfiniteTEBD

Random.seed!(20260523)

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: bell_gate, pauli_matrices, right_canonical_error

@testset "STRUCTURED PRODUCT AND CONTROLLED GATES" begin
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    P = Diagonal([1.0, 0.0])
    Q = Diagonal([0.0, 1.0])
    α = -0.27im
    product = EndpointGate(X + Y, Diagonal([0.2, -1.1, 0.7]), α)
    matrix = exp(α * kron(X + Y, Diagonal([0.2, -1.1, 0.7])))
    @test sum(kron.(product.left, product.right)) ≈ matrix atol=1e-13
    @test all(op -> op isa Diagonal, product.right)
    @test_throws ArgumentError EndpointGate([0 1; 0 0], X, α)

    H = kron(P, Y, Q)
    controlled = ControlledGate(P, exp(α * Y), Q)
    rng = MersenneTwister(391)
    input = iMPS(ComplexF64, [randn(rng, ComplexF64, 3, 2, 3) for _ in 1:5])
    for (start, stop) in ((1, 3), (4, 1)), cap in (2, 8)
        direct, prepared = deepcopy(input), deepcopy(input)
        _, reference = applygate!(direct, exp(α * H), start, stop; maxdim=cap, return_stats=true)
        _, stats = applygate!(prepared, controlled, start, stop; maxdim=cap, return_stats=true)
        overlap = inner_product(direct, prepared) /
            sqrt(inner_product(direct) * inner_product(prepared))
        @test overlap ≈ 1 atol=1e-10
        @test getfield.(stats.bond_stats, :discarded_weight) ≈
            getfield.(reference.bond_stats, :discarded_weight) atol=1e-13
    end
    @test_throws ArgumentError applygate!(deepcopy(input), controlled, 1, 2)
    @test_throws ArgumentError ControlledGate(Diagonal([1, 0.5]), X, Q)
end

@testset "FIXED_DIMENSION GATE_CHECKPOINT" begin
    rng = MersenneTwister(964)
    input = iMPS(ComplexF64, [randn(rng, ComplexF64, 3, 2, 3) for _ in 1:4])
    G = Diagonal(exp.(0.3 .* [1.0, -1, -1, 1]))
    prepared = EndpointGate(G, (2, 2))
    plain = deepcopy(input)
    _, stats = applygate!(plain, G, 2, 3; maxdim=2, return_stats=true)
    expected_dims = length.(plain.λ)
    restore_canonical!(plain; method=:auto)
    for gate in (G, prepared)
        updated = deepcopy(input)
        _, actual = applygate!(updated, gate, 2, 3; maxdim=2, gauge=:auto, return_stats=true)
        @test length.(updated.λ) == expected_dims
        @test actual.max_discarded_weight ≈ stats.max_discarded_weight atol=1e-13
        @test inner_product(updated, plain) ≈ 1 atol=1e-10
        @test InfiniteTEBD._check_restored_gauge(updated, 1e-10)
    end
    bad = deepcopy(input)
    @test_throws ArgumentError applygate!(bad, G, 2, 3; gauge=:auto, recanonicalize=true)
    @test bad.Γ == input.Γ
    @test bad.λ == input.λ
end

@testset "FACTORED CUTS HONOUR THE MINIMUM AVAILABLE DIMENSION" begin
    X = ComplexF64[0 1; 1 0]
    P = Diagonal([1.0, 0.0])
    G = Matrix{ComplexF64}(I, 8, 8)
    gates = (EndpointGate(Matrix{ComplexF64}(I, 4, 4), (2, 2)),
             ControlledGate(P, X, P))
    for gate in gates
        expected = product_iMPS(ComplexF64, [[0, 1], [1, 0], [1, 0]])
        actual = deepcopy(expected)
        # Both gates are the identity on this input.
        applygate!(expected, G, 1, 3; mindim=2, maxdim=2)
        _, stats = applygate!(actual, gate, 1, 3; mindim=2, maxdim=2, return_stats=true)
        @test length.(actual.λ) == length.(expected.λ) == [2, 2, 1]
        @test all(cut -> cut.chi_keep == 2 && cut.discarded_weight ≈ 0, stats.bond_stats)
        @test inner_product(actual, expected) ≈ 1 atol=1e-10
    end
end

@testset "QR-TEBD CERTIFIES THE WRITTEN RIGHT PROJECTION" begin
    M = [1.0 0.0; 1.0 1.0]
    seed = reshape([1.0, -1.0], 2, 1) / sqrt(2)
    S, Vt, info = InfiniteTEBD._qr_two_site_split(M, seed;
        maxdim=1, projection_tol=0.4, expansion=0)
    @test info.used_qr && info.target_met
    @test only(S) ≈ sqrt(2)
    @test info.projection_error ≈ 1/3
    @test info.projection_error ≈ norm(M - M * Vt' * Vt)^2 / norm(M)^2

    _, _, fallback = InfiniteTEBD._qr_two_site_split(M, seed;
        maxdim=1, projection_tol=0.01, expansion=0)
    @test !fallback.used_qr && !fallback.target_met
    @test fallback.projection_error ≈ last(svdvals(M))^2 / sum(abs2, M)

    # The right-space certificate also applies to the unnormalized tensor
    # reconstructed by the actual adjacent periodic gate path.
    rng = MersenneTwister(803)
    d, χ = 4, 3
    a = iMPS([randn(rng, ComplexF64, χ, d, χ) for _ in 1:3])
    X = Diagonal(ComplexF64[1, 1, -1, -1])
    G = exp(-1e-3im * kron(X, X))
    b = deepcopy(a)
    stats = InfiniteTEBD._applygate_qr_experiment!(a, G, 3, 1;
        maxdim=χ, projection_tol=1e-5, expansion=2)
    applygate!(b, G, 3, 1; maxdim=χ)
    @test stats.used_qr && stats.target_met
    @test maximum(length.(a.λ)) <= χ
    @test inner_product(a, b) / sqrt(inner_product(a)*inner_product(b)) ≈ 1 atol=1e-6
end

@testset "TWO_SITE_RECONSTRUCTION_WITHOUT_SCHMIDT_INVERSION" begin
    A = reshape(ComplexF64[1, 2, 3], 3, 1, 1)
    B = ones(ComplexF64, 1, 1, 1)
    λ = [1.0, 1e-25, 0.0]
    tensors, spectra, stats = InfiniteTEBD.tensor_applygate!(
        ones(ComplexF64, 1, 1), [A, B], λ; maxdim=1, return_stats=true)
    # The supported tiny row remains present, and an exactly unrepresented
    # row keeps the zero convention. Signs of singular vectors are arbitrary.
    @test abs.(vec(first(tensors))) ≈ [1.0, 2.0, 0.0]
    @test only(spectra) ≈ [1.0]
    @test only(stats).discarded_weight == 0

    ψ = rand_iMPS(ComplexF64, 2, 2, 3)
    H = ComplexF64[0 0 0 1; 0 0 2 0; 0 2 0 0; 1 0 0 0]
    G = exp(-0.2im * H)
    A, B = ψ.Γ
    λ = ψ.λ[end]
    weighted = InfiniteTEBD.tensor_group_2(A, B)
    InfiniteTEBD.tensor_lmul!(λ, weighted)
    weighted = InfiniteTEBD.tensor_umul(G, weighted)
    for renormalize in (true, false)
        opts = (; maxdim=2, truncerr=1e-5, svd_min=1e-12, renormalize, return_stats=true)
        ref_B, ref_S, ref_stats = InfiniteTEBD.tensor_decomp!(copy(weighted), λ, 2; opts...)
        new_B, new_S, new_stats = InfiniteTEBD._apply_two_site_gate(G, A, B, λ, 1;
            mindim=1, opts...)
        @test only(ref_S) ≈ only(new_S) atol=1e-12
        @test InfiniteTEBD.tensor_group(ref_B) ≈ InfiniteTEBD.tensor_group(new_B) atol=1e-12
        @test only(ref_stats).chi_req == only(new_stats).chi_req
        @test only(ref_stats).discarded_weight ≈ only(new_stats).discarded_weight atol=1e-13
    end
end

@testset "ENDPOINT_GATES_MATCH_CONTIGUOUS_ORACLE" begin
    rng = MersenneTwister(156)
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    pair = exp(-0.2im * (kron(X, Z) + 0.3 * kron(Z, X)))
    # Embed the same labelled endpoints in the four-site kron convention.
    dense = zeros(ComplexF64, 16, 16)
    for a in 0:1, b in 0:1, c in 0:1, e in 0:1, mid in 0:3
        dense[1+8a+2mid+b, 1+8c+2mid+e] = pair[1+2a+b, 1+2c+e]
    end
    template = iMPS([randn(rng, ComplexF64, 2, 2, 2) for _ in 1:6])
    prepared = EndpointGate(pair, (2, 2))
    for (i, j) in ((1, 4), (5, 2))
        @test expect(template, prepared, i, j) ≈ expect(template, dense, i, j) atol=1e-12
        reference = deepcopy(template)
        applygate!(reference, dense, i, j; maxdim=32)
        for method in (:mpo, :swap)
            ψ = deepcopy(template)
            apply_endpoint_gate!(ψ, pair, i, j; method, maxdim=32)
            @test inner_product(ψ, reference) / sqrt(inner_product(ψ)*inner_product(reference)) ≈ 1 atol=1e-10
        end

        # The MPO backend sweeps the same cuts and uses the same error
        # criterion. Agreement at a tight cap is stronger than exact action.
        a, b = deepcopy(template), deepcopy(template)
        _, expected = applygate!(a, dense, i, j; maxdim=2, truncerr=1e-6, return_stats=true)
        _, actual = apply_endpoint_gate!(b, pair, i, j; maxdim=2, truncerr=1e-6, return_stats=true)
        @test [x.bond for x in actual.bond_stats] == [x.bond for x in expected.bond_stats]
        @test [x.discarded_weight for x in actual.bond_stats] ≈
              [x.discarded_weight for x in expected.bond_stats] atol=1e-12
        @test inner_product(a, b) / sqrt(inner_product(a)*inner_product(b)) ≈ 1 atol=1e-10
        c = deepcopy(template)
        evolve!(c, [(prepared, i, j)], 1; maxdim=2, truncerr=1e-6)
        @test inner_product(c, b) / sqrt(inner_product(c)*inner_product(b)) ≈ 1 atol=1e-10
    end

    diagonal = Diagonal(exp.(-0.2im .* [1, -1, -1, 1]))
    @test length(first(InfiniteTEBD._endpoint_factors(diagonal, 2, 2, nothing))) == 2
    for j in (2, 3)
        a, b = deepcopy(template), deepcopy(template)
        embedded = j == 2 ? diagonal :
            Diagonal(exp.(-0.2im .* diag(kron(Z, Matrix{Int}(I, 2, 2), Z))))
        applygate!(a, Matrix(embedded), 1, j; maxdim=8)
        apply_endpoint_gate!(b, diagonal, 1, j; maxdim=8)
        @test inner_product(a, b) / sqrt(inner_product(a)*inner_product(b)) ≈ 1 atol=1e-10
    end
    @test_throws ArgumentError apply_endpoint_gate!(template, pair, 1, 7)
    @test_throws DimensionMismatch apply_endpoint_gate!(template, X, 1, 4)
end

@testset "GATE_INDICES_NON_WRAPPING" begin
    psi = product_iMPS(ComplexF64, [[1,0], [0,1], [1,0]])
    inds = InfiniteTEBD._gate_indices(psi, 1, 2)
    @test inds == 1:2
    @test inds isa UnitRange
end

@testset "GATE_INDICES_WRAPPING" begin
    psi = product_iMPS(ComplexF64, [[1,0], [0,1], [1,0]])
    inds = InfiniteTEBD._gate_indices(psi, 2, 1)
    @test collect(inds) == [2, 3, 1]
end

@testset "GATE_INDICES_SINGLE_SITE" begin
    psi = product_iMPS(ComplexF64, [[1,0], [0,1]])
    inds = InfiniteTEBD._gate_indices(psi, 1, 1)
    @test collect(inds) == [1]
end

@testset "ONE_SITE_GATE_CORRECTNESS" begin
    X = [0.0 1.0; 1.0 0.0]
    psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    applygate!(psi, X, 1, 1)
    @test psi.Γ[1][1,:,1] ≈ [0, 1] atol=1e-12
    @test psi.Γ[2][1,:,1] ≈ [0, 1] atol=1e-12
end

@testset "TENSOR_UMUL_INPLACE_CORRECTNESS" begin
    G = ComplexF64[0 1; 1 0]
    Γ = rand(ComplexF64, 3, 2, 3)
    Γ_orig = copy(Γ)
    Γ_new = InfiniteTEBD.tensor_umul(G, Γ)
    InfiniteTEBD.tensor_umul!(G, Γ)
    @test Γ ≈ Γ_new atol=1e-12
    @test Γ ≈ InfiniteTEBD.tensor_umul(G, Γ_orig) atol=1e-12
end

@testset "APPLYGATE_ONE_SITE_PATH_RETURNS_EMPTY_STATS" begin
    P = pauli_matrices()
    ψ = product_iMPS(ComplexF64, [[1, 0]])

    ψ_after, stats = applygate!(ψ, P.X, 1, 1; return_stats=true)

    @test ψ_after === ψ
    @test InfiniteTEBD.expect(ψ, P.Z, 1, 1) ≈ -1.0 atol=1e-12
    @test stats.support == (1, 1)
    @test isempty(stats.bond_stats)
    @test stats.max_discarded_weight == 0.0
    @test stats.num_saturated == 0
end

@testset "EVOLVE_RETURN_STATS_AGGREGATES_BOND_UPDATES" begin
    # evolve! with return_stats=true should aggregate per-gate stats and
    # report max_discarded_weight, mean_discarded_weight, and max_kept_dim
    # consistent with the underlying gate updates. Previously these
    # aggregate fields had no direct test.
    P = pauli_matrices()
    G = exp(-0.1im * kron(P.X, P.X))
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    _, stats = evolve!(ψ, [(G, 1, 2), (G, 2, 1)], 3;
        maxdim=4, truncerr=1e-12, return_stats=true)

    @test stats.max_discarded_weight >= 0.0
    @test stats.mean_discarded_weight >= 0.0
    @test stats.max_discarded_weight >= stats.mean_discarded_weight
    @test stats.max_kept_dim <= 4
    @test stats.num_saturated >= 0
    @test !isempty(stats.gate_updates)
    # Per-gate stats should be the typed BondStat objects, not Any.
    @test eltype(first(stats.gate_updates).bond_stats) === InfiniteTEBD.BondStat
end

@testset "APPLYGATE_NORMALIZES_PERIODIC_SITE_INDICES" begin
    P = pauli_matrices()
    n = 3
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1], [1, 0]])

    ψ_after, stats = applygate!(ψ, P.X, n + 1, n + 1; return_stats=true)

    @test ψ_after === ψ
    @test InfiniteTEBD.expect(ψ, P.Z, 1, 1) ≈ -1.0 atol=1e-12
    @test stats.support == (1, 1)
    @test isempty(stats.bond_stats)
end

@testset "APPLYGATE_IDENTITY_PRESERVES_PRODUCT_STATE" begin
    ψ0 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    applygate!(ψ, Matrix{ComplexF64}(I, 4, 4), 1, 2; maxdim=4)

    @test InfiniteTEBD.inner_product(ψ, ψ0) ≈ 1.0 atol=1e-12
    @test maximum(length.(ψ.λ)) == 1
end

@testset "APPLYGATE_WRAPAROUND_INVERSE_RESTORES_STATE" begin
    G = bell_gate()
    ψ0 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])

    applygate!(ψ, G, 2, 1; maxdim=4)
    applygate!(ψ, adjoint(G), 2, 1; maxdim=4)

    @test InfiniteTEBD.inner_product(ψ, ψ0) ≈ 1.0 atol=1e-12
    @test maximum(length.(ψ.λ)) == 1
end

@testset "APPLYGATE_NORMALIZES_WRAPPED_MULTI_SITE_INDICES" begin
    G = bell_gate()
    n = 3
    ψ_direct = product_iMPS(ComplexF64, [[1, 0], [1, 0], [0, 1]])
    ψ_wrapped = deepcopy(ψ_direct)

    applygate!(ψ_direct, G, 3, 1; maxdim=4)
    _, stats = applygate!(ψ_wrapped, G, 0, n + 1; maxdim=4, return_stats=true)

    @test stats.support == (3, 1)
    @test length.(ψ_wrapped.λ) == length.(ψ_direct.λ)
    @test InfiniteTEBD.inner_product(ψ_wrapped, ψ_direct) ≈ 1.0 atol=1e-12
end

@testset "APPLYGATE_WRAPAROUND_RESTORES_CANONICAL_FORM" begin
    # Wrap-around gates used to leave the iMPS non-canonical because the
    # local SVD inside tensor_applygate! only canonicalizes the affected
    # block, not the wraparound seam to the rest of the cell. Verify the
    # post-gate state passes the right-canonical invariant.
    Random.seed!(2026_05_25)
    ψ = rand_iMPS(ComplexF64, 4, 2, 4)
    canonical!(ψ)
    @test right_canonical_error(ψ) < 1e-10  # baseline

    Z = ComplexF64[1 0; 0 -1]
    X = ComplexF64[0 1; 1 0]
    G = exp(-0.1im * kron(Z, X))
    applygate!(ψ, G, 4, 1)  # j < i → wraparound

    @test right_canonical_error(ψ) < 1e-10
    @test InfiniteTEBD.inner_product(ψ) ≈ 1.0 atol=1e-10
end

@testset "APPLYGATE_SINGLE_SITE_RENORMALIZES_NON_UNITARY_GATE" begin
    # The single-site path used to silently ignore renormalize=true. A
    # non-unitary 1-site gate (e.g. an imaginary-time term exp(-dτ h))
    # would leave the per-cell norm drifted to exp(-2dτ⟨h⟩) instead of 1.
    P = pauli_matrices()
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])  # site 2 is |↓⟩
    G = exp(-0.5 * P.Z)  # |↓⟩ has Z=-1, so G|↓⟩ = exp(0.5)|↓⟩
    applygate!(ψ, G, 2, 2; renormalize=true)
    @test InfiniteTEBD.inner_product(ψ) ≈ 1.0 atol=1e-12

    # renormalize=false preserves the un-normalized magnitude so the caller
    # can rescale themselves; verify the documented opt-out works.
    ψ2 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    applygate!(ψ2, G, 2, 2; renormalize=false)
    @test InfiniteTEBD.inner_product(ψ2) ≈ exp(1) atol=1e-12  # |exp(0.5)|² = e
end

@testset "LEGACY_CUTOFF_KEYWORD_IS_ACCEPTED" begin
    G = bell_gate()
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])

    # Default behaviour: both nothing -> SVDTOL.
    @test InfiniteTEBD._resolve_svd_min(nothing, nothing) == InfiniteTEBD.SVDTOL
    # Explicit svd_min, no cutoff.
    @test InfiniteTEBD._resolve_svd_min(InfiniteTEBD.SVDTOL, nothing) == InfiniteTEBD.SVDTOL
    @test InfiniteTEBD._resolve_svd_min(1.5e-10, nothing) == 1.5e-10
    # Cutoff alone resolves to the cutoff value.
    @test InfiniteTEBD._resolve_svd_min(nothing, 0.0) == 0.0
    @test InfiniteTEBD._resolve_svd_min(nothing, 1.5e-10) == 1.5e-10
    # Passing both is always an error, even if svd_min equals the historical
    # default — the previous behaviour silently dropped svd_min, which made
    # it impossible to tell whether the caller intended SVDTOL or had a typo.
    @test_throws ArgumentError InfiniteTEBD._resolve_svd_min(InfiniteTEBD.SVDTOL, 0.0)
    @test_throws ArgumentError InfiniteTEBD._resolve_svd_min(1.5e-10, 2e-12)
    @test applygate!(ψ, G, 1, 2; maxdim=4, cutoff=0.0) === ψ
    @test maximum(length.(ψ.λ)) == 2
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; cutoff=1e-12, svd_min=2e-12)
end

@testset "EVOLVE_ZERO_STEPS_IS_NOOP_WITH_EMPTY_STATS" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    before_Γ = deepcopy(ψ.Γ)
    before_λ = deepcopy(ψ.λ)

    ψ_after, stats = evolve!(
        ψ,
        [(Matrix{ComplexF64}(I, 4, 4), 1, 2)],
        0;
        return_stats=true
    )

    @test ψ_after === ψ
    @test ψ.Γ == before_Γ
    @test ψ.λ == before_λ
    @test isempty(stats.gate_updates)
    @test stats.max_discarded_weight == 0.0
    @test stats.mean_discarded_weight == 0.0
    @test stats.max_kept_dim == 0
    @test stats.num_saturated == 0
end

@testset "LEGACY_ADAPTIVE_EVOLVE_KEYWORDS_ARE_ACCEPTED" begin
    G = bell_gate()
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])

    @test evolve!(
        ψ,
        [(G, 1, 2)],
        1;
        maxdim=4,
        mindim=1,
        cutoff=0.0,
        chi_policy=:adaptive,
        q=1.0,
        alpha=0.1,
    ) === ψ
    @test maximum(length.(ψ.λ)) == 2
    @test all(length(λ) <= 4 for λ in ψ.λ)

    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; chi_policy=:unknown)
end

@testset "PUBLIC_GATE_VALIDATION_ERRORS" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    G = Matrix{ComplexF64}(I, 4, 4)

    @test_throws ArgumentError applygate!(ψ, G, 1, 2; maxdim=0)
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; mindim=0)
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; maxdim=1, mindim=2)
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; truncerr=-1e-3)
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; svd_min=-1e-3)

    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], -1)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; maxdim=0)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; mindim=0)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; maxdim=1, mindim=2)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; truncerr=-1e-3)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; svd_min=-1e-3)
end

@testset "CONVERT_OPERATOR_ORDERING" begin
    M = reshape(collect(1.0:16.0), 4, 4)
    I4 = Matrix{Float64}(I, 4, 4)
    C = InfiniteTEBD.convert_operator(M, 2, 2)

    @test size(C) == (4, 4)
    @test InfiniteTEBD.convert_operator(C, 2, 2) == M
    @test InfiniteTEBD.convert_operator([1.0 2.0; 3.0 4.0], 2, 1) == [1.0 2.0; 3.0 4.0]
end
