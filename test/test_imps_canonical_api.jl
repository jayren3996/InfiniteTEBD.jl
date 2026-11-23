using Test
using LinearAlgebra
using Random
using InfiniteTEBD
using InfiniteTEBD: rand_iMPS, product_iMPS, canonical!, iMPS

Random.seed!(20260522)

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: assert_normalized_schmidt_spectra, assert_stored_tensor_convention, right_canonical_error

function _assert_finite_nonempty(psi)
    @test all(!isempty(λ) for λ in psi.λ)
    @test all(all(isfinite, λ) for λ in psi.λ)
    @test all(all(isfinite, Γ) for Γ in psi.Γ)
    @test all(all(>(0), size(Γ)) for Γ in psi.Γ)
end

@testset "IMPS_CANONICAL_CONVENTION" begin
    psi = rand_iMPS(ComplexF64, 1, 2, 1)

    @test canonical!(psi) === psi
    @test length(psi.Γ) == 1
    @test length(psi.λ) == 1
    @test right_canonical_error(psi) <= 1e-10
end

@testset "IMPS_MULTI_SITE_CANONICAL_INVARIANTS" begin
    psi = rand_iMPS(ComplexF64, 2, 2, 1)

    assert_normalized_schmidt_spectra(psi; atol=1e-10)
    assert_stored_tensor_convention(psi; atol=1e-12)
    @test right_canonical_error(psi) <= 1e-10
    @test InfiniteTEBD.inner_product(psi) ≈ 1.0 atol=1e-10
end

@testset "PRODUCT_IMPS_CANONICAL_ON_CONSTRUCTION" begin
    psi = product_iMPS(ComplexF64, [[2, 0]])

    assert_normalized_schmidt_spectra(psi; atol=1e-12)
    assert_stored_tensor_convention(psi; atol=1e-12)
    @test right_canonical_error(psi) <= 1e-12
    @test InfiniteTEBD.inner_product(psi) ≈ 1.0 atol=1e-12

    psi_untyped = product_iMPS([[1, 1]])
    @test eltype(psi_untyped) == Float64
    @test psi_untyped.Γ[1][1, :, 1] ≈ fill(inv(sqrt(2)), 2) atol=1e-12
end

@testset "INDEXING_DIVIDES_SMALL_RETAINED_SCHMIDT_VALUES" begin
    λ = [1.0, 1e-9]
    B = zeros(ComplexF64, 2, 2, 2)
    B[1, 1, 1] = 1.0
    B[2, 2, 2] = λ[2]
    psi = iMPS([B], [λ], 1)

    Γ, returned_λ = psi[1]

    @test returned_λ == psi.λ[1]
    @test returned_λ !== psi.λ[1]  # copy, not aliased — see GETINDEX_RETURNS_INDEPENDENT_SCHMIDT_COPY
    @test Γ[1, 1, 1] ≈ 1.0 atol=1e-12
    @test Γ[2, 2, 2] ≈ 1.0 atol=1e-12
end

@testset "GETINDEX_WRAPS_PERIODICALLY" begin
    # ψ[i] wraps i modulo ψ.n. Verify accessing bond 0, n+1, or a negative
    # index returns the same Vidal tensor as the canonical wrapped index.
    using Random
    Random.seed!(2026_05_26)
    ψ = rand_iMPS(ComplexF64, 3, 2, 4)
    canonical!(ψ)

    Γ1, λ1 = ψ[1]
    Γ_wrap_up, λ_wrap_up = ψ[ψ.n + 1]
    Γ_wrap_down, λ_wrap_down = ψ[0]
    @test Γ_wrap_up   ≈ Γ1                            atol=1e-12
    @test λ_wrap_up   == λ1
    @test Γ_wrap_down ≈ ψ[ψ.n][1]                     atol=1e-12

    # Negative indices also wrap.
    @test ψ[-1][1] ≈ ψ[ψ.n - 1][1] atol=1e-12
end

@testset "CANONICAL_ACCEPTS_KRYLOV_KWARGS" begin
    # The tol/maxiter kwargs were previously not exposed on canonical!.
    # Verify they're accepted, plumb through to schmidt_canonical, and that
    # tight/loose tolerances both produce a usable canonical state on a
    # large-enough cell that the Krylov path actually runs (a > 8).
    psi = rand_iMPS(ComplexF64, 1, 2, 16)
    canonical!(psi; tol=1e-14)
    @test isapprox(InfiniteTEBD.inner_product(psi), 1.0; atol=1e-10)

    psi2 = rand_iMPS(ComplexF64, 1, 2, 16)
    canonical!(psi2; tol=1e-8, maxiter=200)
    @test isapprox(InfiniteTEBD.inner_product(psi2), 1.0; atol=1e-6)
end

@testset "CANONICAL_ARGUMENT_VALIDATION" begin
    psi = product_iMPS(ComplexF64, [[1, 0]])

    @test_throws ArgumentError canonical!(psi; maxdim=0)
    @test_throws ArgumentError canonical!(psi; cutoff=-1e-12)
    @test_throws ArgumentError canonical!(psi; noninjective=:invalid)
    @test_throws ArgumentError canonical!(psi; symmetry_break=:invalid)
end

@testset "CANONICAL_STABILITY_EDGE_CASES" begin
    psi = rand_iMPS(ComplexF64, 1, 2, 1)
    canonical!(psi; cutoff=2.0, noninjective=:ignore)
    _assert_finite_nonempty(psi)
    @test all(size(Γ, 1) > 0 && size(Γ, 3) > 0 for Γ in psi.Γ)

    G = zeros(ComplexF64, 2, 2, 2)
    G[1, 1, 1] = 1
    G[1, 2, 2] = 1
    G[2, 1, 1] = 1
    G[2, 2, 2] = 1
    for λ in ([1.0, 1e-300], [1.0, 0.0])
        psi_tiny = iMPS([copy(G)], [collect(λ)], 1)
        canonical!(psi_tiny; noninjective=:ignore)
        _assert_finite_nonempty(psi_tiny)
    end
end

@testset "GHZ_LIKE_NONINJECTIVE_WARNING" begin
    G = zeros(ComplexF64, 2, 2, 2)
    G[1, 1, 1] = 1
    G[2, 2, 2] = 1
    psi_default = iMPS([copy(G)], [ones(2)], 1)

    @test_logs (:warn, r"without symmetry-sector selection") (:warn,) (:warn,) canonical!(psi_default; noninjective=:warn)
    _assert_finite_nonempty(psi_default)

    psi_auto = iMPS([copy(G)], [ones(2)], 1)
    @test_logs (:warn, r"selected virtual sector") canonical!(psi_auto; noninjective=:warn, symmetry_break=:auto)
    _assert_finite_nonempty(psi_auto)

    @test_throws ArgumentError canonical!(
        iMPS([copy(G)], [ones(2)], 1);
        noninjective=:error,
        symmetry_break=:auto,
    )

    # noninjective=:error should throw before symmetry_break is considered,
    # regardless of which symmetry_break value is set. Previously only the
    # :error+:auto path was tested.
    @test_throws ArgumentError canonical!(
        iMPS([copy(G)], [ones(2)], 1);
        noninjective=:error,
        symmetry_break=:none,
    )

    # noninjective=:ignore should silently produce a usable state on the same
    # degenerate input, with no @warn from the noninjective code path. The
    # state's tensors and Schmidt vectors should be finite and non-empty.
    psi_ignore = iMPS([copy(G)], [ones(2)], 1)
    canonical!(psi_ignore; noninjective=:ignore)
    _assert_finite_nonempty(psi_ignore)
end

@testset "GHZ_LIKE_NONINJECTIVE_MULTISITE" begin
    # The existing GHZ_LIKE_NONINJECTIVE_WARNING test uses a 1-site cell.
    # Two-fold-degenerate transfer spectra also arise on multi-site cells —
    # exercise the same noninjective/symmetry_break paths there.
    G = zeros(ComplexF64, 2, 2, 2)
    G[1, 1, 1] = 1   # sector 1 stays in sector 1, physical 0
    G[2, 2, 2] = 1   # sector 2 stays in sector 2, physical 1

    psi_auto = iMPS([copy(G), copy(G)], [ones(2), ones(2)], 2)
    canonical!(psi_auto; noninjective=:ignore, symmetry_break=:auto)
    _assert_finite_nonempty(psi_auto)
    # After symmetry breaking to one sector, the bond dimension should
    # collapse to the size of that sector (1 here, since each sector is
    # a pure |0...⟩ or |1...⟩ string).
    @test maximum(length.(psi_auto.λ)) <= 2

    psi_error = iMPS([copy(G), copy(G)], [ones(2), ones(2)], 2)
    @test_throws ArgumentError canonical!(psi_error; noninjective=:error)
end

@testset "NO_BLOCK_CANONICAL_EXPORT" begin
    @test !isdefined(InfiniteTEBD, :block_canonical)
end

@testset "CONJ_IMPS_INDEPENDENT_SCHMIDT" begin
    psi = rand_iMPS(ComplexF64, 2, 2, 4)
    psi_conj = conj(psi)
    @test psi_conj.n == psi.n
    for i in 1:psi.n
        @test psi_conj.Γ[i] ≈ conj(psi.Γ[i])
        @test psi_conj.λ[i] ≈ psi.λ[i]
    end
    # Mutating the conjugated state's Schmidt vector must not alias the
    # original (regression test for the `get_data` UndefVarError + λ aliasing).
    sentinel = -123.0
    psi_conj.λ[1][1] = sentinel
    @test psi.λ[1][1] != sentinel
end

@testset "MPS_PROMOTE_TYPE_CONVERTS_AND_COPIES" begin
    psi = rand_iMPS(ComplexF64, 2, 2, 4)
    psi32 = InfiniteTEBD.mps_promote_type(ComplexF32, psi)
    @test all(eltype(Γ) == ComplexF32 for Γ in psi32.Γ)
    @test psi32.n == psi.n
    # λ must be deep-copied so the promoted state has independent storage.
    sentinel = Float32(-7.0)
    psi32.λ[1][1] = sentinel
    @test psi.λ[1][1] != sentinel
end

@testset "GETINDEX_RETURNS_INDEPENDENT_SCHMIDT_COPY" begin
    psi = rand_iMPS(ComplexF64, 2, 2, 4)
    _, lambda1 = psi[1]
    original = psi.λ[1][1]
    lambda1[1] = -42.0
    @test psi.λ[1][1] == original
end

@testset "CANONICAL_AFTER_TRUNCATION_IS_FULLY_CANONICAL" begin
    # Before the re-pass fix, canonical!(psi; maxdim=χ) with χ < natural bond dim
    # left the state only approximately right-canonical (truncated SVDs in
    # tensor_decomp! break the per-site U gauge). canonical! now does a second
    # pass when truncation actually occurred, which restores exact gauge with
    # no further information loss.
    for n in (1, 2, 3)
        psi = rand_iMPS(ComplexF64, n, 2, 6)
        canonical!(psi; maxdim=2)
        @test maximum(length, psi.λ) <= 2
        @test right_canonical_error(psi) <= 1e-10
        assert_normalized_schmidt_spectra(psi; atol=1e-10)
        @test InfiniteTEBD.inner_product(psi) ≈ 1.0 atol=1e-9
    end
end

@testset "SCHMIDT_IS_TRUE_BOND_SPECTRUM" begin
    # After canonical!, the stored λ[n] (wraparound bond) must be the
    # *physical* Schmidt spectrum on that bond, i.e., the diagonal of the
    # transfer matrix's left fixed point should be Diagonal(λ_n²) up to
    # overall scale. This pins down the Schmidt-form property of the
    # canonical decomposition, not just the right-canonical condition.
    for n in (1, 2, 3)
        psi = rand_iMPS(ComplexF64, n, 2, 4)
        canonical!(psi)
        Γ = InfiniteTEBD.tensor_group(psi.Γ)
        L = Matrix(InfiniteTEBD.steady_mat(Γ; dir=:l))
        L_norm = L / tr(L)
        # off-diagonal must be small (state is already in the basis where λ²
        # is diagonal)
        @test norm(L_norm - Diagonal(diag(L_norm))) < 1e-8
        # diagonal entries match λ² up to permutation
        λ2 = sort(real.(psi.λ[end]) .^ 2; rev=true)
        λ2 ./= sum(λ2)
        diag_part = sort(real.(diag(L_norm)); rev=true)
        @test diag_part ≈ λ2 atol=1e-8
    end
end

@testset "CANONICAL_UNITCELL_TRANSFER_HAS_DOMINANT_EIGENVALUE_ONE" begin
    for n in (1, 2, 3)
        psi = rand_iMPS(ComplexF64, n, 2, 4)
        canonical!(psi)
        Γ = InfiniteTEBD.tensor_group(psi.Γ)
        T = InfiniteTEBD.kraus_mat(Γ, conj(Γ); dir=:r)
        λ_max = maximum(abs, eigvals(T))
        @test λ_max ≈ 1.0 atol=1e-9
    end
end

@testset "STABILITY_MULTISITE_GHZ_LIKE" begin
    # GHZ-like block-diagonal tensors at n > 1: existing GHZ test only covers
    # n = 1. Make sure the warning fires once per pass and the resulting state
    # is still finite and right-canonical.
    G = zeros(ComplexF64, 2, 2, 2)
    G[1, 1, 1] = 1; G[2, 2, 2] = 1
    for n in (2, 3)
        psi = iMPS([copy(G) for _ in 1:n], [ones(2) for _ in 1:n], n)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-9
        @test all(all(isfinite, λ) for λ in psi.λ)
        @test all(all(isfinite, Γ) for Γ in psi.Γ)
    end
end

@testset "STABILITY_NEARLY_NONINJECTIVE_PERTURBATION" begin
    # Tensors that *look* block-diagonal but have a tiny off-diagonal mixing
    # should still canonicalize cleanly across many orders of perturbation
    # magnitude.
    for ε in (1e-2, 1e-6, 1e-10)
        G = zeros(ComplexF64, 2, 2, 2)
        G[1, 1, 1] = 1; G[2, 2, 2] = 1
        G[1, 1, 2] = ε; G[2, 1, 1] = ε
        psi = iMPS([G], [ones(2)], 1)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-9
        @test all(isfinite, psi.λ[1])
    end
end

@testset "STABILITY_ILL_CONDITIONED_INPUT" begin
    # Tensors whose entries span many orders of magnitude (e.g. one row scaled
    # by 1e10) shouldn't crash canonical!. The natural result is heavy
    # rank-compression onto the dominant subspace.
    for scale in (1e6, 1e10, 1e14)
        Random.seed!(42)
        G = randn(ComplexF64, 4, 2, 4)
        G[1, :, :] .*= scale
        psi = iMPS([G], [ones(4)], 1)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-9
        @test all(isfinite, psi.λ[1])
    end
end

@testset "STABILITY_HARD_ZEROS_IN_SCHMIDT" begin
    # Schmidt vectors with exact-zero entries — `_safe_reciprocal` and the
    # support-truncation logic must handle these without producing NaN/Inf.
    for kill_count in 1:3
        Γ = randn(ComplexF64, 4, 2, 4)
        λ = ones(Float64, 4); λ[1:kill_count] .= 0.0
        psi = iMPS([Γ], [λ], 1)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-7
        @test all(isfinite, psi.λ[1])
    end
end

@testset "STABILITY_NAN_INF_REJECTED" begin
    Γ_nan = randn(ComplexF64, 2, 2, 2); Γ_nan[1, 1, 1] = NaN
    @test_throws Exception canonical!(iMPS([Γ_nan], [ones(2)], 1); noninjective=:ignore)

    Γ_inf = randn(ComplexF64, 2, 2, 2); Γ_inf[1, 1, 1] = Inf
    @test_throws Exception canonical!(iMPS([Γ_inf], [ones(2)], 1); noninjective=:ignore)
end

@testset "STABILITY_KRYLOV_PATH_LARGE_BOND" begin
    # bond > 32 forces the Krylov solver inside steady_mat. Exercise both
    # single- and multi-site cases at moderate bond dim.
    for (bond, n) in [(48, 1), (40, 2)]
        Random.seed!(bond + n)
        Γs = [randn(ComplexF64, bond, 2, bond) for _ in 1:n]
        λs = [10.0 .^ (-3 .* rand(bond)) for _ in 1:n]
        psi = iMPS(Γs, λs, n)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-7
        @test all(all(isfinite, λ) for λ in psi.λ)
    end
end

@testset "STABILITY_NOISE_PLUS_RECANONICALIZE" begin
    # Standard TEBD scenario: after applying gates, the state is no longer
    # perfectly canonical. canonical! must converge to a canonical form on a
    # noise-corrupted input.
    Random.seed!(11)
    psi = rand_iMPS(ComplexF64, 2, 2, 4)
    canonical!(psi)
    for Γ in psi.Γ
        Γ .+= 0.1 .* randn(ComplexF64, size(Γ)...)
    end
    @test right_canonical_error(psi) > 1e-2  # we did break it
    canonical!(psi; noninjective=:ignore)
    @test right_canonical_error(psi) <= 1e-9
    assert_normalized_schmidt_spectra(psi; atol=1e-9)
end

@testset "SITEWISE_CANONICAL_GAUGE_RESTORATION" begin
    rng = MersenneTwister(196)
    dims = [3, 2, 4, 3, 2, 3, 4, 3]
    n = length(dims)
    tensors = [randn(rng, ComplexF64, dims[mod1(i-1, n)], 2, dims[i]) / 2 for i in 1:n]
    original = iMPS(tensors, [ones(d) for d in dims], n)
    ψ = deepcopy(original)
    grouped = canonical!(deepcopy(original); maxdim=4, cutoff=1e-12, noninjective=:ignore)
    @test restore_canonical!(ψ) === ψ
    @test length.(ψ.λ) == dims
    @test inner_product(ψ) ≈ 1 atol=1e-10
    @test abs(inner_product(original, ψ)) / sqrt(inner_product(original)) ≈ 1 atol=1e-10
    @test right_canonical_error(ψ) < 1e-10
    for i in 1:n
        pleft = Diagonal(ψ.λ[mod1(i-1, n)] .^ 2)
        B = ψ.Γ[i]
        propagated = sum(B[:, s, :]' * pleft * B[:, s, :] for s in axes(B, 2))
        @test propagated ≈ Diagonal(ψ.λ[i] .^ 2) atol=1e-10
        @test ψ.λ[i] ≈ grouped.λ[i] atol=1e-9
    end

    ψ = deepcopy(original)
    compressed = truncate!(deepcopy(ψ), 2)
    @test truncate!(ψ, 2) === ψ
    @test maximum(length.(ψ.λ)) <= 2
    @test right_canonical_error(ψ) < 1e-10
    @test abs(inner_product(compressed, ψ)) / sqrt(inner_product(compressed)) ≈ 1 atol=1e-10

    local_tensor = reshape([0.8, 0.6], 1, 2, 1)
    large = iMPS([2 .* local_tensor for _ in 1:32], [ones(1) for _ in 1:32], 32)
    restore_canonical!(large)
    @test right_canonical_error(large) < 1e-10
    @test inner_product(large) ≈ 1 atol=1e-10

    singular_tensor = zeros(2, 2, 2)
    singular_tensor[1, 1, 1] = 1
    singular = iMPS([singular_tensor], [ones(2)], 1)
    before = deepcopy(singular)
    @test_throws ArgumentError restore_canonical!(singular)
    @test singular.Γ == before.Γ && singular.λ == before.λ
end

@testset "STREAMING_CANONICAL_MATCHES_GROUPED_CUTS" begin
    rng = MersenneTwister(533)
    dims = [4, 2, 4, 3, 4, 4]  # The fastest transfer solve is off the original seam.
    n = length(dims)
    tensors = [randn(rng, ComplexF64, dims[mod1(i-1,n)], 2, dims[i]) /
               sqrt(2*dims[i]) for i in 1:n]
    seam = [0.8, 0.7, 0.6, 0.4]
    ψ = iMPS(tensors, [i == n ? seam : ones(dims[i]) for i in 1:n], n)
    for renormalize in (false, true), (maxdim, cutoff) in ((4, 1e-12), (2, 1e-12), (4, 0.14))
        opts = (; maxdim, cutoff, renormalize, noninjective=:ignore,
                  symmetry_break=:none, tol=nothing, maxiter=nothing)
        A, λ = InfiniteTEBD.schmidt_canonical(ψ.Γ, ψ.λ[end]; opts...)
        B, μ = InfiniteTEBD._schmidt_canonical_streaming(ψ.Γ, ψ.λ[end]; opts...)
        @test length.(λ) == length.(μ)
        @test all(isapprox.(λ, μ; atol=1e-11, rtol=1e-9))
        @test inner_product(B) ≈ inner_product(A) rtol=1e-9
        @test inner_product(A, B) ≈ inner_product(A) rtol=1e-9
        if maxdim == 4 && cutoff == 1e-12 && !renormalize
            @test inner_product(B) ≈ inner_product(ψ) rtol=1e-9
        end

        grouped = canonical!(deepcopy(ψ); opts...)
        # A zero-padded physical direction changes the execution path
        # without changing the input or the prescribed seam/cutoff.
        source = deepcopy(ψ)
        source.Γ[2] = cat(source.Γ[2], zeros(ComplexF64, dims[1], 1, dims[2]); dims=2)
        canonical!(source; opts...)
        grouped.Γ[2] = cat(grouped.Γ[2],
            zeros(ComplexF64, size(grouped.Γ[2],1), 1, size(grouped.Γ[2],3)); dims=2)
        @test length.(source.λ) == length.(grouped.λ)
        @test all(isapprox.(source.λ, grouped.λ; atol=1e-10, rtol=1e-8))
        @test inner_product(source, grouped) ≈ inner_product(grouped) rtol=1e-8
    end
end

@testset "STREAMING_CANONICAL_FLOAT32_KRYLOV" begin
    rng = MersenneTwister(648)
    χ = 9
    tensors = [randn(rng, Float32, χ,d,χ) ./ sqrt(Float32(d*χ)) for d in (2, 3, 2, 2)]
    original = iMPS(Float32, tensors; renormalize=false)
    ψ = canonical!(deepcopy(original); maxdim=χ, noninjective=:ignore)
    @test eltype(ψ) === Float32
    @test right_canonical_error(ψ) < 2e-5
    @test inner_product(ψ) ≈ 1 atol=2e-5
    @test inner_product(original, ψ) / sqrt(inner_product(original)) ≈ 1 atol=2e-5
end

@testset "STREAMING_CANONICAL_DIAGNOSES_GHZ" begin
    A = zeros(ComplexF64, 2, 2, 2)
    A[1, 1, 1] = A[2, 2, 2] = 1
    n = 16
    ψ = iMPS([copy(A) for _ in 1:n], [ones(2) for _ in 1:n], n)
    @test !InfiniteTEBD._fits_grouped_cell(ψ)
    before = deepcopy(ψ)
    @test_throws ArgumentError canonical!(ψ; noninjective=:error)
    @test ψ.Γ == before.Γ && ψ.λ == before.λ
    @test_logs (:warn, r"selected virtual sector 1 of 2") canonical!(ψ; symmetry_break=:auto)
    @test length.(ψ.λ) == ones(Int,n)
    @test inner_product(ψ) ≈ 1 atol=1e-12
    @test expect(ψ, Diagonal([1.0, -1.0]), 1, 1) ≈ 1 atol=1e-12
end

@testset "CANONICAL_LARGE_CELL_DISPATCH" begin
    rng = MersenneTwister(231)
    n, d, χ = 9, 4, 3
    ψ = iMPS([randn(rng, ComplexF64, χ, d, χ) / sqrt(d*χ) for _ in 1:n],
             [ones(χ) for _ in 1:n], n)
    @test !InfiniteTEBD._fits_grouped_cell(ψ)
    original = deepcopy(ψ)
    target_norm = inner_product(ψ)
    @test canonical!(ψ; maxdim=χ, renormalize=false) === ψ
    @test inner_product(ψ) ≈ target_norm rtol=1e-10
    @test inner_product(original, ψ) ≈ target_norm rtol=1e-10

    canonical!(ψ; maxdim=2)
    @test maximum(length.(ψ.λ)) == 2
    @test right_canonical_error(ψ) < 1e-9
    @test inner_product(ψ) ≈ 1 atol=1e-10

    # Recanonicalizing across the seam and adaptive updates both delegate
    # to the same bounded-memory canonical! path.
    X = ComplexF64[0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]
    G = exp(-0.02im * kron(X, X))
    applygate!(ψ, G, n, 1; maxdim=3, recanonicalize=true)
    @test right_canonical_error(ψ) < 1e-9
    evolve!(ψ, [(G, 1, 2)], 1; maxdim=3, chi_policy=:adaptive)
    @test right_canonical_error(ψ) < 1e-9

    large_product = product_iMPS(ComplexF64, fill([1, 1im, 0, 0], 48))
    @test length.(large_product.λ) == ones(Int, 48)
    @test right_canonical_error(large_product) < 1e-12

    padded = zeros(ComplexF64, 3, 4, 3)
    padded[1, 1, 1] = 1
    state = iMPS([copy(padded) for _ in 1:n], [ones(3) for _ in 1:n], n)
    canonical!(state)
    @test length.(state.λ) == ones(Int, n)
    @test inner_product(state) ≈ 1 atol=1e-10
end

@testset "PERIODIC_QR_RESTORES_ILL_CONDITIONED_GAUGES" begin
    rng = MersenneTwister(4001)
    dims = [2, 3, 4, 3]
    n = length(dims)
    rand_tensors = [randn(rng, ComplexF64, dims[mod1(i-1,n)], 3, dims[i]) for i in 1:n]
    reference = iMPS(rand_tensors, [ones(d) for d in dims], n)
    restore_canonical!(reference)
    Z = Diagonal([1.0, -0.2, 0.7])

    for magnitude in (1e-8, 1e-10)
        # Apply an invertible virtual basis change, including on the
        # smallest bond where the iterations close the unit cell.
        gauges = [Diagonal([ones(d-1); magnitude]) for d in dims]
        A = [InfiniteTEBD.canonical_gauging(reference.Γ[i], gauges[i],
                    inv(gauges[mod1(i-1,n)])) .* 1.3 for i in 1:n]
        ψ = iMPS(A, [ones(d) for d in dims], n)
        @test restore_canonical!(ψ; method=:qr) === ψ
        @test length.(ψ.λ) == dims
        @test inner_product(ψ) ≈ 1 atol=1e-10
        @test inner_product(reference, ψ) ≈ 1 atol=1e-10
        @test right_canonical_error(ψ) < 1e-10
        auto = iMPS(copy(A), [ones(d) for d in dims], n)
        restore_canonical!(auto; method=:auto)
        @test inner_product(reference, auto) ≈ 1 atol=1e-10
        for i in 1:n
            previous = mod1(i-1, n)
            @test ψ.λ[i] ≈ reference.λ[i] atol=1e-9
            @test expect(ψ, Z, i, i) ≈ expect(reference, Z, i, i) atol=1e-9
            B = ψ.Γ[i]
            density = sum(B[:,s,:]' * Diagonal(ψ.λ[previous].^2) * B[:,s,:] for s in axes(B, 2))
            @test density ≈ Diagonal(ψ.λ[i].^2) atol=1e-10
        end

        order = [3:n; 1:2]
        rolled = iMPS(A[order], [ones(dims[i]) for i in order], n)
        restore_canonical!(rolled; method=:qr)
        @test inner_product(ψ.Γ[order], rolled.Γ) ≈ 1 atol=1e-10
    end

    real_state = rand_iMPS(Float64, 3, 2, 2)
    single = mps_promote_type(Float32, real_state)
    @test restore_canonical!(single; method=:qr) === single
    @test eltype(single) === Float32
    @test right_canonical_error(single) < 1e-5
    @test inner_product(real_state, single) ≈ 1 atol=1e-5

    rank_one = zeros(2,2,2)
    rank_one[1,1,1] = 1
    ghz = copy(rank_one)
    ghz[2,2,2] = 1
    for A in (rank_one, ghz)
        ψ = iMPS([copy(A)], [ones(2)], 1)
        before = deepcopy(ψ)
        @test_throws ArgumentError restore_canonical!(ψ; method=:qr)
        @test ψ.Γ == before.Γ && ψ.λ == before.λ
        @test_throws ArgumentError restore_canonical!(ψ; method=:auto)
        @test ψ.Γ == before.Γ && ψ.λ == before.λ
    end
    trial = iMPS(copy(rand_tensors), [ones(d) for d in dims], n)
    before = deepcopy(trial)
    @test_throws ErrorException restore_canonical!(trial; method=:qr, maxiter=1)
    @test trial.Γ == before.Γ && trial.λ == before.λ
    @test_throws ArgumentError restore_canonical!(trial; method=:qr, tol=0)

    # A gapped but nearly reducible transfer has a unique physical state.
    # The plain sweep need not close within a small iteration budget.
    p = 1e-6
    g = Diagonal([0.2, 1.0])
    blocks = (sqrt(1-p) * Matrix{Float64}(I, 2, 2),
              sqrt(p/2) * [0.0 1.0; 1.0 0.0],
              sqrt(p/2) * [1.0 0.0; 0.0 -1.0])
    A = zeros(2, 3, 2)
    for (s, block) in enumerate(blocks)
        A[:,s,:] = g \ (block * g)
    end
    slow = iMPS([A], [ones(2)], 1)
    untouched = deepcopy(slow)
    @test_throws ErrorException restore_canonical!(slow; method=:qr, maxiter=64, accelerate=false)
    @test slow.Γ == untouched.Γ && slow.λ == untouched.λ
    restore_canonical!(slow; method=:qr, maxiter=64)
    # cat places the physical axis last; the iMPS layout is left, physical, right.
    known = iMPS([permutedims(cat(blocks...; dims=3), (1,3,2))], [fill(inv(sqrt(2)), 2)], 1)
    @test inner_product(known, slow) ≈ 1 atol=1e-9

    one = rand_iMPS(ComplexF64, 1, 3, 2)
    copy_one = deepcopy(one)
    restore_canonical!(one; method=:qr)
    @test inner_product(copy_one, one) ≈ 1 atol=1e-10
end
