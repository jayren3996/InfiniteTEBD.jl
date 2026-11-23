using InfiniteTEBD
using LinearAlgebra
using Random
using Test

@testset "Normalized per-site fidelity" begin
    a_sites = [[1.0, 2.0], [2.0, -1.0]]
    b_sites = [[2.0, 1.0], [1.0, -2.0], [0.75, 1.0]]
    tensors(sites) = [reshape(ComplexF64.(v), 1, length(v), 1) for v in sites]
    a = iMPS(ComplexF64, tensors(a_sites); renormalize=false)
    b = iMPS(ComplexF64, tensors(b_sites); renormalize=false)
    expected = sum(log(abs(dot(x, y)) / (norm(x) * norm(y)))
                   for (x, y) in zip(repeat(a_sites, 3), repeat(b_sites, 2))) / 6
    @test log_per_site_fidelity(a, b; period=6) ≈ expected
    @test log_per_site_fidelity(a.Γ, b.Γ; period=12) ≈ expected
    @test per_site_fidelity(a, b; normalize=true, period=6) ≈ exp(expected)
    @test_throws ArgumentError per_site_fidelity(a, b)
    @test_throws ArgumentError log_per_site_fidelity(a, b)
    @test_throws ArgumentError log_per_site_fidelity(a, b; period=3)
    @test_throws ArgumentError log_per_site_fidelity(a, b; period=0)

    cached = inner_product(a; strict=true)
    @test log_per_site_fidelity(a, b; period=6, self_overlaps=(cached, nothing)) ≈ expected
    @test log_per_site_fidelity(b, a; period=6, self_overlaps=(nothing, cached)) ≈ expected
    @test log_per_site_fidelity(a, a) == 0
    @test log_per_site_fidelity(a, a; self_overlaps=(cached, cached)) == 0
    @test_throws DomainError log_per_site_fidelity(a, a; self_overlaps=(0.5cached, cached))
    @test_throws DomainError log_per_site_fidelity(a, a; self_overlaps=(NaN, nothing))

    up = product_iMPS([[1.0, 0.0]])
    down = product_iMPS([[0.0, 1.0]])
    @test log_per_site_fidelity(up, down) == -Inf
    @test per_site_fidelity(up, down; normalize=true) == 0
    zero_state = iMPS(Float64, [zeros(1, 2, 1)]; renormalize=false)
    @test_throws DomainError log_per_site_fidelity(up, zero_state)
    @test_throws DimensionMismatch log_per_site_fidelity(up, product_iMPS([[1.0, 0.0, 0.0]]))

    # The unnormalized transfer radii individually fit in Float64, while their
    # products (or squares) overflow or underflow.
    for scale in (1e-120, 1e120)
        s = iMPS(Float64, [scale .* up.Γ[1]]; renormalize=false)
        t = iMPS(Float64, [scale .* reshape([0.6, 0.8], 1, 2, 1)]; renormalize=false)
        @test log_per_site_fidelity(s, t) ≈ log(0.6) atol=2e-13
    end
end

@testset "Matrix-free normalized overlap and strict failure" begin
    rng = MersenneTwister(421)
    χ = 12
    right_isometries = map(1:2) do _
        Q = Matrix(qr(randn(rng, ComplexF64, 2χ, χ)).Q)
        reshape(Matrix(Q[:, 1:χ]'), χ, 2, χ)
    end
    a = iMPS(ComplexF64, right_isometries; renormalize=false)
    gauge = Diagonal(exp.(range(-1, 1; length=χ)))
    tensors = [similar(A) for A in a.Γ]
    for (j, A) in enumerate(a.Γ), s in axes(A, 2)
        tensors[j][:, s, :] = (2.4im + j) .* (gauge \ A[:, s, :] * gauge)
    end
    b = iMPS(ComplexF64, tensors; renormalize=false)
    @test log_per_site_fidelity(a, b) ≈ 0 atol=2e-12
    @test log_per_site_fidelity(a, b; self_overlaps=(1.0, nothing)) ≈ 0 atol=2e-12
    scaled = iMPS(ComplexF64, [1e-35 .* B for B in tensors]; renormalize=false)
    @test log_per_site_fidelity(a, scaled) ≈ 0 atol=2e-12
    a32 = iMPS(ComplexF32, right_isometries; renormalize=false)
    b32 = iMPS(ComplexF32, tensors; renormalize=false)
    @test log_per_site_fidelity(a32, b32) ≈ 0 atol=2e-6
    @test log_per_site_fidelity(a, b32) ≈ 0 atol=2e-6

    # Force a Krylov failure on a transfer too large for the primary dense path
    # but small enough that the legacy bounded fallback is available.
    raw = [randn(rng, ComplexF64, χ, 2, χ)]
    @test_throws ErrorException inner_product(raw; strict=true, tol=1e-24, maxiter=1)
    @test_throws ErrorException log_per_site_fidelity(raw, raw; tol=1e-24, maxiter=1)
    dense = maximum(abs, eigvals(InfiniteTEBD.gtrm(raw, raw)))
    @test inner_product(raw; strict=true) ≈ dense rtol=1e-12
end

@testset "Uneven mixed transfer seam" begin
    rng = MersenneTwister(593)
    a = [randn(rng, ComplexF64, 10, 2, 2), randn(rng, ComplexF64, 2, 2, 10)]
    b = [randn(rng, ComplexF64, 11, 2, 3), randn(rng, ComplexF64, 3, 2, 7),
         randn(rng, ComplexF64, 7, 2, 11)]
    # Their native seams have product dimension 110; after cyclic rotation
    # the common period of six can be contracted from product dimension 6.
    ac, bc = repeat(a, 3), repeat(b, 2)
    mixed = InfiniteTEBD.gtrm(ac, bc)
    λ = argmax(abs, eigvals(mixed))
    @test InfiniteTEBD._dominant_chain_eigenvalue(ac, bc; strict=true) ≈ λ rtol=1e-12
    norm_a = maximum(abs, eigvals(InfiniteTEBD.gtrm(a, a)))
    norm_b = maximum(abs, eigvals(InfiniteTEBD.gtrm(b, b)))
    expected = log(abs(λ)) / 6 - log(norm_a) / 4 - log(norm_b) / 6
    @test log_per_site_fidelity(a, b; period=6) ≈ expected atol=1e-12
end
