using Test
using LinearAlgebra
using Random
using InfiniteTEBD

@testset "LOCAL_REDUCED_DENSITY" begin
    # A width-three window starts in the middle of a heterogeneous cell.
    a = normalize(ComplexF64[1, im])
    b = normalize(ComplexF64[2, 1-im, im/2])
    product = iMPS([reshape(a, 1, 2, 1), reshape(b, 1, 3, 1)], [[1.0], [1.0]], 2)
    w = kron(b, a, b)
    @test reduced_density(product, 0, 3) ≈ w * w'
    @test reduced_density(measurement_environments(product), 2, 3) ≈ w * w'
    @test_throws ArgumentError reduced_density(product, 1, 0)

    # Applying environments at the edges must retain scalar promotion even
    # when the Schmidt spectrum and the state tensors use different precision.
    mixed_precision = iMPS([ComplexF32.(A) for A in product.Γ], product.λ, 2)
    for width in (1, 3)
        density = reduced_density(mixed_precision, 2, width)
        @test eltype(density) === ComplexF64
        @test density ≈ reduced_density(product, 2, width) atol=1e-7
    end

    # Legacy SolitonFinder basis uses the first site as the low bit and maps
    # bit zero to tensor index 2. Check the production permutation against
    # the package's leftmost-factor-first basis on a width-six window.
    rng = MersenneTwister(41)
    ψ = rand_iMPS(rng, ComplexF64, 4, 2, 3)
    start, width = 4, 6
    previous = mod1(start - 1, ψ.n)
    last = mod1(start + width - 1, ψ.n)
    χl, χr = length(ψ.λ[previous]), length(ψ.λ[last])
    amplitudes = zeros(ComplexF64, χl * χr, 1 << width)
    permutation = zeros(Int, 1 << width)
    for bits in 0:(1 << width)-1
        A = Matrix(Diagonal(ComplexF64.(ψ.λ[previous])))
        physical_idx = 0
        for k in 0:width-1
            s = isodd(bits >> k) ? 1 : 2
            physical_idx = 2physical_idx + (s - 1)
            A = A * ψ.Γ[mod1(start+k, ψ.n)][:, s, :]
        end
        amplitudes[:, bits+1] = vec(A)
        permutation[bits+1] = physical_idx + 1
    end
    legacy = transpose(amplitudes' * amplitudes)
    legacy /= real(tr(legacy))
    ρ = reduced_density(ψ, start, width)
    @test ρ[permutation, permutation] ≈ legacy atol=1e-12
    @test tr(ρ) ≈ 1
    @test minimum(eigvals(ρ)) >= -1e-14

    O = kron(ComplexF64[0 -im; im 0], [1 0; 0 -1])
    @test tr(reduced_density(ψ, 4, 2) * O) ≈ expect(ψ, O, 4, 5) atol=1e-12
end

@testset "REDUCED_DENSITY_TRANSFER_SNAPSHOT" begin
    rng = MersenneTwister(73)
    ψ = iMPS(ComplexF64, [randn(rng, ComplexF64, 2, 2, 3),
                         randn(rng, ComplexF64, 3, 3, 2)])
    # Transform every bond by a nonscalar, complex gauge. Stored Schmidt
    # weights are intentionally uninformative in the new representation.
    gauges = [I + 0.25randn(rng, ComplexF64, χ, χ) for χ in length.(ψ.λ)]
    raw = [copy(A) for A in ψ.Γ]
    for j in 1:ψ.n, s in axes(raw[j], 2)
        raw[j][:, s, :] = gauges[mod1(j - 1, ψ.n)] \ (ψ.Γ[j][:, s, :] * gauges[j])
    end
    raw[1] .*= 1e70 * cis(0.2)
    raw[2] .*= 1e-70 * cis(-0.6)
    unnormalized = iMPS(raw, [ones(length(s)) for s in ψ.λ], ψ.n)
    input = deepcopy(unnormalized)
    env = measurement_environments(unnormalized)
    @test unnormalized.Γ == input.Γ
    @test unnormalized.λ == input.λ
    for start in -1:2, width in 1:3
        @test reduced_density(env, start, width) ≈ reduced_density(ψ, start, width) atol=1e-11
    end
    unnormalized.Γ[1] .= 0
    @test reduced_density(env, 1, 1) ≈ reduced_density(ψ, 1, 1) atol=1e-11
    @test_throws ArgumentError measurement_environments(unnormalized)
    @test_throws ArgumentError measurement_environments(ψ; maxiter=0)
    @test_throws ArgumentError measurement_environments(ψ; tol=0)

    # This transfer is scaled away from unit radius and has two sectors.
    cat = zeros(Float64, 2, 2, 2)
    cat[1,1,1] = cat[2,2,2] = 3
    @test_throws ArgumentError measurement_environments(iMPS([cat], [ones(2)], 1))
end
