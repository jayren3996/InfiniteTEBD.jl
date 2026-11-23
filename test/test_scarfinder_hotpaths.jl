using Test, InfiniteTEBD, LinearAlgebra

@testset "Annihilation screen uses relative norms" begin
    P = [1.0 -1.0; -1.0 1.0] ./ 2
    W = iMPO([reshape(P, 1, 2, 2, 1)])
    for scale in (1.0, 2.0^80)
        ψ = iMPS(Float64, [reshape(scale .* [1.0, 1 + 2.0^-47], 1, 2, 1)]; renormalize=false)
        tensor = copy(ψ.Γ[1])
        @test constraint_leakage(ψ, W) == Inf
        @test_throws MPOAnnihilationError apply!(ψ, W)
        @test ψ.Γ[1] == tensor
    end
end

@testset "Both Schmidt cuts are applied in one site slice" begin
    dims = (2, 3, 2, 2)
    tensors = [reshape(collect(1.0:2 * dims[mod1(i - 1, 4)] * dims[i]),
                       dims[mod1(i - 1, 4)], 2, dims[i]) for i in 1:4]
    weight = [sqrt(0.5), sqrt(0.5)]
    weights = [copy(weight), normalize([0.8, 0.5, 0.3]), copy(weight), copy(weight)]
    ψ = iMPS(copy(tensors), copy(weights), 4)
    InfiniteTEBD._truncate_canonical!(ψ; maxdim=4, cutoff=0.4, truncerr=0.0,
                                     renormalize=false, recanonicalize=false)
    @test ψ.Γ[1] === tensors[1] && ψ.Γ[4] === tensors[4]
    @test ψ.Γ[2] == tensors[2][:, :, 1:2]
    @test ψ.Γ[3] == tensors[3][1:2, :, :]
    @test ψ.λ[1] === weights[1] && ψ.λ[3] === weights[3] && ψ.λ[4] === weights[4]
    @test ψ.λ[2] == weights[2][1:2]
    unnormalized = [[0.7, 0.7], weights[2], weights[3], weights[4]]
    ψ = iMPS(copy(tensors), copy(unnormalized), 4)
    InfiniteTEBD._truncate_canonical!(ψ; maxdim=4, cutoff=0.4, truncerr=0.0,
                                     renormalize=true, recanonicalize=false)
    @test unnormalized[1] == [0.7, 0.7]
    @test ψ.λ[1] == unnormalized[1] ./ norm(unnormalized[1])
    one_site = iMPS([tensors[1]], [weight], 1)
    InfiniteTEBD._truncate_canonical!(one_site; maxdim=1, cutoff=0.0, truncerr=0.0,
                                     renormalize=false, recanonicalize=false)
    @test only(one_site.Γ) == tensors[1][1:1, :, 1:1]
    @test only(one_site.λ) == weight[1:1]
end

@testset "Unmerged MPO observations preserve the cut and endpoint gauge" begin
    X = [0.0 1.0; 1.0 0.0]
    P = Diagonal([1.0, 0.0])
    H = kron(P, X, P)
    U = trotter_mpos([[(H, 2, 4)], [(H, 1, 3)]], 0.11; n=2)
    initial = product_iMPS(ComplexF64, [[0.8, 0.6], [0.6, 0.8]])
    for (structured, renormalize) in ((true, true), (false, true), (false, false))
        opts = (; maxdim=1, merge=false, structured, renormalize, return_stats=:summary)
        samples = []
        state, stats = evolve!(deepcopy(initial), U, 3; opts...,
                              observe=(ψ, k, r) -> 0 < k < 3 && push!(samples, ψ))
        reference, reference_stats = evolve!(deepcopy(initial), U, 3; opts...)
        @test stats == reference_stats
        @test stats.max_discarded_weight > 0
        @test state.Γ == reference.Γ && state.λ == reference.λ
        @test length(samples) == 2
        for (k, sample) in enumerate(samples)
            prefix, _ = evolve!(deepcopy(initial), U, k; opts...)
            @test sample.Γ == prefix.Γ && sample.λ == prefix.λ
            sample.Γ[1][1] += 1
            @test state.Γ == reference.Γ
        end
    end
end

@testset "Unreported ScarFinder and mutable observed windows" begin
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    layers = [[(X, 1, 1)], [(Z, 1, 1)]]
    U = trotter_mpos(layers, 0.19; n=1)
    initial = product_iMPS(ComplexF64, [[1, 0]])

    regular = scarfinder!(deepcopy(initial), U, 1, 3; nstep=2, refine=false)
    manual = deepcopy(initial)
    for _ in 1:3
        @test scarfinder_step!(manual, U, 1; nstep=2).loss ≈ 0 atol=1e-14
    end
    @test regular.Γ == manual.Γ && regular.λ == manual.λ
    @test project!(deepcopy(initial), 1; terms=[(X, 1, 1)]).energy_error === nothing

    replacement = layer_mpo([(exp(-0.47im * X), 1, 1)], 1)
    changed = copy(U)
    manual = deepcopy(initial)
    scarfinder_step!(manual, changed, 1; nstep=2)
    changed[1] = replacement
    scarfinder_step!(manual, changed, 1; nstep=2)
    changed = copy(U)
    observe(ψ, k, report) = k == 1 && (changed[1] = replacement)
    watched = scarfinder!(deepcopy(initial), changed, 1, 2; nstep=2, refine=false, observe)
    @test watched.Γ == manual.Γ && watched.λ == manual.λ
end
