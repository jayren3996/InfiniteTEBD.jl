using Test
using LinearAlgebra
using Random
using InfiniteTEBD

function check_evolution_summary(summary, detail, field; cap)
    updates = getproperty(detail, field)
    bonds = [bond for update in updates for bond in update.bond_stats]
    @test !hasproperty(summary, field)
    @test all(name -> getproperty(summary, name) == getproperty(detail, name), keys(summary))
    @test summary.num_bond_updates == length(bonds)
    @test summary.sum_discarded_weight ≈ sum(b -> b.discarded_weight, bonds; init=0.0)
    @test summary.max_discarded_weight == maximum(b -> b.discarded_weight, bonds; init=0.0)
    @test summary.num_saturated == count(b -> b.saturated, bonds)
    @test summary.num_at_cap == count(b -> b.chi_keep == cap, bonds)
    @test summary.num_target_unmet == count(b -> !b.target_met, bonds)
end

@testset "Gate observers and streaming statistics" begin
    rng = MersenneTwister(71)
    initial = rand_iMPS(rng, ComplexF64, 4, 2, 2)
    X = [0.0 1.0; 1.0 0.0]
    Z = Diagonal([1.0, -1.0])
    G = exp(-0.27im * kron(X, Z))
    gates = [(G, 1, 2), (G, 4, 1)]
    cap = 2
    for gauge in (nothing, :auto)
        observed = Tuple[]
        observe(ψ, k, s) = push!(observed, (k, deepcopy(ψ), s))
        final, summary = evolve!(deepcopy(initial), gates, 5;
                                 maxdim=cap, gauge, observe, every=2, return_stats=:summary)
        reference, detail = evolve!(deepcopy(initial), gates, 5;
                                     maxdim=cap, gauge, return_stats=true)
        @test final.Γ == reference.Γ && final.λ == reference.λ
        check_evolution_summary(summary, detail, :gate_updates; cap)
        @test first.(observed) == [0, 2, 4, 5]
        @test observed[1][2].Γ == initial.Γ && isnothing(observed[1][3])
        @test [s.num_bond_updates for (_, _, s) in observed[2:end]] == [4, 4, 2]
        @test sum(s.sum_discarded_weight for (_, _, s) in observed[2:end]) ≈ summary.sum_discarded_weight
        for (k, sample, _) in observed[2:end]
            prefix = evolve!(deepcopy(initial), gates, k; maxdim=cap, gauge)
            @test sample.Γ == prefix.Γ && sample.λ == prefix.λ
        end
    end

    seen = Int[]
    @test evolve!(initial, gates, 0; observe=(ψ, k, s) -> push!(seen, k)) === initial
    @test seen == [0]
    @test_throws ArgumentError evolve!(initial, gates, 1; every=0)
    @test_throws ArgumentError evolve!(initial, gates, 1; return_stats=:unknown)
end

@testset "Layer samples preserve merged and adaptive gate schedules" begin
    rng = MersenneTwister(78)
    initial = rand_iMPS(rng, ComplexF64, 4, 2, 2)
    X = [0.0 1.0; 1.0 0.0]
    Z = Diagonal([1.0, -1.0])
    layers = [[(kron(X, X), 4, 1)], [(kron(Z, X), 1, 2)]]
    for (trotter, chi_policy) in ((:second, :fixed), (:fourth, :adaptive), (:fourth_opt, :fixed))
        opts = (; maxdim=3, trotter, chi_policy)
        samples = Tuple[]
        observe(ψ, k, s) = push!(samples, (k, deepcopy(ψ), s))
        ψ, stats = evolve!(deepcopy(initial), layers, 0.16, 3; opts..., observe, return_stats=true)
        reference, detail = evolve!(deepcopy(initial), layers, 0.16, 3; opts..., return_stats=true)
        @test ψ.Γ == reference.Γ && ψ.λ == reference.λ
        @test stats == detail
        @test length(stats.gate_updates) == length(trotter_gates(layers, 0.16; trotter, steps=3))
        @test first.(samples) == 0:3
        @test sum(length(s.gate_updates) for (_, _, s) in samples[2:end]) == length(stats.gate_updates) + 2
        for (k, sample, _) in samples[2:end]
            prefix = evolve!(deepcopy(initial), layers, 0.16, k; opts...)
            @test sample.Γ == prefix.Γ && sample.λ == prefix.λ
        end
    end
end

@testset "MPO interval statistics and observation compatibility" begin
    X = [0.0 1.0; 1.0 0.0]
    P = Diagonal([1.0, 0.0])
    H = kron(P, X, P)
    mpos = trotter_mpos([[(H, 2, 4)], [(H, 1, 3)]], 0.08; n=2)
    initial = product_iMPS(ComplexF64, [[0.8, 0.6], [0.6, 0.8]])
    cap = 2
    for merge in (false, true)
        samples = Tuple[]
        observe(ψ, k, s) = push!(samples, (k, deepcopy(ψ), s))
        ψ, totals = evolve!(deepcopy(initial), mpos, 5; maxdim=cap, merge,
                            observe, every=2, return_stats=:summary)
        reference, detail = evolve!(deepcopy(initial), mpos, 5; maxdim=cap, merge, return_stats=true)
        @test ψ.Γ == reference.Γ && ψ.λ == reference.λ
        check_evolution_summary(totals, detail, :updates; cap)
        @test first.(samples) == [0, 2, 4, 5]
        @test all(!hasproperty(s, :updates) for (_, _, s) in samples[2:end])
        # Two observed intermediate completions use one extra MPO each only
        # when the main trajectory proceeds with a merged boundary.
        @test sum(s.num_bond_updates for (_, _, s) in samples[2:end]) ==
            totals.num_bond_updates + (merge ? 2initial.n : 0)
        for (k, sample, _) in samples[2:end]
            prefix = evolve!(deepcopy(initial), mpos, k; maxdim=cap, merge)
            @test sample.Γ == prefix.Γ && sample.λ == prefix.λ
        end
    end

    counts = Int[]
    function observe_legacy(ψ, k, stats)
        k > 0 && push!(counts, length(stats.updates))
    end
    evolve!(deepcopy(initial), mpos, 5; merge=false, observe=observe_legacy, every=2)
    @test counts == [2length(mpos), 2length(mpos), length(mpos)]
    empty_mpos = mpos[1:0]
    seen = Int[]
    _, summary = evolve!(deepcopy(initial), empty_mpos, 3;
                         observe=(ψ, k, s) -> push!(seen, k), every=2, return_stats=:summary)
    @test seen == [0, 2, 3]
    @test summary.num_bond_updates == 0 && summary.sum_discarded_weight == 0
end
