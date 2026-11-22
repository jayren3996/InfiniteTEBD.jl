using Test
using LinearAlgebra
using Random
using InfiniteTEBD

# The constrained ScarFinder step (folded imaginary time, bounded repair), observation of
# trajectories, reproducible seeds, translation, and the prepared plan.
module SFW
using LinearAlgebra
using Random
using InfiniteTEBD
const N = ComplexF64[0 0; 0 1]
const X = ComplexF64[0 1; 1 0]
const P = ComplexF64[1 0; 0 0]
const I2 = Matrix{ComplexF64}(I, 2, 2)
const P3 = Diagonal(ComplexF64[(k & (k >> 1)) == 0 for k in 0:7])
fold(G) = P3 * G
h(c) = kron(P, X, P) + c * kron(I2, N, I2)
stag(μ, δ) = [[(h(μ + δ), 2, 4)], [(h(μ - δ), 1, 3)]]      # centres: site 1 (A), site 2 (B)
const LAYERS = stag(0.9, 0.0)
const U = trotter_mpos(LAYERS, 0.05; n=2, gate_map=fold)
const B = Blockade()
const PM = iMPO(B, 2)
gauss(id) = iMPS(ComplexF64, [randn(MersenneTwister(80100 + id), ComplexF64, 2, 2, 2) for _ in 1:2];
                 renormalize=true)
gauss_seq(id) = (rng = MersenneTwister(80100 + id);
                 iMPS(ComplexF64, [randn(rng, ComplexF64, 2, 2, 2) for _ in 1:2]; renormalize=true))
example() = apply!(product_iMPS(ComplexF64, [[cos(0.6), sin(0.6) * cis(0.3)], [cos(0.5), sin(0.5)]]), PM)
occ(ψ, j) = real(expect(ψ, N, j, j))
energy(ψ, layers=LAYERS) = energy_density(ψ, reduce(vcat, layers))
end
using .SFW: SFW

const _PLAN = ScarFinderPlan(SFW.LAYERS, 0.05; n=2, gate_map=SFW.fold, constraint=SFW.B)

@testset "static validation before mutation" begin
    q = kron(SFW.N, SFW.I2, SFW.N)
    ψ0 = SFW.gauss_seq(3)
    # two commuting three-site diagonal terms in one layer crossing the same bond (review reproducer)
    bad = [[(q, 1, 3), (q, 2, 4)]]
    ψ = deepcopy(ψ0)
    @test_throws ArgumentError scarfinder_step!(ψ, SFW.U, 2; nstep=2, maxdim=16, terms=bad, target=0.317,
                                                constraint=SFW.B)
    @test ψ.Γ == ψ0.Γ && ψ.λ == ψ0.λ
    @test_throws ArgumentError energy_fix!(deepcopy(ψ0), bad, 2; target=0.3)
    # a projector that does not commute with the generator
    Q = Matrix{ComplexF64}(Diagonal([1, 1, 1, 0, 1, 1, 1, 1]))
    Rx = kron(SFW.I2, [0.6 0.8; 0.8 -0.6], SFW.I2)
    badmap = G -> (Rx * Q * Rx') * G
    for kw in ((; gate_map=badmap, terms=SFW.LAYERS, target=0.3),
               (; terms=[[(kron(SFW.I2, SFW.I2), 1, 3)]], target=0.3),          # size vs support
               (; terms=SFW.LAYERS, target=0.3, repair=true),                    # repair needs constraint
               (; terms=SFW.LAYERS, target=0.3, constraint=SFW.B, repair=true, max_project=-1),
               (; terms=SFW.LAYERS, target=0.3, constraint=SFW.B, repair=true, leak_tol=NaN))
        ψ = deepcopy(ψ0)
        @test_throws ArgumentError scarfinder_step!(ψ, SFW.U, 2; nstep=2, maxdim=16, kw...)
        @test ψ.Γ == ψ0.Γ && ψ.λ == ψ0.λ
    end
    @test_throws ArgumentError ScarFinderPlan(bad, 0.05; n=2)
    @test_throws ArgumentError ScarFinderPlan(SFW.LAYERS, 0.05; n=2, gate_map=badmap)
    @test_throws ArgumentError ScarFinderPlan(SFW.LAYERS, 0.05; n=2, terms=reduce(vcat, SFW.stag(0.9, 0.1)))
    @test ScarFinderPlan(SFW.LAYERS, 0.05; n=2, terms=reverse(reduce(vcat, SFW.LAYERS))) isa ScarFinderPlan
end

@testset "folded imaginary time" begin
    leaky = truncate!(evolve!(SFW.gauss(4), SFW.U, 20; maxdim=16), 2)
    @test constraint_leakage(leaky, SFW.B) > 1e-6
    e = SFW.energy(leaky) + 0.05
    a, b = deepcopy(leaky), deepcopy(leaky)
    ra = energy_fix!(a, SFW.LAYERS, 2; target=e, tol=1e-8, n=2)
    rb = energy_fix!(b, SFW.LAYERS, 2; target=e, tol=1e-8, n=2, gate_map=SFW.fold)
    @test constraint_leakage(b, SFW.B) < 1e-2 * constraint_leakage(a, SFW.B)
    @test rb.initial_residual == ra.initial_residual
    @test rb.status in (:converged, :maxstep, :stalled) && rb.steps > 0
    # identity map: unchanged gates, so unchanged numbers
    c = deepcopy(leaky)
    rc = energy_fix!(c, SFW.LAYERS, 2; target=e, tol=1e-8, n=2, gate_map=identity)
    @test c.Γ == a.Γ && rc.steps == ra.steps
    # dτ → 0 with a folded projector is P, not the identity: a finite energy displacement
    zero_step = trotter_mpos(SFW.LAYERS, 0.0; n=2, evolution=:imaginary, gate_map=SFW.fold)
    t = deepcopy(leaky)
    for W in zero_step
        apply!(t, W; maxdim=2)
    end
    @test abs(SFW.energy(t) - SFW.energy(leaky)) > 1e-6
    legal = SFW.example()
    t = deepcopy(legal)
    for W in zero_step
        apply!(t, W; maxdim=8)
    end
    @test abs(SFW.energy(t) - SFW.energy(legal)) < 1e-12
end

@testset "energy correction far from the target: reachable and unreachable" begin
    for id in (1, 4)                                   # residual ≈ 0.83 and 0.70
        ψ = SFW.gauss(id)
        r = energy_fix!(ψ, SFW.LAYERS, 2; target=-0.4, tol=1e-6, n=2, gate_map=SFW.fold, maxstep=80)
        @test abs(r.initial_residual) > 0.6
        @test r.converged && r.status === :converged && abs(r.residual) <= 1e-6
        @test r.residual == SFW.energy(ψ) + 0.4
    end
    # an exhausted budget is not unreachability: from seed 3, e = 0.5 fails within 80
    # trials while seeds 1 and 4 reach it, and seed 3 itself reaches 0.7
    r3 = energy_fix!(SFW.gauss(3), SFW.LAYERS, 2; target=0.5, tol=1e-6, n=2, gate_map=SFW.fold, maxstep=80)
    @test r3.status === :maxstep && abs(r3.residual) > 1e-3
    @test energy_fix!(SFW.gauss(1), SFW.LAYERS, 2; target=0.5, n=2, gate_map=SFW.fold, maxstep=80).converged
    @test energy_fix!(SFW.gauss(3), SFW.LAYERS, 2; target=0.7, n=2, gate_map=SFW.fold, maxstep=80).converged
    # every term has spectrum in [-1, 1.9], so an energy density of 2.5 is unreachable
    @test all(t -> eigmax(Hermitian(t[1])) <= 1.9 + 1e-12, reduce(vcat, SFW.LAYERS))
    ψ = SFW.gauss(1)
    r = energy_fix!(ψ, SFW.LAYERS, 2; target=2.5, tol=1e-6, n=2, gate_map=SFW.fold, maxstep=40)
    @test !r.converged && r.status in (:maxstep, :stalled)
    @test r.residual <= -(2.5 - 1.9) && r.residual == SFW.energy(ψ) - 2.5 && r.steps <= 40
    @test energy_fix!(deepcopy(ψ), SFW.LAYERS, 2; target=0.3, maxstep=0, n=2).status === :not_run
end

@testset "bounded repair" begin
    # the reviewed example: five ordinary steps, then one evolution window
    s = SFW.example()
    e = SFW.energy(s)
    for _ in 1:5
        scarfinder_step!(s, SFW.U, 2; nstep=20, maxdim=16, terms=SFW.LAYERS, target=e, constraint=SFW.B)
    end
    r1 = scarfinder_step!(deepcopy(s), SFW.U, 2; nstep=20, maxdim=16, constraint=SFW.B,
                          repair=true, max_project=1, leak_tol=1e-10)
    @test r1.loss_svd ≈ 0.01285235 atol = 1e-7
    @test r1.loss ≈ 0.01195834 atol = 1e-7
    @test r1.project_steps == 1 && r1.project_status === :budget && r1.project_converged == false
    r3 = scarfinder_step!(deepcopy(s), SFW.U, 2; nstep=20, maxdim=16, constraint=SFW.B, repair=true)
    @test r3.project_steps == 3 && r3.loss > r1.loss && r3.leakage_repaired < 1e-11   # loss rises, leak falls
    @test r3.leakage_projected ≈ 1.36e-3 rtol = 0.01
    # constraint alone stays a diagnostic; repair=false keeps the SVD candidate
    a, b = deepcopy(s), deepcopy(s)
    ra = scarfinder_step!(a, SFW.U, 2; nstep=20, maxdim=16, constraint=SFW.B)
    rb = scarfinder_step!(b, SFW.U, 2; nstep=20, maxdim=16)
    @test a.Γ == b.Γ && ra.loss == rb.loss == ra.loss_svd == r1.loss_svd
    @test ra.project_status === :off && isnothing(ra.project_converged) && ra.leakage_repaired == ra.leakage_projected
    # loss on the repaired candidate, against the retained evolved state
    ψ = deepcopy(s)
    r = scarfinder_step!(ψ, SFW.U, 2; nstep=20, maxdim=16, constraint=SFW.B, repair=true, max_project=2)
    bref = evolve!(deepcopy(s), SFW.U, 20; maxdim=16)
    @test abs(r.loss + log(per_site_fidelity(bref, ψ))) < 1e-12
    @test abs(r.leakage_repaired - constraint_leakage(ψ, SFW.B)) < 1e-14
    # eight random examples: the budget is often exhausted, and that is reported
    outcomes = map(1:8) do id
        scarfinder_step!(SFW.gauss(id), SFW.U, 2; nstep=20, maxdim=16, constraint=SFW.B, repair=true)
    end
    @test all(o -> o.project_status in (:clean, :converged, :budget, :stalled), outcomes)
    @test all(o -> o.project_converged == (o.leakage_repaired <= 1e-10), outcomes)
    @test count(o -> !o.project_converged, outcomes) >= 1
    # already legal: no pass needed
    rl = scarfinder_step!(product_iMPS(ComplexF64, [[0, 1], [1, 0]]), SFW.U, 2; nstep=2, maxdim=2,
                          constraint=SFW.B, repair=true)
    @test rl.project_status === :clean && rl.project_steps == 0
    # the full configuration: repair, folded correction, physical-component energy
    ψ = deepcopy(s)
    r = scarfinder_step!(ψ, SFW.U, 2; nstep=20, maxdim=16, terms=SFW.LAYERS, target=e, tol=1e-8,
                         gate_map=SFW.fold, constraint=SFW.B, repair=true, projected_energy=true)
    @test r.energy_initial_error != r.energy_error
    @test r.energy_fix_status in (:converged, :maxstep, :stalled)
    @test r.converged == (r.energy_fix_status === :converged)
    @test r.energy_projected_status === :ok
    # each folded trial truncates back to χ, so the correction can reintroduce some leakage;
    # it is reported, and far below what the unfolded correction leaves
    ru = scarfinder_step!(deepcopy(s), SFW.U, 2; nstep=20, maxdim=16, terms=SFW.LAYERS, target=e,
                          tol=1e-8, constraint=SFW.B, repair=true)
    @test r.leakage_fixed isa Float64 && r.leakage_fixed < 1e-2 * ru.leakage_fixed
end

@testset "observing evolve! leaves the trajectory unchanged" begin
    ψ0 = truncate!(evolve!(SFW.gauss_seq(3), SFW.U, 3; maxdim=8), 2)
    for (merge, steps, every, cap) in ((true, 10, 3, 2), (true, 10, 1, 16), (false, 7, 2, 2))
        plain = evolve!(deepcopy(ψ0), SFW.U, steps; maxdim=cap, merge)
        samples = Any[]
        obs = (φ, k, st) -> push!(samples, (k, deepcopy(φ), st))
        watched = evolve!(deepcopy(ψ0), SFW.U, steps; maxdim=cap, merge, observe=obs, every)
        @test watched.Γ == plain.Γ && watched.λ == plain.λ
        ks = first.(samples)
        @test ks == unique(vcat(0, collect(every:every:steps), steps))
        @test isnothing(samples[1][3]) && samples[1][2].Γ == ψ0.Γ
        for (k, φ, st) in samples[2:end]
            prefix = evolve!(deepcopy(ψ0), SFW.U, k; maxdim=cap, merge)
            @test φ.Γ == prefix.Γ && φ.λ == prefix.λ
            @test st.max_kept_dim <= cap && st.num_at_cap >= 0
        end
    end
    # sampling by repeated one-step calls is a different integrator (no merged boundary)
    a = evolve!(deepcopy(ψ0), SFW.U, 10; maxdim=2)
    b = deepcopy(ψ0)
    for _ in 1:10
        evolve!(b, SFW.U, 1; maxdim=2)
    end
    @test abs(SFW.occ(a, 1) - SFW.occ(b, 1)) > 1e-8
    # release record
    obs, rec = release_recorder(ψ0; dt=0.05, site_ops=(; n=SFW.N))
    evolve!(deepcopy(ψ0), SFW.U, 20; maxdim=4, observe=obs, every=5)
    @test rec.step == [0, 5, 10, 15, 20] && rec.t ≈ [0, 0.25, 0.5, 0.75, 1.0]
    @test rec.return_fidelity[1] ≈ 1 && all(<=(1 + 1e-12), rec.return_fidelity)
    @test all(d -> all(<=(4), d), rec.dims) && length(rec.entropy[end]) == 2
    @test rec.ops[:n][1] ≈ [SFW.occ(ψ0, 1), SFW.occ(ψ0, 2)]
    @test any(>(0), rec.num_at_cap)
end

@testset "observing scarfinder!" begin
    ψ0 = SFW.example()
    e = SFW.energy(ψ0)
    kw = (; nstep=4, maxdim=12, terms=SFW.LAYERS, target=e, gate_map=SFW.fold, constraint=SFW.B,
          repair=true)
    seen = Any[]
    hist = Any[]
    a = scarfinder!(deepcopy(ψ0), SFW.U, 2, 4; refine=false, history=hist, kw...,
                    observe=(ψ, k, r) -> push!(seen, (k, deepcopy(ψ), r)))
    b = deepcopy(ψ0)
    loop = [(scarfinder_step!(b, SFW.U, 2; kw...), deepcopy(b)) for _ in 1:4]
    @test a.Γ == b.Γ && first.(seen) == 0:4 && isnothing(seen[1][3])
    @test all(seen[k+1][2].Γ == loop[k][2].Γ && seen[k+1][3].loss == loop[k][1].loss for k in 1:4)
    @test length(hist) == 4
    # refinement is not observed and not in the history
    seen2 = Int[]
    hist2 = Any[]
    scarfinder!(deepcopy(ψ0), SFW.U, 2, 2; refine=true, refine_step=3, history=hist2, kw...,
                observe=(ψ, k, r) -> push!(seen2, k))
    @test seen2 == [0, 1, 2] && length(hist2) == 2
    # the plan driver gives the same trajectory
    c = scarfinder!(deepcopy(ψ0), _PLAN, 2, 4; refine=false, nstep=4, maxdim=12, target=e, repair=true)
    @test c.Γ == a.Γ
end

@testset "reproducible seeds" begin
    Random.seed!(7)
    a = rand_iMPS(ComplexF64, 2, 2, 3)
    b = rand_iMPS(Xoshiro(7), ComplexF64, 2, 2, 3)
    @test a.Γ == b.Γ && a.λ == b.λ                               # same distribution and stream
    c = rand_iMPS(Xoshiro(hash((17, 3))), ComplexF64, 2, 2, 2)
    d = rand_iMPS(Xoshiro(hash((17, 3))), ComplexF64, 2, 2, 2)
    f = rand_iMPS(Xoshiro(hash((17, 4))), ComplexF64, 2, 2, 2)
    @test c.Γ == d.Γ && c.Γ != f.Γ
    @test rand_iMPS(Xoshiro(1), Float64, 3, 2, 2) isa iMPS
end

@testset "translate" begin
    Random.seed!(3)
    ψ = rand_iMPS(ComplexF64, 3, 2, 3)
    truncate!(ψ, 3)
    ψ.Γ[2] = ψ.Γ[2][:, :, 1:2]; ψ.Γ[3] = ψ.Γ[3][1:2, :, :]; ψ.λ[2] = ψ.λ[2][1:2]
    canonical!(ψ)
    @test length.(ψ.λ) != fill(length(ψ.λ[1]), 3)                # unequal bonds
    for k in (-4, -2, -1, 0, 1, 2, 3, 4)
        t = translate(ψ, k)
        @test all(t.Γ[i] == ψ.Γ[mod1(i - k, 3)] && t.λ[i] == ψ.λ[mod1(i - k, 3)] for i in 1:3)
        @test translate(t, -k).Γ == ψ.Γ
    end
    @test translate(ψ, 3).Γ == ψ.Γ && translate(ψ, -3).λ == ψ.λ
    t = translate(ψ, 1)
    t.Γ[2][1] = 99
    t.λ[2][1] = 99
    @test ψ.Γ[1][1] != 99 && ψ.λ[1][1] != 99                       # independent arrays
    @test abs(per_site_fidelity(translate(translate(ψ, 1), 2), ψ) - 1) < 1e-12
    # staggered chain: a one-site translation maps δ to −δ
    φ = SFW.gauss_seq(3)
    μ, δ = 0.9, 0.3
    @test SFW.energy(translate(φ, 1), SFW.stag(μ, δ)) ≈ SFW.energy(φ, SFW.stag(μ, -δ)) atol = 1e-12
    @test SFW.energy(translate(φ, 1), SFW.stag(μ, δ)) - SFW.energy(φ, SFW.stag(μ, δ)) ≈
          -δ * (SFW.occ(φ, 1) - SFW.occ(φ, 2)) atol = 1e-12
end

@testset "annihilation screen with the Frobenius bound" begin
    id1 = iMPO([reshape(SFW.I2, 1, 2, 2, 1)])
    for n in (2, 78, 120)
        ψ = product_iMPS(ComplexF64, [[1, 0] for _ in 1:n])
        @test abs(inner_product(apply!(deepcopy(ψ), id1)) - 1) < 1e-12
    end
    # rescaled and poorly gauged inputs
    ψ = SFW.gauss_seq(2)
    raw = InfiniteTEBD.iMPS([1e3 .* ψ.Γ[1], 1e-2 .* ψ.Γ[2]], [ones(2), ones(2)], 2)
    out = apply!(deepcopy(raw), SFW.U[2]; maxdim=8)
    ref = apply!(deepcopy(ψ), SFW.U[2]; maxdim=8)
    @test abs(per_site_fidelity(out, ref) - 1) < 1e-12
    # true annihilation and small finite weights
    full = product_iMPS(ComplexF64, [[0, 1], [0, 1]])
    @test_throws MPOAnnihilationError apply!(deepcopy(full), SFW.U[2])
    @test_throws MPOAnnihilationError apply!(deepcopy(full), SFW.U[2]; structured=false)
    Pz = iMPO([reshape(SFW.P, 1, 2, 2, 1)])
    for n in (1, 2, 8)
        rare = product_iMPS(ComplexF64, [[1e-6, sqrt(1 - 1e-12)] for _ in 1:n])
        @test abs(constraint_leakage(rare, Pz) - 13.815510557964274) < 1e-9
        @test inner_product(apply!(deepcopy(rare), Pz)) ≈ 1
    end
end

@testset "ScarFinderPlan agrees with the vector path" begin
    @test _PLAN.mpos[1].W == SFW.U[1].W && _PLAN.mpos[2].W == SFW.U[2].W
    snapshot = deepcopy([W.W for W in _PLAN.mpos])
    for id in (1, 2)
        a = SFW.gauss(id)
        apply!(a, SFW.PM; maxdim=2)
        b = deepcopy(a)
        for (ns, it) in ((20, 3), (2, 6))
            for _ in 1:it
                ra = scarfinder_step!(a, _PLAN, 2; nstep=ns, maxdim=16, target=0.25, repair=true,
                                      projected_energy=true)
                rb = scarfinder_step!(b, SFW.U, 2; nstep=ns, maxdim=16, terms=SFW.LAYERS, n=2,
                                      target=0.25, gate_map=SFW.fold, constraint=SFW.B, repair=true,
                                      projected_energy=true)
                @test a.Γ == b.Γ && a.λ == b.λ
                @test ra.energy_fix_steps == rb.energy_fix_steps && ra.loss == rb.loss &&
                      ra.project_steps == rb.project_steps && ra.energy_projected === rb.energy_projected
            end
        end
    end
    @test [W.W for W in _PLAN.mpos] == snapshot                     # the plan is not modified
end

@testset "shared plan, independent tasks" begin
    run(id) = begin
        rng = Xoshiro(hash((5, id)))
        ψ = iMPS(ComplexF64, [randn(rng, ComplexF64, 2, 2, 2) for _ in 1:2]; renormalize=true)
        apply!(ψ, SFW.PM; maxdim=2)
        rs = [scarfinder_step!(ψ, _PLAN, 2; nstep=4, maxdim=12, target=0.25, repair=true) for _ in 1:3]
        (ψ, rs)
    end
    serial = run.(1:4)
    tasks = [Threads.@spawn run(id) for id in 1:4]
    parallel = fetch.(tasks)
    for (s, p) in zip(serial, parallel)
        @test s[1].Γ == p[1].Γ && s[1].λ == p[1].λ
        @test [r.energy_fix_status for r in s[2]] == [r.energy_fix_status for r in p[2]]
    end
end
