# Regression tests for symmetric iMPS construction (2026-06-09 review).
#
# Bug 1 (P0): `rand_iMPS(:U1, [-1, 1]; χ, n)` derived ALL auto bond sectors
# from fuse(P ⊗ P) — pure even charges for the spin-1/2 convention P = (±1) —
# so every site hom-space `V ⊗ P ← V` had dimension zero and the constructor
# silently returned an identically-zero state (canonical!, evolve!, expect,
# ent_S all "worked" and returned 0.0). The bond spaces are now built by
# cumulative charge fusion around the unit cell (alternating parity for ±1
# charges), and zero-dimensional site hom-spaces throw an informative
# ArgumentError everywhere (auto-built and explicit spaces alike).
#
# Bug 2 (P1): the symmetric two-site `applygate!` accepted ψ.n == 1, where
# sites i and j alias the same tensor; the second SVD-half assignment
# clobbered the first and silently corrupted the state. It now throws an
# ArgumentError up front, leaving the state untouched.

using Test
using InfiniteTEBD
using LinearAlgebra: I, norm
using Random

# Explicit imports from TensorKit to avoid name conflicts with ITensors
# (both are test dependencies and share some exported names like `dim`).
using TensorKit: U1Irrep, Z2Irrep, Vect, dim, blocks, id, sectors, domain,
                 codomain, AbstractTensorMap, DiagonalTensorMap, ←, ⊗, @tensor

@testset "docstring example rand_iMPS(:U1, [-1,1]; χ=8, n=2) is a nonzero state" begin
    # Verbatim docstring example. Before the fix this returned a state whose
    # every tensor was identically zero, with no error.
    Random.seed!(7)
    ψ = rand_iMPS(:U1, [-1, 1]; χ=8, n=2)
    @test ψ.n == 2
    for i in 1:2
        @test norm(ψ.Γ[i]) > 0
        @test 1 ≤ dim(domain(ψ.Γ[i])[1]) ≤ 8
    end
    # The auto-built bonds alternate parity for ±1 physical charges: one bond
    # carries only odd U(1) charges, the next only even ones.
    parities = [unique(mod(Int(s.charge), 2) for s in sectors(domain(ψ.Γ[i])[1]))
                for i in 1:2]
    @test sort([only(p) for p in parities]) == [0, 1]

    # canonical! works on the fixed state and yields a norm-1 canonical form:
    # Schmidt weights sum to 1 on every bond and each Γ is a right isometry.
    canonical!(ψ)
    Sz, _, _, _ = spin_half_ops(:U1)
    for i in 1:2
        @test norm(ψ.Γ[i]) > 0
        @test isapprox(sum(abs2, schmidt_values(ψ, i)), 1.0; atol=1e-9)
        Γ = ψ.Γ[i]
        @tensor R[a; b] := Γ[a, s, c] * conj(Γ[b, s, c])
        for (_, blk) in blocks(R)
            @test isapprox(blk, Matrix{ComplexF64}(I, size(blk)); atol=1e-7)
        end
        # One-site observable is finite (the dead state pinned this at 0.0
        # with no way to tell anything was wrong).
        v = expect(ψ, Sz, i, i)
        @test isfinite(real(v)) && isfinite(imag(v))
    end
end

@testset "gates actually change the fixed docstring state" begin
    # The zero state was a fixed point of every gate; the repaired state must
    # respond to evolution.
    Random.seed!(7)
    ψ = rand_iMPS(:U1, [-1, 1]; χ=8, n=2)
    canonical!(ψ)
    Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)
    h = SzSz + 0.5 * (SpSm + SmSp)
    G = exp(-0.2 * h)   # imaginary-time Heisenberg gate, decidedly non-identity

    before = schmidt_values(ψ, 1)
    applygate!(ψ, G, 1, 2; maxdim=16)
    after = schmidt_values(ψ, 1)
    @test all(norm(ψ.Γ[i]) > 0 for i in 1:2)
    @test length(before) != length(after) || !isapprox(before, after; atol=1e-10)
    # Observables are no longer pinned at exactly 0.0.
    @test abs(energy_density(ψ, h)) > 1e-3

    # evolve! smoke on a fresh copy of the same construction.
    Random.seed!(7)
    ψ2 = rand_iMPS(:U1, [-1, 1]; χ=8, n=2)
    canonical!(ψ2)
    evolve!(ψ2, [(G, 1, 2), (G, 2, 1)], 3; maxdim=16)
    @test all(norm(ψ2.Γ[i]) > 0 for i in 1:2)
    for i in 1:ψ2.n
        @test isapprox(sum(abs2, schmidt_values(ψ2, i)), 1.0; atol=1e-8)
    end
end

@testset "charge-frustrated constructions throw informative ArgumentError" begin
    # Spin-1/2 U(1) charges ±1 on an odd unit cell: the cell's net charge is
    # half-odd-integer, so no flux-0 wraparound exists — n=1 is impossible.
    err = try
        rand_iMPS(:U1, [-1, 1]; χ=8, n=1)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("frustrat", err.msg)
    @test occursin("n=1", err.msg)

    # Any odd n is equally impossible for ±1 charges.
    @test_throws ArgumentError rand_iMPS(:U1, [-1, 1]; χ=8, n=3)

    # A Z2 physical leg with only the odd charge is frustrated on odd cells too.
    @test_throws ArgumentError rand_iMPS(:Z2, [1]; χ=2, n=1)
    # ... but fine on even cells.
    ψz = rand_iMPS(:Z2, [1]; χ=2, n=2)
    @test all(norm(ψz.Γ[i]) > 0 for i in 1:2)

    # Explicit user-provided bond spaces are honoured but validated: a
    # pure-even bond space cannot connect to itself through ±1 physical
    # charges (zero-dimensional site hom-space → identically-zero state).
    P = graded_space(:U1, 1=>1, -1=>1)
    V_dead = graded_space(:U1, 0=>2, 2=>1, -2=>1)
    err2 = try
        rand_iMPS(P, V_dead, 2)
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("frustrat", err2.msg)

    # :Trivial has no grading and can never be frustrated.
    ψt = rand_iMPS(:Trivial, [0, 0]; χ=4, n=1)
    @test norm(ψt.Γ[1]) > 0
end

@testset "two-site applygate! on an n=1 unit cell throws and leaves state intact" begin
    # Valid n=1 symmetric state: mixed-parity bond space (same configuration
    # as the n=1 canonical! testset in test_symmetric_basic.jl).
    Random.seed!(2)
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 1=>2, -1=>2)
    ψ = rand_iMPS(P, V, 1)
    canonical!(ψ)
    Iop = id(ComplexF64, P ⊗ P)

    Γ_before = copy(ψ.Γ[1])
    λ_before = copy(ψ.λ[1])
    err = try
        applygate!(ψ, Iop, 1, 2)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("at least 2 sites", err.msg)
    # The guard fires before any mutation: the state is bit-for-bit untouched.
    @test norm(ψ.Γ[1] - Γ_before) == 0
    @test norm(ψ.λ[1] - λ_before) == 0
    # evolve! routes through the same applygate!, so it throws too.
    @test_throws ArgumentError evolve!(ψ, [(Iop, 1, 2)], 1)
    @test norm(ψ.Γ[1] - Γ_before) == 0

    # Same guard on a deterministic n=1 product state (Z2, zero occupation).
    ψz = product_iMPS(:Z2, [0, 1], [0])
    Pz = codomain(ψz.Γ[1])[2]
    Iz = id(ComplexF64, Pz ⊗ Pz)
    Γz_before = copy(ψz.Γ[1])
    @test_throws ArgumentError applygate!(ψz, Iz, 1, 2)
    @test norm(ψz.Γ[1] - Γz_before) == 0

    # Sanity: the guard does not affect n ≥ 2 — the same gate applies cleanly,
    # including via periodically-normalized site labels.
    ψ2 = product_iMPS(:U1, [-1, 1], [1, -1])
    P2 = codomain(ψ2.Γ[1])[2]
    I2 = id(ComplexF64, P2 ⊗ P2)
    applygate!(ψ2, I2, 1, 2; maxdim=8)
    applygate!(ψ2, I2, 3, 2; maxdim=8)
    @test all(norm(ψ2.Γ[i]) > 0 for i in 1:2)
end
