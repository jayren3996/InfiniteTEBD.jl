using Test
using LinearAlgebra
using Random
using InfiniteTEBD

# Blockade leakage checked against the projector MPO and an independent transfer matrix.
module CDPXP
using LinearAlgebra
using InfiniteTEBD
const X = ComplexF64[0 1; 1 0]
const P = ComplexF64[1 0; 0 0]
const N = ComplexF64[0 0; 0 1]
const I2 = Matrix{ComplexF64}(I, 2, 2)
const P3 = Diagonal(ComplexF64[!(((s >> 2) & 1 == 1 && (s >> 1) & 1 == 1) ||
                                 ((s >> 1) & 1 == 1 && s & 1 == 1)) for s in 0:7])
const CA, CB = 1.2, 0.6
h(c) = kron(P, X, P) + c * kron(I2, N, I2)
layers() = [[(h(CA), 2, 4)], [(h(CB), 1, 3)]]
terms() = reduce(vcat, layers())
product(u, k) = product_iMPS(ComplexF64, [ComplexF64.(u) for _ in 1:k])

# Perron root of the product of the two classical hard-core site transfers.
function exact_leakage(u1, u2=u1)
    us = (u1 ./ norm(u1), u2 ./ norm(u2))
    T = [[w[1] w[2]; w[1] 0.0] for w in (us[1] .^ 2, us[2] .^ 2)]
    F = eigen(T[1] * T[2])
    k = argmax(abs.(F.values))
    return -log(real(F.values[k])) / 4
end
end
using .CDPXP: CDPXP

const _BLK = Blockade()
const _PMPO = iMPO(_BLK, 2)
const _UF = trotter_mpos(CDPXP.layers(), 0.05; n=2, gate_map=G -> CDPXP.P3 * G)
const _UU = trotter_mpos(CDPXP.layers(), 0.05; n=2)
_tile(ψ, k) = InfiniteTEBD.iMPS(repeat(ψ.Γ, k), repeat(ψ.λ, k), k * ψ.n)

@testset "Blockade agrees with the projector MPO" begin
    Random.seed!(11)
    ref = layer_mpo([(I - kron(CDPXP.N, CDPXP.N), k, k + 1) for k in 1:2], 2)
    χ8 = rand_iMPS(ComplexF64, 2, 2, 8)
    @test abs(constraint_leakage(χ8, _PMPO) - constraint_leakage(χ8, ref)) < 1e-13
    legal = apply!(CDPXP.product([cos(0.6), sin(0.6)], 2), _PMPO)
    evolved = evolve!(deepcopy(legal), _UU, 5; maxdim=16)          # PXP keeps it legal
    truncated = truncate!(deepcopy(evolved), 2)                     # the projection leaks
    inputs = [legal, evolved, truncated, χ8]
    for ψ in inputs
        @test abs(constraint_leakage(ψ, _BLK) - constraint_leakage(ψ, _PMPO)) < 1e-13
    end
    @test constraint_leakage(evolved, _BLK) < 1e-13
    @test constraint_leakage(truncated, _BLK) > 1e-12
end

@testset "already legal states" begin
    for ψ in (apply!(CDPXP.product([cos(0.6), sin(0.6)], 2), _PMPO),
              product_iMPS(ComplexF64, [[0, 1], [1, 0]]))
        for P in (_BLK, _PMPO)
            @test constraint_leakage(ψ, P) < 1e-14
        end
    end
end

@testset "leaky product state: exact leakage" begin
    for (u1, u2) in (([cos(0.35), sin(0.35)], [cos(0.35), sin(0.35)]),
                     ([cos(0.9), sin(0.9)], [cos(0.9), sin(0.9)]), ([0.3, 0.8], [0.9, -0.5]))
        ex = CDPXP.exact_leakage(u1, u2)
        ψ = product_iMPS(ComplexF64, [ComplexF64.(u1 ./ norm(u1)), ComplexF64.(u2 ./ norm(u2))])
        for P in (_BLK, _PMPO)
            @test abs(constraint_leakage(ψ, P) - ex) < 1e-13
        end
    end
end

@testset "small but finite projection weight" begin
    # Both sublattices mostly occupied; the cheaper legal pattern (B empty) dominates, so the
    # weight per cell is ≈ δ_B² while P ψ stays well conditioned.
    for (δA, δB) in ((1e-6, 1e-3), (1e-9, 1e-5))
        u1, u2 = [δA, 1.0], [δB, 1.0]
        ex = CDPXP.exact_leakage(u1, u2)
        @test ex > 3
        cell = [ComplexF64.(u1 ./ norm(u1)), ComplexF64.(u2 ./ norm(u2))]
        for k in (1, 2, 4)                       # tiling invariance
            ψ = product_iMPS(ComplexF64, repeat(cell, k))
            @test abs(constraint_leakage(ψ, _BLK) / ex - 1) < 1e-12
            ψs = deepcopy(ψ)                     # rescaling invariance (not normalized)
            ψs.Γ[1] .*= 1e-3
            ψs.Γ[end] .*= 7.0
            @test abs(constraint_leakage(ψs, _BLK) / ex - 1) < 1e-12
        end
    end
    # Uniform near full packing: nearly a two-fold crystal (transfer gap ~ δ). The leakage
    # needs no canonical form and stays accurate. Closed form: the site transfer
    # [w₀ w₁; w₀ 0] has Perron root (w₀ + √(w₀² + 4w₀w₁))/2.
    for δ in (1e-4, 1e-6), k in (1, 2, 8)
        w0, w1 = δ^2, 1 - δ^2
        exact = -log((w0 + sqrt(w0^2 + 4w0 * w1)) / 2) / 2
        ψ = product_iMPS(ComplexF64, [ComplexF64[δ, sqrt(1 - δ^2)] for _ in 1:k])
        @test abs(constraint_leakage(ψ, _BLK) / exact - 1) < 1e-11
    end
    # a correlated state with χ = 8 (Arnoldi path), damped towards occupation
    Random.seed!(5)
    ψ = rand_iMPS(ComplexF64, 2, 2, 8)
    apply!(ψ, iMPO([reshape(ComplexF64[1e-3 0; 0 1], 1, 2, 2, 1)]); maxdim=64, cutoff=0.0)
    ref = constraint_leakage(ψ, _BLK)
    @test 3 < ref < Inf
    @test all(abs(constraint_leakage(_tile(ψ, k), _BLK) / ref - 1) < 1e-12 for k in 2:3)
    ψs = deepcopy(ψ)
    ψs.Γ[2] .*= 1e-5
    @test abs(constraint_leakage(ψs, _BLK) / ref - 1) < 1e-12
end

@testset "exact annihilation" begin
    full = product_iMPS(ComplexF64, [[0, 1], [0, 1]])
    for P in (_BLK, _PMPO)
        @test constraint_leakage(full, P) == Inf
    end
    @test constraint_leakage(_tile(full, 3), _BLK) == Inf
end

@testset "scarfinder_step!: static inputs validated before mutation" begin
    ψ0 = apply!(CDPXP.product([cos(0.35), sin(0.35)], 2), _PMPO)
    e = energy_density(ψ0, CDPXP.terms())
    bad = [(; terms=CDPXP.terms(), target=NaN),
           (; terms=CDPXP.terms(), target=e, tol=-1.0),
           (; terms=CDPXP.terms(), target=e, maxstep=-1),
           (; target=e)]
    for kw in bad
        ψ = deepcopy(ψ0)
        @test_throws Exception scarfinder_step!(ψ, _UF, 2; nstep=4, maxdim=16, kw...)
        @test ψ.Γ == ψ0.Γ && ψ.λ == ψ0.λ
    end
end
