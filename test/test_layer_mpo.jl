using Test
using LinearAlgebra
using Random
using InfiniteTEBD

const _X = ComplexF64[0 1; 1 0]
const _P0 = ComplexF64[1 0; 0 0]
const _P1 = ComplexF64[0 0; 0 1]
const _I2 = Matrix{ComplexF64}(I, 2, 2)

_rand_unitary(rng, d) = Matrix(qr(randn(rng, ComplexF64, d, d)).Q)
function _rand_hermitian(rng, d)
    A = randn(rng, ComplexF64, d, d)
    return (A + A') / 2
end

# Gate controlled by (diagonal on) its outer sites: overlapping translates commute.
function _controlled(f)
    G = zeros(ComplexF64, 8, 8)
    for (a, Pa) in enumerate((_P0, _P1)), (b, Pb) in enumerate((_P0, _P1))
        G .+= kron(Pa, f(a, b), Pb)
    end
    return G
end

_tile(ψ, m) = iMPS(repeat([copy(A) for A in ψ.Γ], m), repeat([copy(l) for l in ψ.λ], m), ψ.n * m)

@testset "iMPO type" begin
    W = iMPO([reshape(_X, 1, 2, 2, 1)])
    @test W.n == 1 && bonddims(W) == [1]
    @test W[3] === W.W[1]
    @test_throws DimensionMismatch iMPO([zeros(1, 2, 2, 2), zeros(3, 2, 2, 1)])
    @test_throws ArgumentError iMPO(Array{Float64,4}[], 0)
    @test occursin("iMPO", sprint(show, W))
end

@testset "support convention" begin
    @test InfiniteTEBD._term_support(2, 4, 2) == (2, 4)
    @test InfiniteTEBD._term_support(4, 6, 2) == (2, 4)
    @test InfiniteTEBD._term_support(4, 1, 4) == (4, 5)
    @test InfiniteTEBD._term_support(3, 1, 4) == (3, 5)
    @test_throws ArgumentError layer_mpo([(kron(_X, _X), 1, 3)], 2)   # 4×4 gate on 3 sites
end

@testset "layer MPO equals sequential gates" begin
    rng = MersenneTwister(7)
    # one random controlled gate, repeated with period 2 (support longer than the cell)
    G = _controlled((a, b) -> _rand_unitary(rng, 2))
    W = layer_mpo([(G, 2, 4)], 2)
    @test bonddims(W) == [2, 2]

    ψ2 = rand_iMPS(ComplexF64, 2, 2, 3)
    ψ4 = _tile(ψ2, 2)
    apply!(ψ2, W; maxdim=1000, cutoff=1e-14)
    evolve!(ψ4, [(G, 2, 4), (G, 4, 2)], 1; maxdim=1000, cutoff=1e-14)
    canonical!(ψ4; maxdim=1000, cutoff=1e-14)
    @test abs(per_site_fidelity(_tile(ψ2, 2), ψ4) - 1) < 1e-12

    # two different gates in a four-site cell, overlapping on sites 2 and 4
    G1 = _controlled((a, b) -> _rand_unitary(rng, 2))
    G2 = _controlled((a, b) -> _rand_unitary(rng, 2))
    W4 = layer_mpo([(G1, 2, 4), (G2, 4, 6)], 4)
    @test bonddims(W4) == [2, 2, 2, 2]
    φ = rand_iMPS(ComplexF64, 4, 2, 2)
    χ = deepcopy(φ)
    apply!(φ, W4; maxdim=1000, cutoff=1e-14)
    evolve!(χ, [(G1, 2, 4), (G2, 4, 2)], 1; maxdim=1000, cutoff=1e-14)
    canonical!(χ; maxdim=1000, cutoff=1e-14)
    @test abs(per_site_fidelity(φ, χ) - 1) < 1e-12

    # full second-order Trotter steps: MPOs on a 2-site cell vs gates on a 4-site cell
    hA = _controlled((a, b) -> _rand_hermitian(rng, 2))
    hB = _controlled((a, b) -> _rand_hermitian(rng, 2))
    dt = 0.07
    U = trotter_mpos([[(hA, 2, 4)], [(hB, 1, 3)]], dt; n=2)
    @test length(U) == 3 && all(W -> bonddims(W) == [2, 2], U)
    ψ2 = rand_iMPS(ComplexF64, 2, 2, 2)
    ψ4 = _tile(ψ2, 2)
    evolve!(ψ2, U, 2; maxdim=1000, cutoff=1e-14)
    layers4 = [[(hA, 2, 4), (hA, 4, 2)], [(hB, 1, 3), (hB, 3, 1)]]
    evolve!(ψ4, layers4, dt, 2; maxdim=1000, cutoff=1e-14)
    canonical!(ψ4; maxdim=1000, cutoff=1e-14)
    @test abs(per_site_fidelity(_tile(ψ2, 2), ψ4) - 1) < 1e-12
end

@testset "MPO products and half-step merging" begin
    rng = MersenneTwister(11)
    h = _controlled((a, b) -> _rand_hermitian(rng, 2))
    half = layer_mpo([(exp(-0.05im * h), 2, 4)], 2)
    full = layer_mpo([(exp(-0.1im * h), 2, 4)], 2)
    exact = mpo_multiply(half, half; compress=false)
    @test bonddims(exact) == [4, 4]
    merged = half * half
    @test bonddims(merged) == [2, 2]
    for W in (exact, merged)
        ψa = rand_iMPS(ComplexF64, 2, 2, 3)
        ψb = deepcopy(ψa)
        apply!(ψa, W; maxdim=1000, cutoff=1e-14)
        apply!(ψb, full; maxdim=1000, cutoff=1e-14)
        @test abs(per_site_fidelity(ψa, ψb) - 1) < 1e-12
    end
    # the scale is kept, not only the direction: compare the operators per site
    z = InfiniteTEBD._mixed_cell_eigenvalue(
        [reshape(A, size(A, 1), 4, size(A, 4)) for A in merged.W],
        [reshape(A, size(A, 1), 4, size(A, 4)) for A in full.W])
    nf = InfiniteTEBD._mixed_cell_eigenvalue(
        [reshape(A, size(A, 1), 4, size(A, 4)) for A in full.W],
        [reshape(A, size(A, 1), 4, size(A, 4)) for A in full.W])
    @test z ≈ nf rtol=1e-12

    # evolve! with merging equals evolve! without it
    hB = _controlled((a, b) -> _rand_hermitian(rng, 2))
    U = trotter_mpos([[(h, 2, 4)], [(hB, 1, 3)]], 0.05; n=2)
    ψa = rand_iMPS(ComplexF64, 2, 2, 2)
    ψb = deepcopy(ψa)
    _, sa = evolve!(ψa, U, 3; maxdim=1000, cutoff=1e-14, return_stats=true)
    _, sb = evolve!(ψb, U, 3; maxdim=1000, cutoff=1e-14, merge=false, return_stats=true)
    @test length(sa.updates) == 7 && length(sb.updates) == 9
    @test abs(per_site_fidelity(ψa, ψb) - 1) < 1e-12
end

@testset "preconditions error clearly" begin
    nX = kron(_P1, _X)
    @test_throws ArgumentError check_commuting([(nX, 1, 2), (nX, 2, 3)])
    @test_throws ArgumentError layer_mpo([(nX, 1, 2)], 1)            # overlaps its own translate
    @test check_commuting([(kron(_P1, _P1), 1, 2), (kron(_P1, _P1), 2, 3)])
    G = _controlled((a, b) -> _X)
    @test_throws ArgumentError layer_mpo([(G, 1, 3), (G, 2, 4)], 4)  # bond 2 crossed twice
    @test_throws ArgumentError layer_mpo([(G, 1, 3)], 1)             # span > n + 1
    @test_throws ArgumentError trotter_mpos([[(nX, 1, 2), (nX, 2, 3)]], 0.1; n=4)
    ψ = rand_iMPS(ComplexF64, 3, 2, 2)
    @test_throws ArgumentError apply!(ψ, layer_mpo([(G, 2, 4)], 2))   # 3 is not a multiple of 2
    @test_throws ArgumentError apply!(rand_iMPS(Float64, 2, 2, 2), layer_mpo([(G * 1im, 2, 4)], 2))
end

@testset "truncate! treats every bond alike" begin
    ψ = rand_iMPS(ComplexF64, 2, 2, 8)
    ψ4 = _tile(ψ, 2)
    _, st = truncate!(ψ, 3; return_stats=true)
    @test length.(ψ.λ) == [3, 3]
    @test st.kept_dims == [3, 3]
    @test all(l -> norm(l) ≈ 1, ψ.λ)
    truncate!(ψ4, 3)
    @test abs(per_site_fidelity(_tile(ψ, 2), ψ4) - 1) < 1e-10
    @test ψ4.λ[1] ≈ ψ4.λ[3] atol=1e-10
    # truncerr picks the smallest admissible dimension per bond
    φ = rand_iMPS(ComplexF64, 2, 2, 6)
    _, st = truncate!(deepcopy(φ), 6; truncerr=1e-2, return_stats=true)
    @test all(s -> s.discarded_weight <= 1e-2, st.bond_stats)
end

@testset "site-dependent energy density" begin
    rng = MersenneTwister(3)
    hA = _controlled((a, b) -> _rand_hermitian(rng, 2))
    hB = _controlled((a, b) -> _rand_hermitian(rng, 2))
    ψ = rand_iMPS(ComplexF64, 2, 2, 3)
    e2 = energy_density(ψ, [(hA, 2, 4), (hB, 1, 3)])
    ψ4 = _tile(ψ, 2)
    canonical!(ψ4)
    e4 = energy_density(ψ4, [(hA, 2, 4), (hA, 4, 2), (hB, 1, 3), (hB, 3, 1)])
    @test e2 ≈ e4 atol=1e-12
    @test energy_density(ψ4, [(hA, 2, 4), (hB, 1, 3)]; n=2) ≈ e2 atol=1e-12
    @test energy_density(ψ4, [(hA, 2, 4), (hB, 1, 3)]; n=2) ≈
          (expect(ψ4, hA, 2, 4) + expect(ψ4, hA, 4, 2) + expect(ψ4, hB, 1, 3) + expect(ψ4, hB, 3, 1)) / 4 atol=1e-12
end

@testset "review regressions" begin
    # Boundary merging must respect time order: the last MPO of a step acts before the
    # first MPO of the next. X then Hadamard, twice, from |0⟩ ends in |1⟩ up to phase.
    H = ComplexF64[1 1; 1 -1] / sqrt(2)
    mx = iMPO([reshape(_X, 1, 2, 2, 1)])
    mh = iMPO([reshape(H, 1, 2, 2, 1)])
    for merge in (false, true)
        ψ = product_iMPS(ComplexF64, [[1, 0]])
        evolve!(ψ, [mx, mh], 2; merge)
        @test abs2(ψ.Γ[1][1, 2, 1]) ≈ 1 atol=1e-12
    end
    # same with random non-commuting single-site layers on a two-site cell
    rng = MersenneTwister(5)
    Ws = [iMPO([reshape(_rand_unitary(rng, 2), 1, 2, 2, 1), reshape(_rand_unitary(rng, 2), 1, 2, 2, 1)])
          for _ in 1:3]
    ψ0 = product_iMPS(ComplexF64, [[1, 0], [0.6, 0.8]])
    a = deepcopy(ψ0); evolve!(a, Ws, 3)
    b = deepcopy(ψ0); for _ in 1:3, W in Ws; apply!(b, W); end
    @test abs(per_site_fidelity(a, b) - 1) < 1e-12

    # Mixed-transfer fidelity must not depend on the virtual gauge: two copies of one state
    # related by a diagonal sign gauge (the identity Krylov seed used to give 0.5).
    D = 16
    id = Matrix(1.0I, D, D)
    signflip(j) = Diagonal([i == j ? -1.0 : 1.0 for i in 1:D])
    A = permutedims(cat(id, circshift(id, (1, 0)), signflip(1), signflip(3); dims=3), (1, 3, 2)) / 2
    G = Diagonal([isodd(i) ? -1.0 : 1.0 for i in 1:D])
    B = copy(A)
    for s in axes(A, 2)
        B[:, s, :] = G * A[:, s, :] * G
    end
    st(T) = iMPS([T], [ones(D) / sqrt(D)], 1)
    @test per_site_fidelity(st(A), st(B)) ≈ 1 atol=1e-12
    @test per_site_fidelity(st(A), st(A)) ≈ 1 atol=1e-12

    # An annihilating MPO throws a typed error and leaves the input unchanged.
    ψ = product_iMPS(ComplexF64, [[1, 0]])
    saved = deepcopy(ψ)
    @test_throws MPOAnnihilationError apply!(ψ, iMPO([reshape(_P1, 1, 2, 2, 1)]))
    @test ψ.Γ == saved.Γ && ψ.λ == saved.λ

    # renormalize=true with recanonicalize=false normalizes the state itself, not only the
    # stored spectra (Bell dimer truncated to bond dimension one).
    bell = zeros(ComplexF64, 1, 2, 2); bell[1, 1, 1] = 1; bell[1, 2, 2] = 1
    bell2 = zeros(ComplexF64, 2, 2, 1); bell2[1, 2, 1] = 1; bell2[2, 1, 1] = 1
    ψ = iMPS(ComplexF64, [bell, bell2])
    apply!(ψ, iMPO([reshape(_I2, 1, 2, 2, 1)]); maxdim=1, recanonicalize=false)
    @test inner_product(ψ) ≈ 1 atol=1e-12
end
