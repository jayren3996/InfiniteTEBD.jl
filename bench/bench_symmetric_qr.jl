using InfiniteTEBD
using TensorKit: TensorMap, ←, ⊗, permute, norm, domain, block, blocks, svd_trunc!,
                 truncrank, trunctol, id, dim, @tensor
using LinearAlgebra: BLAS, Hermitian
using Printf, Random, Statistics

const Ext = Base.get_extension(InfiniteTEBD, :InfiniteTEBDTensorKitExt)
BLAS.set_num_threads(1)
Random.seed!(6024)

function record(f, setup)
    f(setup())
    samples = map(1:5) do _
        input = setup()
        value = @timed f(input)
        (value.time * 1e3, value.bytes / 2.0^20)
    end
    return median(first.(samples)), median(last.(samples)), Sys.maxrss() / 2.0^20
end

split_trial(ψ) = begin
    trial = deepcopy(ψ)
    trial.Γ[1] = Ext._absorb_diag_left(trial.Γ[1], trial.λ[end])
    trial
end

function cut_comparison(ψ, P)
    A, B = ψ.Γ[1], ψ.Γ[2]
    λ = ψ.λ[end]
    physical = P ⊗ P
    raw = randn(ComplexF64, physical ← physical)
    G = exp(-0.002im * (raw + raw'))
    @tensor grouped[a,s,t;c] := λ[a;b] * A[b,s;m] * B[m,t;c]
    @tensor updated[a,s,t;c] := G[s,t;u,v] * grouped[a,u,v;c]
    M = permute(updated, ((1,2), (3,4)))
    seed = permute(B, ((1,), (2,3)))'
    cap = 16
    strategy = truncrank(cap) & trunctol(; atol=1e-12)
    exact(X) = svd_trunc!(X; trunc=strategy)
    qr_cut(X) = Ext._symmetric_qr_split(X, seed; maxdim=cap, projection_tol=1e-4, expansion=2)
    A0, S0, B0, _ = exact(copy(M))
    _, _, _, info = qr_cut(copy(M))
    exact_error = norm(M - A0*S0*B0)^2 / norm(M)^2
    @printf("weighted graded gate cut SVD=%.3f ms QR=%.3f ms accepted=%s projection=(%.6g, %.6g)\n",
            record(exact, () -> copy(M))[1], record(qr_cut, () -> copy(M))[1],
            string(info.used_qr), exact_error, info.projection_error)
end

function main()
    n = isempty(ARGS) ? 8 : parse(Int, only(ARGS))
    n > 0 && n % 4 == 0 || throw(ArgumentError("use a positive multiple of four"))
    P = graded_space(:U1, -2=>1, -1=>1, 0=>1, 1=>1, 2=>1)
    V = graded_space(:U1, -1=>4, 0=>8, 1=>4)
    base = Ext._rand_symmetric_iMPS(P, fill(V, 4))
    canonical!(base; maxdim=16)
    cell(k) = iMPS(repeat(base.Γ, k ÷ 4), repeat(base.λ, k ÷ 4), k)
    fixed(ψ) = (Ext._dominant_fixed_point(ψ; dir=:r, tol=1e-12, maxiter=200),
                Ext._dominant_fixed_point(ψ; dir=:l, tol=1e-12, maxiter=200))
    function iterative_fixed(ψ)
        V = domain(ψ.λ[end])[1]
        return map((:r, :l)) do dir
            f = R -> Ext._apply_transfer_unit_cell(ψ, R; dir)
            Ext.eigsolve(f, id(ComplexF64, V), min(4, dim(V)), :LM;
                         tol=1e-12, maxiter=200, ishermitian=false)
        end
    end
    split(ψ) = Ext._split_unit_cell!(ψ, ψ.λ[end]; maxdim=16, cutoff=1e-12, renormalize=true)
    full(ψ) = canonical!(ψ; maxdim=16)
    # Warm all kernels on a short cell before measuring high-water RSS.
    fixed(base)
    split(split_trial(base))
    full(deepcopy(base))
    ψ = cell(n)
    @printf("n=%d initial process peak=%.2f MiB\n", n, Sys.maxrss()/2.0^20)
    for (label, f, setup) in (("fixed (validated environments)", fixed, () -> ψ),
                              ("fixed (always Krylov)", iterative_fixed, () -> ψ),
                              ("split", split, () -> split_trial(ψ)),
                              ("canonical", full, () -> deepcopy(ψ)))
        @printf("%s %.3f ms, %.3f MiB allocations, process peak %.2f MiB\n",
                label, record(f, setup)...)
    end
    n == 8 && cut_comparison(base, P)
end

main()
