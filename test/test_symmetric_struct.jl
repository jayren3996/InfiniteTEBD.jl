using Test
using InfiniteTEBD

@testset "iMPS parametric struct" begin
    @testset "dense alias" begin
        ψ = rand_iMPS(ComplexF64, 2, 2, 3)
        @test ψ isa DenseIMPS
        @test ψ isa DenseIMPS{ComplexF64, Float64}
        @test eltype(ψ.Γ[1]) === ComplexF64
        @test eltype(ψ.λ[1]) === Float64
        @test typeof(ψ).parameters[1] === Array{ComplexF64, 3}
        @test typeof(ψ).parameters[2] === Vector{Float64}
    end

    @testset "struct exposes Γ, λ, n fields" begin
        ψ = rand_iMPS(ComplexF64, 2, 2, 3)
        @test fieldnames(iMPS) === (:Γ, :λ, :n)
        @test ψ.n == 2
    end
end

@testset "TensorKit-extension stubs (fresh process without TensorKit)" begin
    # `graded_space(:U1, …)` without TensorKit gives the actionable
    # "load TensorKit" error rather than a confusing MethodError. Loading
    # TensorKit anywhere in the test session activates the extension for the
    # rest of the session, and its `graded_space(::Symbol, …)` method shadows
    # the stub — so asserting in-process would depend on which test files ran
    # first (test_symmetric_basic.jl does `using TensorKit`). Assert in a
    # fresh Julia process where only InfiniteTEBD is loaded, keeping this
    # testset independent of file order.
    stub_check = raw"""
        using InfiniteTEBD
        Base.get_extension(InfiniteTEBD, :InfiniteTEBDTensorKitExt) === nothing ||
            error("precondition broken: TensorKit extension loaded in a bare process")
        err = try
            graded_space(:U1, 0=>1)
            nothing
        catch e
            e
        end
        err isa ErrorException || error("expected ErrorException, got $(repr(err))")
        occursin("TensorKit", err.msg) ||
            error("stub error does not mention TensorKit: $(err.msg)")
        """
    # Reuse the parent's julia and project, but drop coverage/allocation
    # flags: they disable native pkgimage loading, which would make this
    # one-off subprocess re-load the whole dependency stack uncached when
    # tests run with coverage enabled (as CI does).
    julia_exec = filter(Base.julia_cmd().exec) do flag
        !startswith(flag, "--code-coverage") && !startswith(flag, "--track-allocation")
    end
    cmd = `$(Cmd(julia_exec)) --startup-file=no --project=$(Base.active_project()) -e $stub_check`
    stub_out = IOBuffer()
    stub_ok = success(pipeline(cmd; stdout=stub_out, stderr=stub_out))
    stub_ok || @error "no-TensorKit stub check failed" output=String(take!(stub_out))
    @test stub_ok
end

@testset "dense schmidt_values returns a fresh copy" begin
    # `schmidt_values` on the dense backend returns a fresh `Vector{Float64}`.
    # Verify the type, the values, and — critically — that mutating the result
    # does not bleed back into `ψ.λ[i]`. Previously this was a thin
    # `convert(Vector{Float64}, ψ.λ[i])` wrapper that aliased when the eltype
    # already matched, so a caller running `sv = schmidt_values(ψ, 1); sv .= 0`
    # would silently zero out the state's Schmidt spectrum.
    ψ = rand_iMPS(ComplexF64, 2, 2, 3)
    sv = schmidt_values(ψ, 1)
    @test sv isa Vector{Float64}
    @test sv == ψ.λ[1]
    original = copy(ψ.λ[1])
    sv .= 0.0
    @test ψ.λ[1] == original
end
