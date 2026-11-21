using InfiniteTEBD, LinearAlgebra, Printf, Statistics

BLAS.set_num_threads(1)

function clock_setup(d)
    Z = Diagonal(cis.(2π .* (0:d-1) ./ d))
    X = zeros(ComplexF64, d, d)
    for i in 1:d
        X[mod1(i+1,d), i] = 1
    end
    field = X + X'
    h = -(kron(Z, Z') + kron(Z', Z)) - 0.8 * (kron(field, I(d)) + kron(I(d), field))
    ψ = product_iMPS(ComplexF64, [vcat(1.0, zeros(d-1)) for _ in 1:2])
    return ψ, Z, Hermitian(Matrix(h))
end

function advance_clock(template, gates, h, Z; dt, cap, projection_tol=nothing, measure=false)
    ψ = deepcopy(template)
    trace = NTuple{3,Float64}[]
    accepted = 0
    unmet = 0
    stride = round(Int, 0.4 / dt)
    steps = round(Int, 2.4 / dt)
    for step in 1:steps
        for (G, i, j) in gates
            if isnothing(projection_tol)
                applygate!(ψ, G, i, j; maxdim=cap)
            else
                info = InfiniteTEBD._applygate_qr_experiment!(ψ, G, i, j; maxdim=cap, projection_tol)
                accepted += info.used_qr
                unmet += !info.target_met
            end
        end
        if measure && step % stride == 0
            q = canonical!(deepcopy(ψ); maxdim=cap, noninjective=:ignore)
            push!(trace, ((expect(q, Z, 1, 1) + expect(q, Z, 2, 2))/2,
                          energy_density(q, h), max(ent_S(q, 1), ent_S(q, 2))))
        end
    end
    return (; trace, accepted, unmet, updates=steps * length(gates))
end

clock_gates(h, dt) = ((exp(-0.5im*dt*h), 1, 2), (exp(-im*dt*h), 2, 1), (exp(-0.5im*dt*h), 1, 2))

function main()
    ψ, Z, h = clock_setup(5)
    refdt = 0.025
    ref = advance_clock(ψ, clock_gates(h, refdt), h, Z; dt=refdt, cap=40, measure=true).trace
    println("Clock quench to t=2.4. Six snapshots, sampled every 0.4. Reference dt=0.025, cap=40.")
    println("Errors are maximum absolute trace errors in Re Z, energy density, and max bond entropy.")
    for dt in (0.05, 0.025), cap in (12, 20)
        gates = clock_gates(h, dt)
        for projection_tol in (nothing, 1e-6, 1e-4)
            result = advance_clock(ψ, gates, h, Z; dt, cap, projection_tol, measure=true)
            advance_clock(ψ, gates, h, Z; dt, cap, projection_tol)
            timings = [@elapsed(advance_clock(ψ, gates, h, Z; dt, cap, projection_tol)) for _ in 1:3]
            errors = [maximum(abs(result.trace[i][j] - ref[i][j]) for i in eachindex(ref))
                      for j in 1:3]
            @printf("dt=%0.3f cap=%2d tol=%7s time=%7.3f ms qr=%3d/%3d unmet=%3d maxerr=(%.5g, %.5g, %.5g)\n",
                    dt, cap, string(projection_tol), 1e3*median(timings),
                    result.accepted, result.updates, result.unmet, errors...)
        end
    end
end

main()
