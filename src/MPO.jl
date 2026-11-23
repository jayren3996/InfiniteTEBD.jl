#---------------------------------------------------------------------------------------------------
# INFINITE MPO
#---------------------------------------------------------------------------------------------------
export iMPO, bonddims, layer_mpo, check_commuting, compress_mpo, mpo_multiply
export apply!, trotter_mpos, truncate!, energy_fix!, constraint_leakage, MPOAnnihilationError

"""
    CanonicalFormError

Thrown when the QR/LQ iteration that restores the canonical form after an MPO
application (or for a projected state) does not converge, typically for states
with a very long correlation length. It signals a solver failure, not a
property of the state.
"""
struct CanonicalFormError <: Exception
    msg::String
end
Base.showerror(io::IO, e::CanonicalFormError) = print(io, "CanonicalFormError: ", e.msg)
export CanonicalFormError

"""
    MPOAnnihilationError

Thrown by [`apply!`](@ref) (and the MPO [`evolve!`](@ref)) when the MPO maps the
state to zero, i.e. to a norm per cell at the roundoff level of the contraction. The state passed to `apply!` is left unchanged. A search can catch this
type to tell "candidate excluded by a projector" apart from a solver failure.
"""
struct MPOAnnihilationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::MPOAnnihilationError) = print(io, "MPOAnnihilationError: ", e.msg)

"""Default relative singular-value cutoff used when splitting gates into MPOs."""
const MPO_CUTOFF::Float64 = 1e-14

"""
    ControlledCenter

Structure of a two-site-cell layer of three-site gates whose outer sites are
controls: every gate on `(control, center, control)` has the form
`G = Σ_{a,b} |a⟩⟨a| ⊗ M[:, :, a, b] ⊗ |b⟩⟨b|`, i.e. it is diagonal in the two
outer occupations. Examples are PXP gates `exp(-i dt (P X P + c n))`, with or
without the blockade projector `P3` folded in, and their imaginary-time
versions. [`layer_mpo`](@ref) detects this structure and stores it in the
[`iMPO`](@ref); [`apply!`](@ref) then uses a sector-resolved update (see the
"Layer MPOs" manual page). Internal; only `control`, `center` and `M` are read.
"""
struct ControlledCenter
    control::Int
    center::Int
    M::Array{ComplexF64,4}     # M[s_out, s_in, a, b]
end

"""
    iMPO{T}

Infinite matrix-product operator with a periodic unit cell of `n` sites.

Fields:
- `W::Vector{Array{T,4}}`
  One tensor per unit-cell site, indexed `W[i][l, o, s, r]`: left bond `l`,
  output (bra) physical index `o`, input (ket) physical index `s`, right bond
  `r`. Shape `(Dl, d_out, d_in, Dr)`. The bond to the right of site `i` must
  match the bond to the left of site `i + 1`, with periodic wraparound.
- `n::Int`
  Unit-cell length.

Indexing `W[i]` returns the tensor of site `mod1(i, n)`. The operator acts on
an iMPS whose unit cell is a multiple of `n`; see [`apply!`](@ref).

Construct from a list of tensors with `iMPO(Ws)`, or from commuting local gates
with [`layer_mpo`](@ref) and [`trotter_mpos`](@ref). `A * B` is the operator
product (`B` acts first); see [`mpo_multiply`](@ref).

Example:
```julia
X = ComplexF64[0 1; 1 0]
W = iMPO([reshape(X, 1, 2, 2, 1)])   # the product operator ⊗_j X_j
bonddims(W)                          # [1]
```
"""
struct iMPO{T<:Number}
    W::Vector{Array{T,4}}
    n::Int
    # Optional layer structure retained from construction (`nothing` for general MPOs,
    # products and compressed MPOs). Only used to select a faster, equivalent update.
    structure::Union{Nothing,ControlledCenter}
    # Copy of the tensors the structure was derived from. The structure is used only while
    # `W` still equals it, so editing tensors through `W[i]` disables the fast path.
    snapshot::Vector{Array{T,4}}

    function iMPO(W::Vector{Array{T,4}}, n::Integer,
                  structure::Union{Nothing,ControlledCenter}=nothing) where {T<:Number}
        n > 0 || throw(ArgumentError("iMPO unit-cell length n must be positive (got $n)"))
        length(W) == n || throw(ArgumentError("iMPO: length(W) = $(length(W)) but n = $n"))
        for i in 1:n
            j = mod1(i + 1, n)
            size(W[i], 4) == size(W[j], 1) || throw(DimensionMismatch(
                "iMPO bond $i: size(W[$i], 4) = $(size(W[i], 4)) but " *
                "size(W[$j], 1) = $(size(W[j], 1))"))
            all(isfinite, W[i]) || throw(ArgumentError("iMPO tensor W[$i] contains non-finite values"))
        end
        snapshot = isnothing(structure) ? Array{T,4}[] : [copy(A) for A in W]
        return new{T}(W, Int(n), structure, snapshot)
    end
end

function iMPO(Ws::AbstractVector{<:AbstractArray{<:Number,4}})
    isempty(Ws) && throw(ArgumentError("iMPO needs at least one tensor"))
    T = mapreduce(eltype, promote_type, Ws)
    return iMPO([Array{T,4}(A) for A in Ws], length(Ws))
end

Base.eltype(::iMPO{T}) where {T} = T
Base.length(W::iMPO) = W.n
Base.getindex(W::iMPO, i::Integer) = W.W[mod1(i, W.n)]
Base.copy(W::iMPO) = iMPO([copy(A) for A in W.W], W.n, _valid_structure(W))

# The stored structure if it still describes the tensors (no edit since construction).
_valid_structure(W::iMPO) =
    !isnothing(W.structure) && W.W == W.snapshot ? W.structure : nothing

"""
    bonddims(W::iMPO)

Bond dimensions of `W`; entry `i` is the bond to the right of site `i`.
"""
bonddims(W::iMPO) = [size(A, 4) for A in W.W]

function Base.show(io::IO, W::iMPO{T}) where {T}
    print(io, "iMPO{$T}(n=$(W.n), d=$([size(A, 3) for A in W.W]), bonddims=$(bonddims(W)))")
end

# The site tensors over `m` sites, cycling through the cell.
_tiled_tensors(W::iMPO, m::Integer) = [W[i] for i in 1:m]

#---------------------------------------------------------------------------------------------------
# Supports longer than the cell
#---------------------------------------------------------------------------------------------------
"""
    _term_support(i, j, n)

Absolute support `(i, j)` with `i ≤ j` and `i ∈ 1:n` of a local term.

Convention (used by [`layer_mpo`](@ref), [`trotter_mpos`](@ref),
[`check_commuting`](@ref) and the site-dependent [`energy_density`](@ref)):
`(i, j)` with `i ≤ j` denotes the chain sites `i, i+1, …, j`, which may be
more sites than the unit cell holds; site `k` is unit-cell site `mod1(k, n)`.
The term is repeated with period `n`. The legacy wraparound form `j < i` of
`trotter_gates` is accepted and means `j + n` (for example `(4, 1)` in a
four-site cell is the pair of sites 4 and 5).
"""
function _term_support(i::Integer, j::Integer, n::Integer)
    n > 0 || throw(ArgumentError("unit-cell length must be positive"))
    j < i && (j += n * cld(i - j, n))
    shift = fld(i - 1, n) * n
    return Int(i - shift), Int(j - shift)
end

function _support_span(G::AbstractMatrix, i::Integer, j::Integer, n::Integer, d)
    size(G, 1) == size(G, 2) || throw(ArgumentError("local operators must be square"))
    a, b = _term_support(i, j, n)
    L = b - a + 1
    D = size(G, 1)
    dd = isnothing(d) ? round(Int, D^(1 / L)) : Int(d)
    dd^L == D || throw(ArgumentError(
        "operator of size $D cannot act on the $L sites $i:$j" *
        (isnothing(d) ? "" : " of local dimension $dd") *
        ". Supports longer than the cell are written with i ≤ j, e.g. (2, 4) for three sites " *
        "of a two-site cell"))
    return a, b, L, dd
end

# Uniform local dimension shared by every term in `terms`.
function _uniform_local_dim(terms, n, d)
    dd = d
    for (G, i, j) in terms
        _, _, _, dg = _support_span(G, i, j, n, dd)
        isnothing(dd) && (dd = dg)
    end
    isnothing(dd) && throw(ArgumentError("cannot infer a local dimension from an empty layer"))
    return dd
end

_dense(G::AbstractMatrix) = Matrix(G)

#---------------------------------------------------------------------------------------------------
# Commutation check
#---------------------------------------------------------------------------------------------------
# Embed G (kron convention) acting on sites a:b into the interval lo:hi.
function _embed(G::AbstractMatrix, a, b, lo, hi, d)
    left = d^(a - lo)
    right = d^(hi - b)
    M = _dense(G)
    left > 1 && (M = kron(Matrix{eltype(M)}(I, left, left), M))
    right > 1 && (M = kron(M, Matrix{eltype(M)}(I, right, right)))
    return M
end

"""
    check_commuting(layer; n=nothing, atol=1e-12, d=nothing)

Check numerically that the overlapping gates of `layer` commute, and throw an
`ArgumentError` naming the first offending pair otherwise. Returns `true`.

`layer` is a list of `(G, i, j)` tuples. Supports follow the convention of
[`layer_mpo`](@ref): `i ≤ j` are chain sites, possibly more than one cell.
With `n` given, the layer is treated as periodic with period `n`, and every
translate of every gate is tested against every overlapping translate
(including a gate against its own translates). Without `n`, only the listed
gates are compared, at the listed positions.

The test is `‖AB − BA‖ ≤ atol · ‖A‖ · ‖B‖` (Frobenius norms) on the union of
the two supports. Gates that do not overlap always commute.

Example:
```julia
n = [0 0; 0 1]; X = [0 1; 1 0]
check_commuting([(kron(n, X), 1, 2), (kron(n, X), 2, 3)])  # throws: X and n meet on site 2
```
"""
function check_commuting(layer; n::Union{Nothing,Integer}=nothing, atol::Real=1e-12,
                         d::Union{Nothing,Integer}=nothing)
    terms = collect(layer)
    isempty(terms) && return true
    period = isnothing(n) ? 0 : Int(n)
    supports = map(terms) do (G, i, j)
        if isnothing(n)
            i <= j || throw(ArgumentError("without n, supports must satisfy i ≤ j (got ($i, $j))"))
            L = j - i + 1
            dd = isnothing(d) ? round(Int, size(G, 1)^(1 / L)) : d
            dd^L == size(G, 1) || throw(ArgumentError("operator size does not match support ($i, $j)"))
            (Int(i), Int(j), dd)
        else
            a, b, _, dd = _support_span(G, i, j, n, d)
            (a, b, dd)
        end
    end
    dd = supports[1][3]
    all(s -> s[3] == dd, supports) || throw(ArgumentError("check_commuting requires a uniform local dimension"))
    for p in eachindex(terms), q in p:length(terms)
        ap, bp, _ = supports[p]
        aq, bq, _ = supports[q]
        shifts = if period == 0
            p == q ? Int[] : [0]
        else
            # translates of q by k·n that overlap p; skip p against itself
            kmin = cld(ap - bq, period)
            kmax = fld(bp - aq, period)
            [k * period for k in kmin:kmax if !(p == q && k == 0)]
        end
        for s in shifts
            a2, b2 = aq + s, bq + s
            max(ap, a2) <= min(bp, b2) || continue
            lo, hi = min(ap, a2), max(bp, b2)
            A = _embed(terms[p][1], ap, bp, lo, hi, dd)
            B = _embed(terms[q][1], a2, b2, lo, hi, dd)
            err = norm(A * B - B * A)
            err <= atol * norm(A) * norm(B) || throw(ArgumentError(
                "gates $p (sites $ap:$bp) and $q (sites $a2:$b2) do not commute " *
                "(‖[A, B]‖ = $err). layer_mpo requires overlapping gates of a layer to commute; " *
                "put them in different layers"))
        end
    end
    return true
end

#---------------------------------------------------------------------------------------------------
# Gate → MPO
#---------------------------------------------------------------------------------------------------
# Operator Schmidt decomposition of an L-site gate (kron convention) into L
# tensors (Dl, d, d, Dr), cutting every internal bond by an SVD and dropping
# singular values below `cutoff` times the largest one on that cut.
function _gate_mpo_pieces(G::AbstractMatrix, d::Integer, L::Integer, cutoff::Real)
    T = promote_type(eltype(G), Float64)
    M = Matrix{T}(G)
    L == 1 && return [reshape(Matrix{T}(M), 1, d, d, 1)]
    # reshape gives indices (o_L, …, o_1, i_L, …, i_1); reorder to (o_1, i_1, …, o_L, i_L)
    tensor = reshape(M, ntuple(_ -> d, 2L))
    perm = Int[]
    for p in 1:L
        push!(perm, L - p + 1, 2L - p + 1)
    end
    rest = reshape(permutedims(tensor, perm), d * d, :)
    pieces = Vector{Array{T,4}}(undef, L)
    Dl = 1
    for p in 1:L-1
        F = svd(reshape(rest, Dl * d * d, :))
        smax = isempty(F.S) ? zero(real(T)) : F.S[1]
        k = max(1, count(>(cutoff * smax), F.S))
        pieces[p] = reshape(F.U[:, 1:k], Dl, d, d, k)
        rest = F.S[1:k] .* F.Vt[1:k, :]
        Dl = k
    end
    pieces[L] = reshape(rest, Dl, d, d, 1)
    return pieces
end

# Inverse of `_gate_mpo_pieces`: the L-site operator (kron convention) the pieces represent.
function _contract_pieces(pieces, d)
    L = length(pieces)
    acc = reshape(pieces[1], d * d, size(pieces[1], 4))          # (o1 i1, r)
    for p in 2:L
        A = pieces[p]
        acc = reshape(acc * reshape(A, size(A, 1), :), :, size(A, 4))   # ((o i)_1..p, r)
    end
    t = reshape(acc, ntuple(_ -> d, 2L))                         # (o1, i1, o2, i2, …)
    perm = vcat([2(L - p) - 1 for p in 0:L-1], [2(L - p) for p in 0:L-1])
    return reshape(permutedims(t, perm), d^L, d^L)
end

# W_new = P ∘ W (P acts after W); bonds fuse as (bond of W) fastest.
function _compose_site(P::AbstractArray{<:Number,4}, W::AbstractArray{<:Number,4})
    @tensor C[lw, lp, o, i, rw, rp] := P[lp, o, m, rp] * W[lw, m, i, rw]
    s = size(C)
    return reshape(C, s[1] * s[2], s[3], s[4], s[5] * s[6])
end

"""
    layer_mpo(layer, n; cutoff=1e-14, check=true, d=nothing, structured=true)

Build the exact infinite MPO of one commuting layer of local gates, repeated
with period `n`.

`layer` is a list of `(G, i, j)` tuples, as in a layer of
[`trotter_gates`](@ref): a dense gate `G` (in the `kron` convention, leftmost
site = leftmost factor) on the contiguous sites `i..j`.

Support convention:
- `(i, j)` with `i ≤ j` are chain sites `i, i+1, …, j`, and site `k` lives on
  unit-cell site `mod1(k, n)`. A support may be longer than the cell: in a
  two-site cell, the three-site term centred on site 1 is `(2, 4)` (sites
  2, 1, 2 of the cell) and the one centred on site 2 is `(1, 3)`. In general
  a term of span `L` centred on `c` is `(c - (L-1)÷2, c + L÷2)`; any shift by
  a multiple of `n` names the same term.
- `j < i` is the legacy wraparound of `trotter_gates` and means `j + n`.
- The gate size must equal `d^L` with `L = j - i + 1` (checked).

Precondition (the supported case):
- Gates may overlap, including a gate with its own translates by `n`, but only
  on sites where they commute. `check=true` verifies this with
  [`check_commuting`](@ref) and throws otherwise.
- Each bond of the cell is crossed by at most one gate (counting translates).
  This also bounds the span by `n + 1`. A violation throws an `ArgumentError`.

Method: each gate is split by an operator Schmidt decomposition (an SVD
across each internal bond, dropping singular values below `cutoff` times the
largest on that cut). Where gate pieces share a site, their single-site
operators are multiplied in layer order (the first listed gate acts first).
Bonds that no gate crosses get bond dimension 1. Since every bond is crossed
by one gate, the MPO bond dimension equals the operator Schmidt rank of that
gate across that bond, and the MPO equals the product of all gates exactly
(up to the dropped singular values).

Returns an [`iMPO`](@ref) with cell length `n`.

Structure: with `structured=true`, a layer of one three-site gate on a
two-site cell (local dimension 2) that is diagonal in its two outer sites —
PXP-type gates, with or without a folded blockade projector, real or imaginary
time — is recorded as a `ControlledCenter` in the returned MPO.
[`apply!`](@ref) then uses an equivalent sector-resolved update (much faster
for PXP); all other layers, and `structured=false`, use the generic update.

Example (PXP: each gate flips the centre only if both neighbours are empty, so
the operator Schmidt rank is 2 on every bond):
```julia
X = [0 1; 1 0]; P = [1 0; 0 0]
U = exp(-0.1im * kron(P, X, P))
W = layer_mpo([(U, 2, 4)], 2)   # gates centred on the odd sites of a 2-site cell
bonddims(W)                     # [2, 2]
```
"""
function layer_mpo(layer, n::Integer; cutoff::Real=MPO_CUTOFF, check::Bool=true,
                   d::Union{Nothing,Integer}=nothing, structured::Bool=true)
    n > 0 || throw(ArgumentError("unit-cell length n must be positive"))
    isfinite(cutoff) && cutoff >= 0 || throw(ArgumentError("cutoff must be finite and non-negative"))
    terms = collect(layer)
    isempty(terms) && throw(ArgumentError("layer_mpo needs at least one gate"))
    dd = _uniform_local_dim(terms, n, d)
    check && check_commuting(terms; n, d=dd)

    T = mapreduce(t -> eltype(t[1]), promote_type, terms)
    T = promote_type(T, Float64)
    # Which gate crosses each cell bond (bond b sits right of site b).
    crossing = zeros(Int, n)
    pieces = Vector{Vector{Array{T,4}}}(undef, length(terms))
    at_site = [Tuple{Int,Int}[] for _ in 1:n]
    for (g, (G, i, j)) in enumerate(terms)
        a, b, L, _ = _support_span(G, i, j, n, dd)
        pieces[g] = [Array{T,4}(A) for A in _gate_mpo_pieces(G, dd, L, cutoff)]
        for p in 1:L
            c = mod1(a + p - 1, n)
            push!(at_site[c], (g, p))
            if p < L
                crossing[c] == 0 || throw(ArgumentError(
                    "bond $c of the cell is crossed by gate $(crossing[c]) and by gate $g " *
                    "(or by two translates of one gate). layer_mpo supports at most one " *
                    "gate across each bond; split the layer, or use a larger cell n"))
                crossing[c] = g
            end
        end
    end

    W = Vector{Array{T,4}}(undef, n)
    for c in 1:n
        # Layer order: earlier gates act first. Among the pieces of one gate on
        # a site (span n + 1), the left translate's piece acts first.
        order = sort(at_site[c]; by=gp -> (gp[1], -gp[2]))
        site = reshape(Matrix{T}(I, dd, dd), 1, dd, dd, 1)
        for (g, p) in order
            site = _compose_site(pieces[g][p], site)
        end
        W[c] = site
    end
    # The structure is read from the stored pieces (not the input gate), so it describes the
    # operator actually stored even when `cutoff` truncated the gate.
    structure = structured && length(terms) == 1 ?
        _detect_controlled_center([(_contract_pieces(pieces[1], dd), terms[1][2], terms[1][3])], n, dd) :
        nothing
    return iMPO(W, n, structure)
end

#---------------------------------------------------------------------------------------------------
# MPO algebra
#---------------------------------------------------------------------------------------------------
"""
    compress_mpo(W; cutoff=1e-14)

Remove redundant bond dimension from an [`iMPO`](@ref).

`W` is treated as an iMPS in operator space (physical leg `d_out · d_in`) and
brought to canonical form; operator Schmidt values below `cutoff` (relative to
a unit Hilbert-Schmidt norm per cell) are dropped. The complex scale per cell
is restored afterwards, so the result represents the same operator, not just
the same operator up to normalization. If no bond shrinks, `W` itself is
returned unchanged.

This is meant for exact (lossless up to `cutoff`) compression, e.g. after a
product of MPOs. It assumes the operator, viewed as a state, is injective.
"""
function compress_mpo(W::iMPO; cutoff::Real=MPO_CUTOFF)
    all(==(1), bonddims(W)) && return W
    T = promote_type(eltype(W), Float64)
    dims = [(size(A, 2), size(A, 3)) for A in W.W]
    Γs = [reshape(Array{T}(A), size(A, 1), size(A, 2) * size(A, 3), size(A, 4)) for A in W.W]
    ψ = iMPS(copy(Γs), [ones(real(T), size(A, 3)) for A in Γs], W.n)
    _canonical_truncate!(ψ; maxdim=typemax(Int), cutoff, truncerr=0.0, renormalize=true)[1]
    newdims = [length(λ) for λ in ψ.λ]
    all(newdims .>= bonddims(W)) && return W
    z = _dominant_chain_eigenvalue(ψ.Γ, Γs; dense_threshold=0, dense_size=64, tol=1e-14)
    # Keep the input operator if its scale could not be recovered.
    isfinite(z) && !iszero(z) || return W
    out = [reshape(Array{T}(ψ.Γ[i]), size(ψ.Γ[i], 1), dims[i]..., size(ψ.Γ[i], 3)) for i in 1:W.n]
    out[1] .*= T <: Real ? real(z) : z
    return iMPO(out, W.n)
end

"""
    mpo_multiply(A, B; compress=true, cutoff=1e-14)
    A * B

Operator product `A · B` of two [`iMPO`](@ref)s (`B` acts first). The result
has cell length `lcm(A.n, B.n)` and bond dimensions multiply. With
`compress=true` (the default, also used by `*`) the product is passed through
[`compress_mpo`](@ref) at the stated `cutoff`, which for instance turns
`U_A(dt/2) * U_A(dt/2)` back into a bond-dimension-2 representation of
`U_A(dt)`.
"""
function mpo_multiply(A::iMPO, B::iMPO; compress::Bool=true, cutoff::Real=MPO_CUTOFF)
    m = lcm(A.n, B.n)
    W = Vector{Array{promote_type(eltype(A), eltype(B)),4}}(undef, m)
    for i in 1:m
        size(A[i], 3) == size(B[i], 2) || throw(DimensionMismatch(
            "site $i: A has input dimension $(size(A[i], 3)) but B has output dimension $(size(B[i], 2))"))
        W[i] = _compose_site(A[i], B[i])
    end
    P = iMPO(W, m)
    return compress ? compress_mpo(P; cutoff) : P
end

Base.:*(A::iMPO, B::iMPO) = mpo_multiply(A, B)

#---------------------------------------------------------------------------------------------------
# Truncation in canonical form
#---------------------------------------------------------------------------------------------------
function _validate_mpo_truncation(maxdim, cutoff, truncerr)
    maxdim isa Integer && maxdim > 0 || throw(ArgumentError("maxdim must be a positive integer"))
    isfinite(cutoff) && cutoff >= 0 || throw(ArgumentError("cutoff must be finite and non-negative"))
    isfinite(truncerr) && truncerr >= 0 || throw(ArgumentError("truncerr must be finite and non-negative"))
    return nothing
end

#---------------------------------------------------------------------------------------------------
# Single-layer canonical form (Vanderstraeten, Haegeman, Verstraete, SciPost Phys. Lect. Notes 7
# (2019), Sec. 2.3, Algorithms 1 and 2)
#---------------------------------------------------------------------------------------------------
# `canonical!` follows Orús and Vidal (PRB 78, 155117): it solves for the transfer fixed points
# l, r and takes their square roots. Their eigenvalues are squared Schmidt values, so precision
# drops to min(√ε, ε/√η) for Schmidt weights η, and the inverses of the square roots leave
# tensors that violate orthonormality at the same order. After an MPO application the small
# values matter. The single-layer route never forms l or r: it iterates (A_L, L) ← QRPos(L A)
# around the cell until the seam factor L is stationary (unique by the positive diagonal),
# accelerated by an Arnoldi step on the mixed transfer map, then does the same from the left
# on the right-isometric chain and diagonalizes the bond centers by SVD.
#
# Bonds produced by an MPO often carry exact null directions. A zero pivot makes the QR
# factor non-unique, so they are removed first by an exact re-representation of the cell:
# a rank-revealing SVD sweep writes the cell as C₀ B₁⋯Bₙ (or A₁⋯Aₙ D₀), and absorbing the seam
# factor into the cell gives a cell with reduced bonds for the same infinite state. Sweeps
# alternate direction until no bond shrinks.

const _ORTH_TOL = 1e-13
const _ORTH_MAXITER = 2000
# The seam residual has a roundoff floor of a few ε·dim. A residual that stops decreasing
# for `_ORTH_STALL` sweeps below `_ORTH_FLOOR` is accepted as converged.
const _ORTH_STALL = 25
const _ORTH_FLOOR = 1e-10

# Convergence bookkeeping: returns (done, best, since_best).
function _orth_progress(δ, best, since, tol)
    δ <= tol && return true, δ, 0
    δ < best && return false, δ, 0
    since += 1
    return since >= _ORTH_STALL && best <= _ORTH_FLOOR, best, since
end

# Phases making the diagonal of a triangular factor real and non-negative. A pivot at the
# roundoff level gets phase 1, so null directions do not stop the iteration.
function _pivot_phases(T::AbstractMatrix)
    p = diag(T)
    tiny = 100 * eps(real(eltype(T))) * max(norm(T), floatmin(real(eltype(T))))
    return [abs(x) > tiny ? x / abs(x) : one(x) for x in p]
end

# M = L Q with L lower triangular (non-negative diagonal) and Q row-orthonormal.
function _pos_lq(M::AbstractMatrix)
    F = lq(M)
    L = Matrix(F.L)
    Q = Matrix(F.Q)
    ph = _pivot_phases(L)
    L .*= transpose(conj.(ph))
    Q .*= ph
    return L, Q
end

# M = Q R with R upper triangular (non-negative diagonal) and Q column-orthonormal.
function _pos_qr(M::AbstractMatrix)
    F = qr(M)
    k = min(size(M)...)
    Q = Matrix(F.Q)[:, 1:k]
    R = Matrix(F.R)
    ph = _pivot_phases(R)
    R .*= conj.(ph)
    Q .*= transpose(ph)
    return Q, R
end

const _NULL_RTOL = 1e-14

function _revealing_svd(M::AbstractMatrix, rtol)
    F = try
        svd(M)
    catch err
        err isa LAPACKException || rethrow()
        svd(M; alg=LinearAlgebra.QRIteration())
    end
    k = max(1, count(>(rtol * F.S[1]), F.S))
    return F.U[:, 1:k], F.S[1:k], F.Vt[1:k, :]
end

function _checked_norm(S)
    nrm = norm(S)
    isfinite(nrm) || throw(ArgumentError("canonical form: non-finite tensor entries"))
    nrm > 0 || throw(MPOAnnihilationError("the state has zero norm (the MPO annihilates it)"))
    return nrm
end

# One rank-revealing sweep, then the seam factor is absorbed into the cell. Returns the new
# cell and the log of the scale removed.
function _reduce_sweep(A, dir::Symbol, rtol)
    n = length(A)
    A = copy(A)
    logc = 0.0
    if dir === :right
        χ = size(A[n], 3)
        C = Matrix{eltype(A[1])}(I, χ, χ)
        for i in n:-1:1
            l, d, r = size(A[i])
            U, S, Vt = _revealing_svd(reshape(reshape(A[i], l * d, r) * C, l, :), rtol)
            A[i] = reshape(Vt, length(S), d, :)
            nrm = _checked_norm(S)
            logc += log(nrm)
            C = U .* transpose(S ./ nrm)
        end
        l, d, r = size(A[n])            # ⋯ C₀ B₁⋯Bₙ C₀ B₁⋯ : put C₀ to the right of Bₙ
        A[n] = reshape(reshape(A[n], l * d, r) * C, l, d, :)
    else
        χ = size(A[1], 1)
        D = Matrix{eltype(A[1])}(I, χ, χ)
        for i in 1:n
            l, d, r = size(A[i])
            U, S, Vt = _revealing_svd(reshape(D * reshape(A[i], l, :), size(D, 1) * d, r), rtol)
            A[i] = reshape(U, size(D, 1), d, :)
            nrm = _checked_norm(S)
            logc += log(nrm)
            D = (S ./ nrm) .* Vt
        end
        l, d, r = size(A[1])            # ⋯ A₁⋯Aₙ D₀ A₁⋯ : put D₀ to the left of A₁
        A[1] = reshape(D * reshape(A[1], l, :), size(D, 1), d, :)
    end
    return A, logc
end

function _remove_null_directions(A; rtol=_NULL_RTOL, maxpasses=20)
    logc = 0.0
    dims = [size(T, 3) for T in A]
    for _ in 1:maxpasses
        A, c1 = _reduce_sweep(A, :right, rtol)
        A, c2 = _reduce_sweep(A, :left, rtol)
        logc += c1 + c2
        new = [size(T, 3) for T in A]
        new == dims && return A, logc
        dims = new
    end
    return A, logc
end

# The sweeps only need the triangular factors; the isometries are needed for the Arnoldi
# map and for the converged output. Each sweep therefore keeps the Householder
# factorizations (`lq!`/`qr!` on the freshly formed products) and the pivot phases, and the
# isometries are formed on demand from exactly those factorizations, with the same phase
# rule as `_pos_lq`/`_pos_qr`. This roughly halves the allocations of a canonicalization.

# Right-to-left sweep: A₁⋯Aₙ C = c C' B₁⋯Bₙ with right-isometric B and lower-triangular C'.
# Returns the factorizations (for `_right_isometries`), C' and log c.
function _lq_sweep(A, C)
    packs = Vector{Any}(undef, length(A))
    logc = 0.0
    for i in length(A):-1:1
        l, d, r = size(A[i])
        F = lq!(reshape(reshape(A[i], l * d, r) * C, l, :))
        L = Matrix(F.L)
        ph = _pivot_phases(L)
        L .*= transpose(conj.(ph))
        nrm = _checked_norm(L)
        C = L ./ nrm
        logc += log(nrm)
        packs[i] = (F, ph, l, d)
    end
    return packs, C, logc
end

function _right_isometries(packs)
    return map(packs) do (F, ph, l, d)
        Q = Matrix(F.Q)
        Q .*= ph
        reshape(Q, l, d, :)
    end
end

# Left-to-right sweep: D B₁⋯Bₙ = A₁⋯Aₙ D' with left-isometric A. Returns the
# factorizations (for `_left_isometries`) and the bond factors (D after each site).
function _qr_sweep(B, D)
    n = length(B)
    packs = Vector{Any}(undef, n)
    centers = Vector{Matrix{eltype(B[1])}}(undef, n)
    for i in 1:n
        l, d, r = size(B[i])
        m = size(D, 1)
        F = qr!(reshape(D * reshape(B[i], l, :), m * d, r))
        R = Matrix(F.R)
        ph = _pivot_phases(R)
        R .*= conj.(ph)
        D = R ./ norm(R)
        centers[i] = D
        packs[i] = (F, ph, m, d, min(m * d, r))
    end
    return packs, centers
end

function _left_isometries(packs)
    return map(packs) do (F, ph, m, d, k)
        Q = Matrix(F.Q)[:, 1:k]
        Q .*= transpose(ph)
        reshape(Q, m, d, :)
    end
end

# Arnoldi acceleration (Algorithm 1, line 9): one unrestarted Krylov pass for the dominant
# eigenvector of a transfer map, started from the current factor. A Krylov pass costs tens of
# transfer applications, and for gapped states the plain sweeps already contract by a factor
# 0.1-0.5 per sweep (measured on PXP states, χ ≤ 128), so the pass is a fallback: it is used
# only after `_ACCEL_AFTER` sweeps whose residual ratio exceeds `_ACCEL_RATIO` (a small
# transfer gap), and not near the roundoff floor, where the ratio only reflects noise.
const _ACCEL_AFTER = 10
const _ACCEL_RATIO = 0.8
const _ACCEL_MIN = 1e-9
function _accelerate(map, X, δ)
    length(X) <= 1 && return X
    vals, vecs, info = try
        eigsolve(map, X, 1, :LM; tol=max(δ / 10, eps()), maxiter=1,
                 krylovdim=min(30, length(X)), ishermitian=false)
    catch err
        err isa Union{ArgumentError,LAPACKException,DomainError} || rethrow()
        return X
    end
    isempty(vecs) && return X
    Y = first(vecs)
    eltype(X) <: Real && (Y = real(Y))
    all(isfinite, Y) ? Y : X
end

# Right-isometric B with A₁⋯Aₙ C = c C B₁⋯Bₙ, C the converged seam factor.
function _right_orthonormalize(A; tol=_ORTH_TOL, maxiter=_ORTH_MAXITER)
    χ = size(A[end], 3)
    C = Matrix{eltype(A[1])}(I, χ, χ) ./ sqrt(χ)
    # X ↦ Σ A X B† over the cell, whose fixed point is C (up to scale)
    packs, Cn, logc = _lq_sweep(A, C)
    δ = norm(Cn - C)
    δprev = Inf
    done, best, since = _orth_progress(δ, Inf, 0, tol)
    iter = 1
    while !done
        iter < maxiter || throw(CanonicalFormError("canonical form: right orthonormalization " *
                                                   "did not converge in $maxiter sweeps (residual $δ)"))
        if iter >= _ACCEL_AFTER && δ > _ACCEL_MIN && δ > _ACCEL_RATIO * δprev
            action = _chain_transfer_action(_right_isometries(packs), A; dir=:r)
            C = first(_pos_lq(_accelerate(action, Cn, δ)))
            C ./= norm(C)
        else
            C = Cn
        end
        δprev = δ
        packs, Cn, logc = _lq_sweep(A, C)
        δ = norm(Cn - C)
        done, best, since = _orth_progress(δ, best, since, tol)
        iter += 1
    end
    return _right_isometries(packs), Cn, logc
end

# Bond factors of a right-isometric chain from left orthonormalization (D B⋯ = A⋯ D).
function _left_factors(B, D0; tol=_ORTH_TOL, maxiter=_ORTH_MAXITER)
    D = D0
    packs, centers = _qr_sweep(B, D)
    δ = norm(centers[end] - D)
    δprev = Inf
    done, best, since = _orth_progress(δ, Inf, 0, tol)
    iter = 1
    while !done
        iter < maxiter || throw(CanonicalFormError("canonical form: left orthonormalization " *
                                                   "did not converge in $maxiter sweeps (residual $δ)"))
        if iter >= _ACCEL_AFTER && δ > _ACCEL_MIN && δ > _ACCEL_RATIO * δprev
            action = _left_chain_transfer_action(_left_isometries(packs), B)
            D = last(_pos_qr(_accelerate(action, centers[end], δ)))
            D ./= norm(D)
        else
            D = centers[end]
        end
        δprev = δ
        packs, centers = _qr_sweep(B, D)
        δ = norm(centers[end] - D)
        done, best, since = _orth_progress(δ, best, since, tol)
        iter += 1
    end
    centers[end] = D
    return centers
end

# The right sweep determines the cell norm before the left Schmidt factors are needed.
function _right_canonical_with_norm(A; reduce::Bool=true)
    A, lognull = reduce ? _remove_null_directions(A) : (A, 0.0)
    B, _, logc = _right_orthonormalize(A)
    return B, logc + lognull
end

# Schmidt-canonical form of the cell `A` in the stored convention B = Γλ.
# Returns (tensors, spectra, log of the norm per cell).
function _mixed_canonical(A::Vector{<:AbstractArray{<:Number,3}}; reduce::Bool=true)
    n = length(A)
    B, logc = _right_canonical_with_norm(A; reduce)
    k = size(B[1], 1)
    D0 = Matrix{eltype(B[1])}(I, k, k)
    D0 ./= norm(D0)
    centers = _left_factors(B, D0)
    Vs = Vector{Matrix{eltype(B[1])}}(undef, n)
    spectra = Vector{Vector{real(eltype(B[1]))}}(undef, n)
    for i in 1:n
        F = svd(centers[i])
        Vs[i] = F.V
        spectra[i] = F.S ./ norm(F.S)
    end
    out = similar(B)
    for i in 1:n
        Vl = Vs[mod1(i - 1, n)]
        l, d, r = size(B[i])
        X = reshape(Vl' * reshape(B[i], l, :), :, r) * Vs[i]
        out[i] = reshape(X, size(Vl, 2), d, size(Vs[i], 2))
    end
    return out, spectra, logc
end

function _precise_canonical!(ψ::DenseIMPS; renormalize::Bool=true, reduce::Bool=true)
    T = eltype(ψ)
    tensors, spectra, lognorm = _mixed_canonical(ψ.Γ; reduce)
    for i in 1:ψ.n
        ψ.Γ[i] = Array{T,3}(tensors[i])
        ψ.λ[i] = convert(typeof(ψ.λ[i]), spectra[i])
    end
    renormalize || (ψ.Γ[1] .*= exp(lognorm))
    return lognorm
end

# Schmidt values below this (relative to the largest, after normalization) are numerical
# null directions and are always dropped.
const _NULL_FLOOR = 1e-14

# Exact canonical form, then every bond keeps its own largest Schmidt values (at most
# `maxdim`, discarded weight ≤ `truncerr`, none below `cutoff`), as Orús and Vidal truncate
# all bonds at once. With `recanonicalize=true` the gauge is restored afterwards; otherwise
# the sliced state is only renormalized (the next canonicalization repairs the gauge, as in
# their scheme). Returns (per-bond BondStats, log norm per cell of the input, whether any
# bond was sliced).
function _canonical_truncate!(ψ::DenseIMPS; maxdim, cutoff, truncerr, renormalize,
                              recanonicalize::Bool=true)
    lognorm = _precise_canonical!(ψ; renormalize)
    stats, sliced = _truncate_canonical!(ψ; maxdim, cutoff, truncerr, renormalize, recanonicalize)
    return stats, lognorm, sliced
end

# Per-bond truncation of a state already in exact canonical form (shared by the generic
# and the structured MPO updates, so both truncate identically).
function _truncate_canonical!(ψ::DenseIMPS; maxdim, cutoff, truncerr, renormalize,
                              recanonicalize::Bool=true)
    stats = Vector{BondStat}(undef, ψ.n)
    keep = Vector{Int}(undef, ψ.n)
    floor = max(cutoff, _NULL_FLOOR)
    for b in 1:ψ.n
        c = _discarded_weight_choice(ψ.λ[b]; mindim=1, maxdim, truncerr, svd_min=floor)
        keep[b] = c.chi_keep
        # Requested dimension and saturation count only Schmidt values at or above the floor:
        # values the cutoff removes anyway do not make `maxdim` binding. (Their number depends
        # on how null directions were removed, so counting them would make the statistic
        # representation-dependent.)
        chi_req = min(c.chi_req, max(1, count(>=(floor), ψ.λ[b])))
        stats[b] = BondStat(b, chi_req, c.chi_keep, c.discarded_weight, chi_req > maxdim,
                            c.target_met, c.smallest_kept_sv, c.largest_discarded_sv)
    end
    sliced = any(b -> keep[b] < length(ψ.λ[b]), 1:ψ.n)
    if sliced
        # In canonical form the bond basis is the Schmidt basis, so keeping the leading k_b
        # values is a slice of the two tensors next to bond b.
        for b in 1:ψ.n
            nb = mod1(b + 1, ψ.n)
            k = keep[b]
            ψ.λ[b] = ψ.λ[b][1:k]
            ψ.Γ[b] = ψ.Γ[b][:, :, 1:k]
            ψ.Γ[nb] = ψ.Γ[nb][1:k, :, :]
        end
        if recanonicalize
            # Restore the gauge on the retained bond support.
            _precise_canonical!(ψ; renormalize, reduce=false)
        elseif renormalize
            for b in 1:ψ.n
                ψ.λ[b] ./= norm(ψ.λ[b])
            end
        end
    end
    return stats, sliced
end

function _truncated_state(ψ::DenseIMPS; maxdim, cutoff, truncerr, renormalize,
                          canonical_input::Bool=false)
    # Canonicalization and slicing replace the tensors and spectra before mutating them.
    trial = iMPS(copy(ψ.Γ), copy(ψ.λ), ψ.n)
    if canonical_input
        stats, sliced = _truncate_canonical!(trial; maxdim, cutoff, truncerr, renormalize)
        return trial, stats, 0.0, sliced
    end
    stats, lognorm, sliced = _canonical_truncate!(trial; maxdim, cutoff, truncerr, renormalize)
    return trial, stats, lognorm, sliced
end

function _truncation_summary(stats::AbstractVector{BondStat}; path::Symbol=:generic)
    return (
        path=path,
        bond_stats=stats,
        max_discarded_weight=isempty(stats) ? 0.0 : maximum(s.discarded_weight for s in stats),
        kept_dims=[s.chi_keep for s in stats],
        num_saturated=count(s -> s.saturated, stats),
    )
end

"""
    truncate!(ψ::iMPS, χ; cutoff=SVDTOL, truncerr=0.0, renormalize=true, return_stats=false)

Truncate every bond of `ψ` to at most `χ` Schmidt values, treating all bonds
alike.

Procedure: bring `ψ` to exact canonical form; then each bond keeps its own
`χ` largest Schmidt values (fewer if `truncerr > 0` allows it, or if values
fall below `cutoff`); the canonical gauge is then restored. Because no bond is
singled out, a state that is invariant under a translation of the cell stays
invariant (up to roundoff) — tiling a two-site state onto four sites and
truncating gives the tiled truncated two-site state.

The state is updated only after truncation and canonicalization succeed.

Returns `ψ`, or `(ψ, stats)` with `return_stats=true`; `stats` has fields
`bond_stats`, `max_discarded_weight`, `kept_dims` and `num_saturated`.

Example:
```julia
psi = rand_iMPS(ComplexF64, 2, 2, 8)
truncate!(psi, 2)
length.(psi.λ)      # [2, 2]
```
"""
function truncate!(ψ::iMPS, χ::Integer; cutoff::Real=SVDTOL, truncerr::Real=0.0,
                   renormalize::Bool=true, return_stats::Bool=false)
    ψ isa DenseIMPS || throw(ArgumentError("truncate! supports dense iMPS states"))
    _validate_mpo_truncation(χ, cutoff, truncerr)
    trial, stats, _ = _truncated_state(ψ; maxdim=χ, cutoff, truncerr, renormalize)
    ψ.Γ .= trial.Γ
    ψ.λ .= trial.λ
    return return_stats ? (ψ, _truncation_summary(stats)) : ψ
end

#---------------------------------------------------------------------------------------------------
# MPO application
#---------------------------------------------------------------------------------------------------
function _check_mpo_state(ψ::iMPS, W::iMPO)
    ψ isa DenseIMPS || throw(ArgumentError("apply! supports dense iMPS states"))
    ψ.n % W.n == 0 || throw(ArgumentError(
        "the iMPS cell length $(ψ.n) must be a multiple of the iMPO cell length $(W.n)"))
    for i in 1:ψ.n
        size(W[i], 3) == size(ψ.Γ[i], 2) || throw(DimensionMismatch(
            "site $i: iMPO input dimension $(size(W[i], 3)) but iMPS physical dimension $(size(ψ.Γ[i], 2))"))
    end
    if eltype(ψ) <: Real && eltype(W) <: Complex
        throw(ArgumentError(
            "cannot apply a complex iMPO in place to an iMPS with real element type $(eltype(ψ)); " *
            "convert the state first with mps_promote_type(ComplexF64, ψ)"))
    end
    return nothing
end

"""
    apply!(ψ, W::iMPO; maxdim=MAXDIM, cutoff=SVDTOL, truncerr=0.0, renormalize=true, recanonicalize=true, return_stats=false, structured=true)

Apply the infinite MPO `W` to `ψ` in place and truncate.

The cell length of `ψ` must be a multiple of `W.n`. Each stored tensor
`B_i = Γ_i λ_i` is contracted with `W[i]` (the bond dimensions multiply),
after which the state is brought to exact canonical form and every bond keeps
its own largest Schmidt values, as in [`truncate!`](@ref): at most `maxdim`,
with discarded weight at most `truncerr` when `truncerr > 0`, and none below
`cutoff`.

`W` need not be unitary: imaginary-time layers and projectors work. With
`renormalize=true` (the default) the result is normalized.

Failure handling: the update is computed on new arrays and committed only on
success, so if `apply!` throws, `ψ` is unchanged. An MPO that annihilates the
state throws [`MPOAnnihilationError`](@ref): `W·ψ` is treated as zero when its
norm per cell, relative to the input's, is at the roundoff level of the
contraction (`≈ 10⁴ ε ∏ᵢ ‖Wᵢ‖`).

This is the procedure of Orús and Vidal (PRB 78, 155117, Sec. III): contract,
orthonormalize, truncate all bonds at once. The orthonormalization is the
single-layer QR iteration of Vanderstraeten, Haegeman and Verstraete (SciPost
Phys. Lect. Notes 7, Sec. 2.3), which resolves Schmidt values to machine
precision rather than its square root. Exact null directions of the enlarged
bond are removed. With `recanonicalize=false` a truncated state is not
regauged: the tensors are not in canonical form and the stored `λ` are only
the sliced spectra, so Schmidt-gauge observables (`expect`, `ent_S`) are not
reliable until the next `apply!`, `truncate!` or `canonical!`. With
`renormalize=true` the state is still normalized per cell (the transfer-matrix
norm is divided out). [`evolve!`](@ref) skips that normalization in all but
its last application, since the next application canonicalizes anyway.

Structured update: if `W` carries a controlled-center structure (see
[`layer_mpo`](@ref)), `structured=true`, the state is a dense complex two-site
cell and `renormalize=true`, the update is computed sector by sector over the
control occupations instead of from the enlarged MPO×MPS. It is the same
state, truncated by the same code; it falls back to the generic update if its
iteration does not converge or anything fails numerically. `stats.path`
reports which update ran (`:controlled` or `:generic`).

Returns `ψ`, or `(ψ, stats)` with `return_stats=true`. `stats` matches the
style of [`evolve!`](@ref): `path`, `bond_stats` (one `BondStat` per bond),
`max_discarded_weight` (fraction of the squared Schmidt weight discarded,
maximized over bonds), `kept_dims` and `num_saturated` (bonds where `maxdim`
removed Schmidt values at or above the truncation floor `max(cutoff, 1e-14)`;
values the cutoff removes anyway are not counted).

Example:
```julia
X = [0 1; 1 0]; P = [1 0; 0 0]
W = layer_mpo([(exp(-0.1im * kron(P, X, P)), 2, 4)], 2)
psi = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
apply!(psi, W; maxdim=16)
```
"""
function apply!(ψ::iMPS, W::iMPO; maxdim::Integer=MAXDIM, cutoff::Real=SVDTOL,
                truncerr::Real=0.0, renormalize::Bool=true, recanonicalize::Bool=true,
                return_stats::Bool=false, structured::Bool=true)
    stats, path = _apply_mpo!(ψ, W; maxdim, cutoff, truncerr, renormalize, recanonicalize,
                              normalize_cells=true, structured)
    return return_stats ? (ψ, _truncation_summary(stats; path)) : ψ
end

# W·ψ as a new iMPS (fused bond index: state index fastest). `ψ` is not modified.
function _contract_mpo(ψ::DenseIMPS, W::iMPO)
    T = eltype(ψ)
    Γ = Vector{Array{T,3}}(undef, ψ.n)
    λ = similar(ψ.λ)
    for i in 1:ψ.n
        B = ψ.Γ[i]
        A = W[i]
        @tensor C[a, l, o, b, r] := A[l, o, s, r] * B[a, s, b]
        s = size(C)
        Γ[i] = Array{T,3}(reshape(C, s[1] * s[2], s[3], s[4] * s[5]))
        # Placeholder spectra with the fused bond layout (only their sizes are used).
        λ[i] = kron(ones(eltype(ψ.λ[i]), size(A, 4)), ψ.λ[i])
    end
    return iMPS(Γ, λ, ψ.n)
end

# W·ψ counts as annihilated when its norm per cell, relative to ψ's, is at the roundoff
# level of the contraction, ~ε·∏ᵢ‖Wᵢ‖ (with a safety factor). A projector can leave
# tensors of order one whose cell transfer matrix is nilpotent; the canonical form of such
# a cell is built from roundoff, so the norm, not the tensors, has to decide.
const _ANNIHILATION_MARGIN = 1e4

function _annihilation_threshold(ψ, W::iMPO)
    return log(_ANNIHILATION_MARGIN * eps(Float64)) +
           sum(log(max(norm(W[i]), floatmin(Float64))) for i in 1:ψ.n)
end

# `lognorm`: log norm per cell of W·ψ (unnormalized input scale). True if annihilated.
#
# The bound ε·∏‖Wᵢ‖ grows with the cell length (for the identity, (√d)ⁿ), so it only
# screens. A positive screen is confirmed against the roundoff scale of the contraction
# itself: the norm per cell of |W|·|ψ|, the same contraction with entrywise absolute
# values (no cancellation). W·ψ is annihilated only if its norm is below ~ε times that.
# For operators without cancellation (identity, 0/1 projectors) the two norms agree and a
# small finite weight is never mistaken for zero, for any cell length.
function _annihilated(lognorm, ψ, W::iMPO)
    threshold = _annihilation_threshold(ψ, W)
    lognorm >= threshold + 40 && return false           # far above: skip the norm of ψ
    # Sufficient bound without an eigensolve: ½ log ρ(E_self) ≤ Σᵢ log‖Bᵢ‖_F (each site's
    # Kraus map has norm ≤ ‖Bᵢ‖², and the spectral radius is submultiplicative). If the
    # norm of W·ψ relative to this upper bound already clears the screen, so does the
    # relative norm itself, for any gauge.
    lognorm - sum(log(norm(A)) for A in ψ.Γ) >= threshold && return false
    lognorm - log(inner_product(ψ)) / 2 < threshold || return false
    return lognorm < log(_ANNIHILATION_MARGIN * eps(Float64)) + _log_abs_contraction(ψ, W)
end

# log norm per cell of the contraction of |W| with |ψ| (entrywise absolute values).
function _log_abs_contraction(ψ::DenseIMPS, W::iMPO)
    maps = map(1:ψ.n) do i
        A = abs.(W[i])
        B = abs.(ψ.Γ[i])
        @tensor C[a, l, o, b, r] := A[l, o, s, r] * B[a, s, b]
        sz = size(C)
        C3 = reshape(C, sz[1] * sz[2], sz[3], sz[4] * sz[5])
        blocks = [Matrix(view(C3, :, o, :)) for o in 1:sz[3]]
        v -> begin
            k = size(blocks[1], 1)
            L = reshape(v, k, k)
            vec(sum(Bo' * L * Bo for Bo in blocks))
        end
    end
    k = size(ψ.Γ[1], 1) * size(W[1], 1)
    logρ = _log_positive_radius(maps, vec(Matrix{Float64}(I, k, k)))
    return logρ / 2
end

# Transactional MPO application. `normalize_cells=false` skips the per-cell normalization of
# an unregauged truncated state (used inside `evolve!`, where the next application
# canonicalizes anyway).
function _apply_mpo!(ψ::iMPS, W::iMPO; maxdim, cutoff, truncerr, renormalize, recanonicalize,
                     normalize_cells::Bool, structured::Bool=true)
    _validate_mpo_truncation(maxdim, cutoff, truncerr)
    _check_mpo_state(ψ, W)
    res = structured && _controlled_eligible(ψ, W, renormalize) ?
        _apply_controlled(ψ, _valid_structure(W); maxdim, cutoff, truncerr, renormalize, recanonicalize) :
        nothing
    if isnothing(res)
        path = :generic
        trial = _contract_mpo(ψ, W)
        stats, lognorm, sliced = _canonical_truncate!(trial; maxdim, cutoff, truncerr, renormalize,
                                                      recanonicalize)
    else
        path = :controlled
        trial, stats, lognorm, sliced = res
    end
    _annihilated(lognorm, ψ, W) && throw(MPOAnnihilationError(
        "the MPO maps the state to zero (norm per cell at the roundoff level)"))
    if sliced && !recanonicalize && renormalize && normalize_cells
        η = inner_product(trial)
        isfinite(η) && η > 0 && (trial.Γ[1] ./= sqrt(η))
    end
    for i in 1:ψ.n
        ψ.Γ[i] = trial.Γ[i]
        ψ.λ[i] = trial.λ[i]
    end
    return stats, path
end

#---------------------------------------------------------------------------------------------------
# Trotterized MPO evolution
#---------------------------------------------------------------------------------------------------
_layer_exp(h::Diagonal, scale) = Diagonal(exp.(scale .* h.diag))
_layer_exp(h::AbstractMatrix, scale) = exp(scale * Matrix(h))

"""
    trotter_mpos(layers, dt; trotter=:second, evolution=:real, n, cutoff=1e-14, check=true, gate_map=identity)

The sequence of layer MPOs for one Trotter step of length `dt`.

`layers` has the same shape as for [`trotter_gates`](@ref): each layer is a
list of local terms `(h, i, j)` whose overlapping terms commute. The supports
follow the convention of [`layer_mpo`](@ref) (`i ≤ j` may span more than the
cell `n`). The stage schedule and time prefactors are those of
`trotter_gates` (`_trotter_stage_schedule`, `_trotter_time_prefactor`), so
`:second` gives `[U_1(dt/2), U_2(dt/2), …, U_m(dt), …, U_1(dt/2)]` and
`:fourth`/`:fourth_opt` their real-time compositions. Each stage becomes
`layer_mpo([(exp(c·h), i, j) …], n)` with `c = -i·coeff·dt` for
`evolution=:real` and `c = -coeff·dt` for `evolution=:imaginary`.

With `check=true`, each layer's terms are checked with
[`check_commuting`](@ref) once (commuting terms give commuting gates).

`gate_map` is applied to every exponentiated gate before it is split into an
MPO, e.g. `gate_map = G -> P3 * G` folds a blockade projector `P3` into each
gate. The mapped gates must still commute where they overlap (with
`check=true` they are checked again).

Returns a `Vector{iMPO}`; apply it with
`evolve!(ψ, mpos, steps)`, which merges the half steps at step boundaries.

Example:
```julia
X = [0 1; 1 0]; P = [1 0; 0 0]; nn = [0 0; 0 1]; I2 = [1 0; 0 1]
h(c) = kron(P, X, P) + c * kron(I2, nn, I2)
layers = [[(h(1.2), 2, 4)], [(h(0.6), 1, 3)]]   # A = site 1, B = site 2
U = trotter_mpos(layers, 0.05; n=2)             # [U_A(dt/2), U_B(dt), U_A(dt/2)]
```
"""
function trotter_mpos(layers, dt::Real; trotter::Symbol=:second, evolution::Symbol=:real,
                      n::Integer, cutoff::Real=MPO_CUTOFF, check::Bool=true,
                      d::Union{Nothing,Integer}=nothing, gate_map=identity)
    isfinite(dt) || throw(ArgumentError("dt must be finite"))
    _validate_trotter_scheme(length(layers), trotter)
    _validate_trotter_evolution(trotter, evolution)
    if check
        for layer in layers
            check_commuting(layer; n, d)
        end
    end
    stages = _trotter_stage_schedule(length(layers), trotter, 1)
    cache = Dict{Tuple{Int,Float64},iMPO}()
    mpos = iMPO[]
    for (k, coeff) in stages
        W = get!(cache, (k, Float64(coeff))) do
            scale = _trotter_time_prefactor(dt, coeff, evolution)
            gates = [(gate_map(_layer_exp(h, scale)), i, j) for (h, i, j) in layers[k]]
            layer_mpo(gates, n; cutoff, check=check && gate_map !== identity, d)
        end
        push!(mpos, W)
    end
    T = mapreduce(eltype, promote_type, mpos)
    return [W isa iMPO{T} ? W : iMPO([Array{T,4}(A) for A in W.W], W.n, _valid_structure(W)) for W in mpos]
end

# Accept a merged boundary MPO only if it is no larger than the parts. In time order
# `last_mpo` (end of one step) acts before `first_mpo` (start of the next), so the merged
# operator is first_mpo · last_mpo.
function _merged_boundary(last_mpo::iMPO, first_mpo::iMPO)
    last_mpo.n == first_mpo.n || return nothing
    merged = mpo_multiply(first_mpo, last_mpo; compress=true)
    bound = max.(bonddims(last_mpo), bonddims(first_mpo))
    merged.n == last_mpo.n && all(bonddims(merged) .<= bound) || return nothing
    structure = _merged_structure(_valid_structure(first_mpo), _valid_structure(last_mpo))
    return isnothing(structure) ? merged : iMPO(merged.W, merged.n, structure)
end

"""
    evolve!(ψ, mpos::Vector{<:iMPO}, steps; maxdim=MAXDIM, cutoff=SVDTOL, truncerr=0.0, renormalize=true, merge=true, return_stats=false, structured=true, observe=nothing, every=1)

Apply the MPO sequence `mpos` (one Trotter step, e.g. from
[`trotter_mpos`](@ref)) `steps` times with [`apply!`](@ref), truncating after
every MPO. Intermediate truncated states are not regauged (the next
application does that); the returned state is canonical and normalized. Each
application is transactional, so if one throws (e.g.
[`MPOAnnihilationError`](@ref)), `ψ` holds the state after the last
successful application.

Half-step merging: with `merge=true` and `steps > 1`, the last MPO of a step
and the first MPO of the next step (for `:second`, `U_A(dt/2) U_A(dt/2)`) are
multiplied and compressed once with [`compress_mpo`](@ref). The merged MPO is
used at every interior step boundary if its bond dimensions do not exceed
those of the two factors (exact for commuting layer MPOs, giving `U_A(dt)`);
otherwise the two are applied separately. Merging removes one MPO
application and one truncation per step. Because of it, `steps` calls of
`evolve!(ψ, mpos, 1)` are **not** the same integrator as one call with
`steps`: use `observe` to sample a trajectory.

Observation: `observe(φ, k, stats)` is called with `k = 0` (the input state as
given, `stats = nothing`), after every `every` complete Trotter steps, and after
the last step (also when `steps` is not a multiple of `every`). `φ` is the
canonical, normalized state at time `k·dt`: exactly the state that
`evolve!(copy, mpos, k)` would return from the same input, also at a merged
boundary, where the pending half step is applied to a **snapshot** with the
final-application settings; the main trajectory, its truncations and the
returned `ψ` are the same as without `observe`. For `0 < k < steps`, `φ` is a
private snapshot the observer may keep; at `k = 0` and `k = steps` it is `ψ`
itself — copy it (`deepcopy`) to keep it. The observer must not modify `φ`.
`stats` aggregates the main-line applications since the previous sample plus,
for an intermediate sample, the application that completed the snapshot; its
fields are those of `return_stats` plus `num_at_cap`. Observation is off by
default and does not change the return value.

Returns `ψ`, or `(ψ, stats)` with `return_stats=true`: `updates` (the stats
of each [`apply!`](@ref)), `max_discarded_weight`, `mean_discarded_weight`,
`max_kept_dim`, `num_saturated` (bonds where the requested dimension exceeded
`maxdim`), `num_at_cap` (bonds whose retained dimension equals `maxdim`), and
`num_structured`, the number of applications that used the structured update.
`structured=true` allows it (see [`apply!`](@ref)).

Example:
```julia
X = [0 1; 1 0]; P = [1 0; 0 0]
layers = [[(kron(P, X, P), 2, 4)], [(kron(P, X, P), 1, 3)]]
U = trotter_mpos(layers, 0.05; n=2)
psi = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
evolve!(psi, U, 20; maxdim=16)
```
"""
function evolve!(ψ::iMPS, mpos::AbstractVector{<:iMPO}, steps::Integer;
                 maxdim::Integer=MAXDIM, cutoff::Real=SVDTOL, truncerr::Real=0.0,
                 renormalize::Bool=true, merge::Bool=true, return_stats::Bool=false,
                 structured::Bool=true, observe=nothing, every::Integer=1)
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    _validate_mpo_truncation(maxdim, cutoff, truncerr)
    every >= 1 || throw(ArgumentError("every must be positive"))
    foreach(W -> _check_mpo_state(ψ, W), mpos)
    k = length(mpos)
    merged = merge && k >= 2 && steps >= 2 ? _merged_boundary(mpos[end], mpos[1]) : nothing
    return _evolve_sequence!(ψ, mpos, merged, steps; maxdim, cutoff, truncerr, renormalize,
                             return_stats, structured, observe, every)
end

# `merged`: precomputed boundary MPO (or nothing for no merging); see `evolve!`.
function _evolve_sequence!(ψ, mpos, merged, steps; maxdim, cutoff, truncerr, renormalize,
                           return_stats, structured, observe=nothing, every=1)
    isnothing(observe) || observe(ψ, 0, nothing)
    if isempty(mpos)
        return return_stats ? (ψ, _aggregate_mpo_stats(Any[]; maxdim)) : ψ
    end
    k = length(mpos)
    record = return_stats || !isnothing(observe)
    updates = record ? Any[] : nothing
    since = 0                                    # first update index of the current interval
    apply_one!(state, W, last) = begin
        # Intermediate truncations are only renormalized: the next application
        # recanonicalizes anyway. The final state is returned in canonical form.
        st, path = _apply_mpo!(state, W; maxdim, cutoff, truncerr, renormalize, recanonicalize=last,
                               normalize_cells=false, structured)
        return record ? _truncation_summary(st; path) : nothing
    end
    for s in 1:steps
        body = isnothing(merged) || s == 1 ? view(mpos, 1:k-1) :
               Iterators.flatten(((merged,), view(mpos, 2:k-1)))
        for W in body
            u = apply_one!(ψ, W, false)
            record && push!(updates, u)
        end
        last_step = s == steps
        due = !isnothing(observe) && (s % every == 0 || last_step)
        if due && !last_step
            # Snapshot at time s·dt: the pending last layer, applied as a final application.
            snap = iMPS(copy(ψ.Γ), copy(ψ.λ), ψ.n)
            u = apply_one!(snap, mpos[end], true)
            observe(snap, s, _aggregate_mpo_stats(vcat(updates[since+1:end], [u]); maxdim))
            since = length(updates)
        end
        if isnothing(merged) || last_step
            u = apply_one!(ψ, mpos[end], last_step)
            record && push!(updates, u)
        end
        if due && last_step
            observe(ψ, s, _aggregate_mpo_stats(updates[since+1:end]; maxdim))
            since = length(updates)
        end
    end
    return return_stats ? (ψ, _aggregate_mpo_stats(updates; maxdim)) : ψ
end

function _aggregate_mpo_stats(updates; maxdim=nothing)
    all_stats = BondStat[]
    for u in updates
        append!(all_stats, u.bond_stats)
    end
    w = [s.discarded_weight for s in all_stats]
    return (
        updates=updates,
        max_discarded_weight=isempty(w) ? 0.0 : maximum(w),
        mean_discarded_weight=isempty(w) ? 0.0 : sum(w) / length(w),
        max_kept_dim=isempty(all_stats) ? 0 : maximum(s.chi_keep for s in all_stats),
        num_saturated=count(s -> s.saturated, all_stats),
        num_at_cap=isnothing(maxdim) ? 0 : count(s -> s.chi_keep >= maxdim, all_stats),
        num_structured=count(u -> u.path === :controlled, updates),
    )
end

#---------------------------------------------------------------------------------------------------
# Site-dependent energies and energy fixing
#---------------------------------------------------------------------------------------------------
"""
    energy_density(ψ, terms; n=ψ.n)

Energy per site of the site-dependent local terms `terms`, a list of
`(h, i, j)` that is repeated with period `n` (the support convention of
[`layer_mpo`](@ref): `i ≤ j` may span more than the cell).

`ψ.n` must be a multiple of `n`. The result is
`(1 / ψ.n) · ∑_{translates within ψ's cell} ∑_terms ⟨h⟩`; for a two-site PXP
cell with `terms = [(h_A, 2, 4), (h_B, 1, 3)]` this is `(⟨h_A⟩ + ⟨h_B⟩) / 2`.
`ψ` must be in canonical form (as returned by `canonical!`, `apply!`,
`truncate!`, …).

This complements `energy_density(ψ, h)`, which averages one density `h` over
all sites.
"""
function energy_density(ψ::iMPS, terms::AbstractVector; n::Integer=ψ.n)
    return _mean_local_expectation(ψ, _energy_observables(ψ, terms, n))
end

function _energy_observables(ψ, terms, n)
    ψ.n % n == 0 || throw(ArgumentError("ψ.n = $(ψ.n) must be a multiple of the term period n = $n"))
    return Iterators.flatten(_term_observables(ψ, h, i, j, n) for (h, i, j) in terms)
end

function _term_observables(ψ, h, i, j, n)
    a, b, _, _ = _support_span(h, i, j, n, size(ψ.Γ[1], 2))
    return Iterators.map(0:n:(ψ.n - 1)) do shift
        inds = [mod1(k + shift, ψ.n) for k in a:b]
        O = _operator_to_internal(h, Int[size(ψ.Γ[k], 2) for k in inds])
        return (O, inds)
    end
end

# A list of layers or a flat list of terms → (layers, terms).
function _layers_and_terms(x, n, d)
    items = collect(x)
    isempty(items) && throw(ArgumentError("no Hamiltonian terms given"))
    if first(items) isa Tuple
        terms = items
        return _auto_layers(terms, n, d), terms
    end
    layers = [collect(l) for l in items]
    return layers, reduce(vcat, layers)
end

# Greedy split of terms into layers that satisfy the layer_mpo precondition.
function _auto_layers(terms, n, d)
    layers = Vector{Any}[]
    for t in terms
        placed = false
        for layer in layers
            candidate = vcat(layer, [t])
            ok = try
                _bond_crossings_ok(candidate, n, d) && check_commuting(candidate; n, d)
            catch err
                err isa ArgumentError || rethrow()
                false
            end
            if ok
                push!(layer, t)
                placed = true
                break
            end
        end
        placed || push!(layers, Any[t])
    end
    return layers
end

function _bond_crossings_ok(terms, n, d)
    crossed = falses(n)
    for (G, i, j) in terms
        a, b, L, _ = _support_span(G, i, j, n, d)
        for p in 1:L-1
            c = mod1(a + p - 1, n)
            crossed[c] && return false
            crossed[c] = true
        end
    end
    return true
end

# Validate the layers and compute spectral bounds before modifying the state.
struct _ImagPrep
    layers::Vector{Vector{Any}}
    terms::Vector{Any}
    n::Int
    d::Int
    spectra::IdDict{Any,Tuple{Float64,Float64}}
end

function _prepare_imaginary(terms_or_layers; n::Integer, d::Integer)
    layers, terms = _layers_and_terms(terms_or_layers, n, d)
    spectra = IdDict{Any,Tuple{Float64,Float64}}()
    for (h, i, j) in terms
        _support_span(h, i, j, n, d)
        ishermitian(h) || throw(ArgumentError("energy_fix! requires Hermitian terms"))
        get!(spectra, h) do
            v = eigvals(Hermitian(Matrix(h)))
            (Float64(first(v)), Float64(last(v)))
        end
    end
    for layer in layers
        _bond_crossings_ok(layer, n, d) || throw(ArgumentError(
            "a layer has two gates (or two translates of one gate) crossing the same bond of " *
            "the $n-site cell; split it into separate layers"))
        check_commuting(layer; n, d)
    end
    return _ImagPrep(layers, terms, Int(n), Int(d), spectra)
end

function _validate_energy_fix_args(ψ, χ, prep::_ImagPrep; target, tol, alpha, maxstep, cutoff)
    isfinite(target) || throw(ArgumentError("target must be finite"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    isfinite(alpha) && alpha >= 0 || throw(ArgumentError("alpha must be finite and nonnegative"))
    maxstep >= 0 || throw(ArgumentError("maxstep must be nonnegative"))
    _validate_mpo_truncation(χ, cutoff, 0.0)
    ψ.n % prep.n == 0 || throw(ArgumentError("ψ.n = $(ψ.n) must be a multiple of the term period n = $(prep.n)"))
    all(A -> size(A, 2) == prep.d, ψ.Γ) || throw(DimensionMismatch(
        "the terms act on local dimension $(prep.d), the state has $(size.(ψ.Γ, 2))"))
    return nothing
end

function _imaginary_trial_mpos(prep::_ImagPrep, dτ)
    shifted = [[(h - (dτ > 0 ? prep.spectra[h][1] : prep.spectra[h][2]) * I, i, j) for (h, i, j) in layer]
               for layer in prep.layers]
    return trotter_mpos(shifted, dτ; evolution=:imaginary, n=prep.n, check=false, d=prep.d)
end

function _energy_fix_core!(ψ, prep::_ImagPrep, χ; target, tol, alpha, maxstep, cutoff, structured=true)
    observables = collect(_energy_observables(ψ, prep.terms, prep.n))
    measure(state) = _mean_local_expectation(state, observables)
    residual = measure(ψ) - target
    steps = 0
    if abs(residual) <= tol || maxstep == 0 || iszero(alpha)
        return (converged=abs(residual) <= tol, residual, steps)
    end
    eltype(ψ) <: Real && throw(ArgumentError("energy_fix! needs a complex iMPS; use mps_promote_type"))
    trial = deepcopy(ψ)
    dτ = alpha * residual
    while steps < maxstep && isfinite(dτ) && !iszero(dτ)
        Ws = _imaginary_trial_mpos(prep, dτ)
        _evolve_sequence!(trial, Ws, nothing, 1; maxdim=χ, cutoff, truncerr=0.0,
                          renormalize=true, return_stats=false, structured)
        attempted = measure(trial) - target
        steps += 1
        if abs(attempted) < abs(residual)
            _copy_imps_state!(ψ, trial)
            abs(attempted) <= tol && return (converged=true, residual=attempted, steps)
            dτ *= clamp(attempted / (residual - attempted), -2, 2)
            residual = attempted
        else
            _copy_imps_state!(trial, ψ)          # restart from the last accepted state
            dτ /= 2
        end
    end
    return (converged=false, residual, steps)
end

"""
    energy_fix!(ψ, terms_or_layers, χ; target, tol=1e-6, alpha=0.1, maxstep=50, n=ψ.n, cutoff=SVDTOL, structured=true)

Move the energy per site of `ψ` toward `target` with short imaginary-time
steps built as layer MPOs from site-dependent terms.

`terms_or_layers` is either a list of commuting layers (as for
[`trotter_mpos`](@ref)) or a flat list of terms `(h, i, j)`, which is split
greedily into layers that satisfy the [`layer_mpo`](@ref) precondition. The
terms are repeated with period `n`, and the energy is measured with
`energy_density(ψ, terms; n)`.

Start with `dτ = alpha · (E − target)`; each trial applies a second-order imaginary-time
step `trotter_mpos(layers, dτ; evolution=:imaginary)` truncated to `χ`.
A trial that reduces `|E − target|` is accepted and the next `dτ` is predicted
from the measured slope (growth clamped to a factor 2). On rejection, the
trial restarts from the last accepted state and `dτ` is halved. `ψ` always holds
the last accepted state. Each term is shifted by a constant before
exponentiation, which changes nothing after normalization but keeps the gates
bounded.

Returns `(converged, residual, steps)`. The residual is `E − target` for
the state left in `ψ`, and `steps` counts the trials.
"""
function energy_fix!(ψ::iMPS, terms_or_layers, χ::Integer; target::Real, tol::Real=1e-6,
                     alpha::Real=0.1, maxstep::Integer=50, n::Integer=ψ.n,
                     cutoff::Real=SVDTOL, d::Union{Nothing,Integer}=nothing,
                     structured::Bool=true)
    dd = isnothing(d) ? size(ψ.Γ[1], 2) : d
    n > 0 || throw(ArgumentError("unit-cell length must be positive"))
    prep = _prepare_imaginary(terms_or_layers; n, d=dd)
    _validate_energy_fix_args(ψ, χ, prep; target, tol, alpha, maxstep, cutoff)
    return _energy_fix_core!(ψ, prep, χ; target, tol, alpha, maxstep, cutoff, structured)
end

#---------------------------------------------------------------------------------------------------
# Per-site fidelity
#---------------------------------------------------------------------------------------------------
"""
    per_site_fidelity(ψ1, ψ2)

`|⟨ψ1|ψ2⟩|^(1/n)` per site for two normalized iMPS with the same cell length
`n`, from the dominant eigenvalue of the mixed cell transfer matrix.
"""
function per_site_fidelity(ψ1::iMPS, ψ2::iMPS)
    ψ1.n == ψ2.n || throw(ArgumentError("per_site_fidelity needs equal cell lengths"))
    return inner_product(ψ1, ψ2; tol=1e-14)^(1 / ψ1.n)
end
export per_site_fidelity

"""
    constraint_leakage(ψ, P::iMPO)

Per-site leakage of `ψ` out of the subspace of the projector MPO `P`:
`−log F_P` with `F_P = |⟨ψ|φ⟩|^(1/n)` the per-site fidelity between `ψ` and its
normalized projection `φ = P|ψ⟩/‖P|ψ⟩‖`. For a projector this is
`−log(⟨ψ|P|ψ⟩) / (2n)` per cell of `n` sites, the same normalization as the
ScarFinder `loss`, so the two can be compared directly. Returns `Inf` if `P`
annihilates `ψ`. `ψ` is not modified.

`P` must be a projector (e.g. the blockade `∏(1 − n_j n_{j+1})` built with
[`layer_mpo`](@ref)); the formula is not meaningful for other operators. The
projection is computed without truncation. For the nearest-neighbour
blockade prefer `constraint_leakage(ψ, Blockade())`, which needs no projection and
keeps small weights accurate; see [`Blockade`](@ref).
"""
function constraint_leakage(ψ::iMPS, P::iMPO)
    _check_mpo_state(ψ, P)
    φ = _contract_mpo(ψ, P)
    lognorm = try
        last(_right_canonical_with_norm(φ.Γ))
    catch err
        err isa MPOAnnihilationError || rethrow()
        return Inf
    end
    _annihilated(lognorm, ψ, P) && return Inf
    lognorm_in = log(inner_product(ψ)) / 2
    return max(-(lognorm - lognorm_in) / ψ.n, 0.0)
end

export release_recorder
"""
    observe, record = release_recorder(ψ0; dt, site_ops=(;), fidelity=true)

An observer for the MPO [`evolve!`](@ref) of a release from `ψ0`, and the
record it fills (a named tuple of vectors, one entry per sample):

- `step`, `t = step · dt` (physical time of the sampled state);
- `entropy[k][b]`: entanglement entropy of bond `b` (`ent_S`, natural log);
- `dims[k][b]`: retained bond dimension (`length(λ[b])`);
- `max_discarded`, `num_saturated` (requested dimension exceeded `maxdim`),
  `num_at_cap` (retained dimension equal to `maxdim`), `max_kept_dim`: over the
  applications since the previous sample (zero at `t = 0`);
- `ops[name][k][j]`: `⟨O⟩` on site `j` of the cell for each `name => O` in
  `site_ops` (e.g. `(; n = [0 0; 0 1])` gives `n_A`, `n_B` on a two-site cell);
- `return_fidelity[k]`: `per_site_fidelity(ψ0, ψ(t))` (amplitude per site; the
  squared overlap per site is its square).

It only reads the state. The project decides how to score entropies (maximum or
mean over bonds, time window) and what counts as saturation; the record keeps
the separate measurements "retained dimension reached the cap", "requested
dimension exceeded the cap" and the entropies themselves.

```julia
obs, rec = release_recorder(ψ0; dt=0.05, site_ops=(; n=N))
evolve!(deepcopy(ψ0), U, 400; maxdim=32, observe=obs, every=5)
```
"""
function release_recorder(ψ0::iMPS; dt::Real, site_ops=(;), fidelity::Bool=true)
    ref = deepcopy(ψ0)
    names = collect(keys(site_ops))
    record = (step=Int[], t=Float64[], entropy=Vector{Float64}[], dims=Vector{Int}[],
              max_discarded=Float64[], num_saturated=Int[], num_at_cap=Int[], max_kept_dim=Int[],
              ops=Dict{Symbol,Vector{Vector{Float64}}}(k => Vector{Float64}[] for k in names),
              return_fidelity=Float64[])
    observe = function (φ, k, stats)
        push!(record.step, k)
        push!(record.t, k * dt)
        push!(record.entropy, [ent_S(φ, b) for b in 1:φ.n])
        push!(record.dims, length.(φ.λ))
        push!(record.max_discarded, isnothing(stats) ? 0.0 : stats.max_discarded_weight)
        push!(record.num_saturated, isnothing(stats) ? 0 : stats.num_saturated)
        push!(record.num_at_cap, isnothing(stats) ? 0 : stats.num_at_cap)
        push!(record.max_kept_dim, isnothing(stats) ? maximum(length, φ.λ) : stats.max_kept_dim)
        for name in names
            O = site_ops[name]
            push!(record.ops[name], [real(expect(φ, O, j, j)) for j in 1:φ.n])
        end
        fidelity && push!(record.return_fidelity, per_site_fidelity(ref, φ))
        return nothing
    end
    return observe, record
end
