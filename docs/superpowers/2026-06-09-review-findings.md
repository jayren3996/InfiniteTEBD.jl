# Systematic bug & performance review — 2026-06-09

Fan-out review of the full package (dense backend, TensorKit extension, tests/CI,
interop) with adversarial verification. Method: 9 slice reviewers (one per source
area, each given the unverified P1 watch-list from the 2026-05-25 triage), then one
adversarial verifier per P0/P1 finding, instructed to refute and to run Julia
reproductions where decisive. 22 agents total; 42 raw findings → 39 unique →
13 verified (12 confirmed, 1 refuted) + 26 unverified P2 minors. The orchestrator
independently reproduced two additional findings.

Status of the five 2026-05-25 P0s: **all fixed** (`:fourth` single-layer validator,
`iMPS` inner constructor, `tensor_decomp!` n==1 guard, scarfinder test regrouping,
convergence tests present).

Note: the local checkout is behind `origin/master` (PR #4's `TensorKit = "0.16, 0.17"`
widening is not pulled). Line numbers below refer to the local tree at c49c136.

---

## P0 — confirmed, repro'd

### 1. `rand_iMPS(:U1, [-1,1]; …)` silently returns an identically-zero state
`ext/InfiniteTEBDTensorKitExt.jl:219-227` (`_auto_bond_space`) derives all bond
sectors from `fuse(pspace ⊗ pspace)`. For the package's own spin-1/2 U(1) convention
`P = Vect[U1Irrep](1=>1, -1=>1)` (used by `spin_half_ops(:U1)` and the docstring
example at line 242), `fuse(P⊗P) = {+2, 0, -2}` — all even — while `V ⊗ P ← V`
requires odd charge transfer. The hom-space has dimension zero, so every site tensor
is a zero TensorMap. **Verified by repro**: the verbatim docstring example
`rand_iMPS(:U1, [-1, 1]; χ=8, n=2)` gives `norm(Γ) == 0.0` with no error;
`canonical!` succeeds (eigsolve converges to eigenvalue 0), `evolve!` runs, and
`expect`/`energy_density`/`ent_S` return 0.0 — silently. The repo's own test
(`test/test_symmetric_basic.jl:469-481`) constructs exactly such a parity-dead state
and passes because it asserts only `isa`. An imaginary-time ground-state search on
this state would report energy 0 with no diagnostic.
**Fix**: build the auto bond space from sectors that close the charge-conservation
constraint (include the odd class, as `product_iMPS`'s cumulative-charge construction
does), and make the raw constructor throw when `dim(V ⊗ P ← V) == 0`.

---

## P1 — confirmed bugs (silent wrong results or crash)

### 2. Symmetric `applygate!` on an n=1 unit cell silently corrupts the state
`ext/InfiniteTEBDTensorKitExt.jl:973-1026`. For `n=1`, `(i,j)=(1,2)` normalizes to
`(1,1)` and the nearest-neighbour validator passes (`mod1(2,1)==1`). The update then
assigns `ψ.Γ[i] = λL⁻¹·U·S` followed by `ψ.Γ[j] = Vt` with `i == j`, clobbering the
U·S half of the SVD. **Repro**: an identity two-site gate (must be a no-op) drifted
`energy_density` −0.0435 → −0.1050 → −0.1711 across two applications, silently; with
truncation it leaves mismatched wraparound spaces that crash the *next* gate.
**Fix**: reject `ψ.n < 2` up front (dense `tensor_decomp!` guards this).

### 3. `_truncate_unitcell!` n==1 branch stores `λ·B` instead of `B`
`src/ScarFinder.jl:123-128` calls the scalar `schmidt_canonical` kernel — which
already returns the stored-convention tensor (`tensor_rmul!(Γ_new, S_new)`,
Schmidt.jl:227) — and then applies `tensor_lmul!(λ, Γ)` on top, inserting an extra
`diag(λ)` on every bond of the infinite chain (not a gauge transformation). The n>1
path's `lmul` is correct only because `tensor_decomp!` divides it back out; n==1 has
no decomp. **Repro**: on `rand_iMPS(ComplexF64,1,2,4)` at unchanged χ,
right-canonical residual ‖ΣBB†−I‖ = 3.46 (control 4e-14), `expect(Z)` 0.303 → 1.211.
Reachable via `scarfinder_step!`/`scarfinder!` on 1-site unit cells.
**Fix**: mirror the vector kernel's `isone(n)` path (trace normalization, no lmul),
or call `canonical!` for n==1.

### 4. Real-time evolution of a default `Float64` state crashes (`InexactError`)
`src/TensorAlgebra.jl:154-159` (`tensor_umul!`) allocates its buffer as
`Array{eltype(Γ)}`, so a complex gate on a real tensor throws `InexactError`; the
two-site path independently fails at the write-back `ψ.Γ[inds[k]] = Γs[k]`
(Gate.jl:215, `Vector{Array{Float64,3}}` can't hold complex arrays). **Repro**: the
all-defaults composition `evolve!(rand_iMPS(2,2,4), layers, 0.1, 1)` throws —
the untyped `rand_iMPS(n,d,dim)` constructs `Float64` (iMPS.jl:232) and any
real-time gate `exp(-im·dt·h)` is complex. Loud crash, but on the most natural
first-user path. **Fix**: promote the state's eltype once at the
`applygate!`/`evolve!` boundary when a complex gate meets a real state (or raise a
descriptive `ArgumentError` telling the user to construct with `ComplexF64`).

### 5. `canonical!(ψ; renormalize=false)` stores λ scaled by an arbitrary factor
`src/Schmidt.jl:223`. `S_new` are singular values of `Yt·(S_in.*X)` where the fixed
points from `steady_mat` are unit-Frobenius-normalized and the dominant eigenvalue
(the physical norm² per cell) is discarded (Krylov.jl:232-237). With
`renormalize=false` the raw values are kept: **repro** shows `norm(λ) = 1/√D`
exactly for an already-canonical input (D=4 → 0.5; expectation values scale by 1/D),
while the stored tensors remain correct. The λ field convention is silently broken.
**Fix**: rescale `S_new` by the dominant transfer eigenvalue (already computed,
currently discarded), or deprecate `renormalize=false` on `canonical!`.

---

## P1 — confirmed API/contract defects

### 6. Multi-site operator convention is the mirror of what the docs state
`docs/src/observables.md:105` documents "Multi-site operators use `kron` (with the
leftmost site as the leftmost factor)", but `tensor_group_2`
(TensorAlgebra.jl:182-188) fuses the physical leg with the **left site's index
fastest**, while Julia's `kron` makes the **right factor's index fastest** — so
`kron(A,B)` acts B on site i and A on site j. **Repro** (orchestrator + verifier,
through exported API): `expect(ψ, kron(Z,I), 1, 2)` on |01⟩ returns −1 (Z landed on
site 2); `applygate!(ψ, kron(X,I), 1, 2)` on |00⟩ flips site 2; verified through
3 sites (`kron(Z,I,I)` on (1,3) hits site 3). `applygate!`/`expect`/`energy_density`
are mutually consistent, so every reflection-symmetric example in docs and tests
masks it — but any asymmetric operator (S⁺S⁻ correlators, DM terms, staggered
fields) silently gives mirrored physics. The TensorKit backend uses the standard
first-leg-on-site-i convention, so the backends disagree. `convert_operator`
implements exactly the needed reversal but is **not exported** and no test pins the
convention (`test_gate_api.jl:254` only checks involution).
**Fix (decision needed)**: either (a) fix the docs everywhere + export
`convert_operator` + add a convention-pinning regression test, or (b) apply
`convert_operator` internally at the API boundary so the documented convention
becomes true (breaking for users who discovered the real convention empirically).

### 7. `imps2mps` emits an MPS that breaks downstream ITensorMPS
`src/ITensorsInterop.jl:45-49` reuses `links[L]` as the left leg of tensor 1.
**Repro**: for a canonical AKLT iMPS, `inner(mps, mps) = χ` (= 2) instead of 1,
`norm = √χ`; `siteind(mps, 1)` returns the *Link* index; `ITensorMPS.expect` and
`correlation_matrix` — the exact uses the docstring advertises — throw. Also: no
boundary Schmidt weighting, and `L % n != 0` silently glues mismatched Schmidt
bases. **Fix**: distinct left-boundary Index weighted by `diag(ψ.λ[end])`, document
the dangling-edge contract, validate `L % n`.

---

## P1 — confirmed performance bottlenecks

### 8. `canonical!` spends 99.8% of its time in an unused diagnostic at χ ≤ 50
`src/Schmidt.jl:52,63` (`_transfer_degeneracy`): the guard `Dl*Dr > 2500` only skips
the dense path for χ ≥ 51, so at the default `MAXDIM=50` every `schmidt_canonical`
call runs `eigvals` on a 2500×2500 nonsymmetric complex matrix. **Measured**:
5.86 s + 288 MiB per call at χ=50 vs 42 µs at χ=51 (a cliff exactly at the default
cap); stock `canonical!` on a χ=50 state: 5.28 s vs 0.012 s with the diagnostic
stubbed. Worse, `canonical!`'s second pass (iMPS.jl:400-410) passes
`noninjective=:ignore, symmetry_break=:none`, for which the diagnostic's result is
provably unused. **Fix**: skip when `:ignore`/`:none`; otherwise use a 2-3
eigenvalue Krylov solve or lower the dense threshold to ~400.

### 9. Every renormalized single-site gate runs a whole-cell eigenvalue solve
`src/Gate.jl:172` calls `inner_product(ψ)` (full unit-cell KrylovKit solve) per
single-site gate whenever `renormalize=true` — the default on all evolve paths.
For unitary real-time gates this is a mathematical no-op. **Measured** at χ=50:
0.219 ms vs 0.0101 ms per gate (≈21×); on a post-quench state `inner_product` alone
is 14.4 ms/call (≈7× the cost of a full two-site SVD update). **Fix**: compute the
norm locally — for a canonical state `norm² = Σ_a λl[a]²·‖(G·B)[a,:,:]‖²` is
O(χ²d) — or skip when `G` is unitary to tolerance.

---

## P2 — verified

- **`_iterative_svd_trim` crashes for `maxdim > 30`** (TensorAlgebra.jl:299):
  `eigsolve` is called with `howmany = min(maxdim, min(m,n))` and no `krylovdim`;
  every compat-allowed KrylovKit version errors when `howmany > krylovdim=30`.
  Repro'd: `svd_trim(randn(100,100); use_iterative=true)` throws; the auto-select
  path (`maxdim < min(m,n)÷10 && min(m,n) > 200`) with default `maxdim=50` requires
  `min(m,n) ≥ 510`, i.e. *whenever auto actually picks iterative, it crashes*.
  Downgraded from P1 only because the path is opt-in/dead by default.
- **Cross-macro-step stage merging collapses single-layer `:second` schedules**
  (Gate.jl:444-453): `_trotter_stage_schedule(1, :second, 300) == [(1, 300.0)]`,
  so `evolve!(ψ, layers, dt, steps)` builds `exp(-i·steps·dt·h)` applied once.
  Exact *within* the documented "terms in a layer commute" contract; silently
  divergent outside it (repro: ⟨Z⟩ 0.922 vs 0.180). Consider refusing
  `num_layers==1 && steps>1` like `:fourth` now does, or not merging across
  macro-step boundaries.
- **`_evolve_uniform!` crashes instead of taking the slow path**
  (ScarFinder.jl:84, orchestrator repro): the uniformity check
  `ψ.λ[1] ≈ ψ.λ[i]` throws `DimensionMismatch` when bond dimensions differ across
  bonds (e.g. `[2,1]` after one 2-site gate) — reachable via
  `scarfinder_step!`/`scarfinder!`/`energy_span` with any single-site gate.
  Fix: `size(a) == size(b) && a ≈ b`.

## P2 — flagged by reviewers, not adversarially verified

Correctness/numerics:
- `expect` does not wrap site indices: `expect(ψ, h, n, n+1)` throws `BoundsError`,
  inconsistent with `applygate!`/`ent_S` (iMPS.jl:559).
- `conj(mps)` on a real-eltype iMPS aliases storage (`Base.conj` is identity on real
  arrays); later in-place single-site gates mutate the source (iMPS.jl:586).
- Symmetric `ent_S` doesn't wrap the bond index (parity break with dense) (ext:1057).
- `_symmetric_tsvd` has no keep-at-least-one floor: full truncation silently
  annihilates the state (ext:405).
- Symmetric `canonical!`'s non-injectivity refusal compares λ_r vs λ_l, which are
  equal precisely for the degenerate inputs it's meant to catch (ext:647).
- `_energy_fix!` gives up silently on the diverging branch; the warning only covers
  loop exhaustion (ScarFinder.jl:337).
- `_truncate_unitcell!` large-cell (n>6) fallback returns a non-canonical state,
  contradicting its docstring (ScarFinder.jl:136).
- Krylov fixed-point hermitization ignores the eigenvector's arbitrary complex
  phase; only a real sign flip is applied (Krylov.jl:162). `steady_mat` proceeds on
  unconverged solves (warn only), and `_krylov_opts` cannot forward `krylovdim`
  (Krylov.jl:153).
- `_dominant_chain_eigenvalue`'s Krylov path uses a deterministic identity seed;
  symmetry-blocked mixed transfers can converge in the wrong sector
  (Miscellaneous.jl:237).
- Duplicate `_support_tol` methods with divergent semantics (TensorAlgebra.jl:58 vs
  iMPS.jl:169): the override adds a `max(scale, 1)` floor, so behavior depends on
  eltype dispatch.
- `natural_bonddim` applies `cutoff` to squared weights → effective floor is
  `sqrt(cutoff)` (Miscellaneous.jl:74).
- `_svd_with_fallback` retries the same DivideAndConquer algorithm that just failed
  and perturbs the matrix as a first resort (TensorAlgebra.jl:255).
- `_iterative_svd_trim` ill-conditioning heuristic tests the smallest matrix
  *entry*, not singular values — spurious warning for any matrix containing a zero
  (TensorAlgebra.jl:278).
- `tensor_svd` silently ignores `mindim` when `truncerr === nothing`
  (TensorAlgebra.jl:616).

Performance:
- `canonical!`/`schmidt_canonical` group the whole cell to `(χ, d^n, χ)`:
  O(d^n·χ³) per fixed-point matvec — exponential in unit-cell length
  (Schmidt.jl:270). A per-bond Orús-Vidal sweep is linear in n.
- `chi_policy=:adaptive` runs a full `canonical!` after every gate (Gate.jl:547).
- `_dominant_eigenvalue_dense` computes full `eigen` (vectors discarded) instead of
  `eigvals` up to 4096×4096 (Miscellaneous.jl:276); the dense-vs-Krylov threshold
  inspects only chain-edge bond dims (Miscellaneous.jl:232).
- Per-gate allocation churn in `svd_trim` slice copies + `tensor_group` even when
  nothing is truncated (TensorAlgebra.jl:437).

API/tests/CI:
- Exported `expect` collides with `ITensorMPS.expect` in the advertised interop
  workflow (InfiniteTEBD.jl:9).
- The `bench` test group (incl. PXP legacy physics regressions) is unreachable from
  CI, which runs `default` (.github/workflows/CI.yml:36).
- `test_performance_improvements.jl:117` never reaches the iterative-SVD branch it
  is meant to test (n=120 < the 200 threshold) — which is why P2 item #1 above went
  unnoticed.
- Stale `.appveyor.yml` tests Julia 1.4/nightly vs the 1.10 floor.
- No test pins the operator-ordering convention (test_gate_api.jl:254).

---

## Refuted during verification

- "CI never exercises newest compat-allowed dependency versions (TensorKit caps
  ITensors/KrylovKit)" — **refuted by direct CI-log evidence**: run 26839553524
  (commit d0a1ea1) resolved and tested ITensors 0.9.30 + KrylovKit 0.10.3 +
  TensorKit 0.16.5 together; run 27060883856 (PR #4 merge) resolved TensorKit
  0.17.0 on both matrix legs, all green.

## Suggested fix order

1. **#1** ext zero-state (P0; small fix + a `norm > 0` construction test).
2. **#4** eltype promotion at the gate boundary (first-user crash on defaults).
3. **#8** `_transfer_degeneracy` skip/threshold (one-line guard; 5.9 s → 12 ms
   `canonical!` at the default χ).
4. **#6** operator-convention decision (docs+export+pinning test at minimum).
5. **#2, #3, #5** state-corruption fixes (each with a regression test asserting an
   identity operation is a no-op).
6. **#9** local-norm single-site renormalization; **#7** imps2mps boundary contract.
7. P2 batch, starting with the verified three.

---

## Status update — fixed 2026-06-10 (branch `fix/review-2026-06-09`)

All P0/P1 findings and the three verified P2s were fixed; regression tests live
in `test/test_review_fixes.jl` (dense core), `test/test_itensors_interop.jl`
(interop), and the symmetric test files (ext). Resolutions that involved a
decision:

- **#6 convention**: option (b) — `applygate!`/`expect` now convert at the API
  boundary (`_operator_to_internal`), so the documented kron convention
  (leftmost factor ↔ leftmost site) is the real one and the dense backend
  agrees with the TensorKit backend. `convert_operator` is unchanged
  (involution) and documented as the low-level escape hatch. **Breaking** for
  users who had discovered the old mirrored convention empirically.
- **#4 eltype**: in-place promotion is impossible (concretely-typed storage);
  `applygate!`/layer-`evolve!` now throw a descriptive `ArgumentError` before
  mutating, and `mps_promote_type` is exported as the remedy.
- **#9 single-site renormalize**: unitary gates (checked to 1e-10·d) skip
  renormalization entirely — 219 µs → 7.7 µs per gate at χ=50; non-unitary
  gates keep the exact whole-cell `inner_product` (a local estimate was tried
  and measurably drifts as repeated non-unitary updates leave canonical form).
- **#5 renormalize=false**: kernel rescales the returned spectrum by `‖M‖_F`
  (the gauge artifact); stored tensors keep the state's norm.
- **#8 degeneracy diagnostic**: skipped outright when
  `noninjective=:ignore, symmetry_break=:none` (canonical!'s second pass), and
  the dense-eigvals threshold lowered 2500 → 400 so the default χ ≤ 50 path
  uses the cheap structural heuristic; `canonical!` at χ=50: 5.3 s → 0.5 ms
  (n=1) / 8.5 ms (n=2).
- **P2 trotter collapse**: single-layer `:second` schedules no longer merge
  across macro steps (one stage per step); multi-layer Strang boundary merging
  unchanged.
- Also fixed from the unverified-P2 list (cheap, adjacent): `expect` periodic
  site wrapping, and `conj(ψ)` storage aliasing for real element types.

Remaining open: the rest of the unverified P2 list (notably the exponential
`d^n` unit-cell grouping in `canonical!`, the `:adaptive` per-gate
recanonicalization cost, `expect`'s unconditional `real`, and the
CI-unreachable `bench` test group).
