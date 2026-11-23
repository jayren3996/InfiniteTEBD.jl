# Changelog

## 0.3.x

Layer MPOs and ScarFinder building blocks.

### Changed

- ScarFinder can reuse the normalized Schmidt basis left by built-in MPO
  evolution for a well-separated projection with an isolated transfer fixed
  point, then restore the gauge on the
  retained support. Tensor entries can differ from an explicit
  `evolve!`–`project!` loop.
- MPO compression and iMPS overlaps share the mixed-transfer eigensolver.
  Compression retains the complex scale even when the virtual support is
  orthogonal to the former all-ones Krylov seed.
- Dense local gates preserve the physical dimension of each site in cells
  with different local dimensions.
- Energy correction prepares local observables once per solve and reuses
  them across imaginary-time trials.
- `ITensors` and `ITensorMPS` are weak dependencies. `InfiniteTEBD.imps2mps`
  is available after `using ITensorMPS`; `using InfiniteTEBD` alone loads in
  about 1 s instead of 4 s.
- One ScarFinder driver for every evolution: an iteration is
  `evolve!(ψ, U, nstep)` followed by the new `project!(ψ, χ; terms, target)`
  (`truncate!`, then `energy_fix!`). `U` is a gate list, a vector of layer MPOs
  or a single local gate; a matrix density `h` stands for uniform terms, also
  with MPOs. Breaking changes on the gate and Hamiltonian forms:
  - `scarfinder_step!` returns `(loss, energy_error, converged)` instead of `ψ`;
  - the projection is `truncate!` (bond-symmetric canonical truncation), so
    trajectories differ slightly from the sequential `truncate_unitcell!` split;
  - the energy correction is `energy_fix!`; it needs a complex state, and its
    step-size keyword is `alpha` (was `α`);
  - `span` and `hspan` are inferred from the operators and no longer accepted;
    `refine_gates` / `refine_mpos` are `refine_U`;
  - `observe` and `history` work for every form.
- `truncate_unitcell!` and `fix_energy!` are deprecated in favour of
  `truncate!` and `energy_fix!`.

### Added

- `iMPO`: infinite MPO with tensors `(Dl, d_out, d_in, Dr)` and cell length
  `n`; `bonddims`, operator product `A * B` / `mpo_multiply` with optional
  compression, `compress_mpo`.
- `layer_mpo(layer, n)`: exact infinite MPO of a layer of commuting local gates
  by operator Schmidt decomposition. Supports may be longer than the cell
  (`(i, j)` with `i ≤ j` are chain sites). `check_commuting(layer; n)` verifies
  the precondition; violations throw.
- `apply!(ψ, W)`: MPO application followed by canonical form and per-bond
  truncation (`maxdim`, `cutoff`, `truncerr`), with `evolve!`-style stats;
  works for non-unitary MPOs.
- `trotter_mpos(layers, dt; trotter, evolution, n, gate_map)` and
  `evolve!(ψ, mpos::Vector{<:iMPO}, steps)`, which merges the half steps at
  step boundaries.
- `truncate!(ψ, χ)`: bond-symmetric truncation. With `ITensorMPS` also loaded,
  qualify it as `InfiniteTEBD.truncate!` (as for `expect`).
- `energy_density(ψ, terms; n)` for site-dependent terms, `energy_fix!` with
  imaginary-time layer MPOs returning `(converged, residual, steps)`,
  `per_site_fidelity`, and `scarfinder_step!(ψ, mpos, χ; …)` returning
  `(loss, energy_error, converged)`.

- Sector-resolved update for controlled-center layers such as PXP gates on a
  two-site cell. `layer_mpo` detects the structure and the MPO updates use it.
- `Blockade` and `iMPO(::Blockade, n)` represent nearest-neighbour blockade.
  Standalone `constraint_leakage` uses a positive transfer for the blockade,
  with relative accuracy for small weights and invariance under tiling and
  rescaling.
- `evolve!(…; observe, every)` samples complete Trotter steps without changing
  the trajectory. `release_recorder` records those samples.
- The MPO `scarfinder!` supports `history`, `observe` and refinement.
- `rand_iMPS(rng, T, n, d, dim)` and `translate(ψ, k)`.

### Removed

- Removed the experimental `ScarFinderPlan`, `projected_energy_density`, and
  MPO ScarFinder options `repair`, `max_project`, `leak_tol`, `projected_energy`,
  `constraint`, `gate_map`, `max_rejections` and `evolve_truncerr`, along with
  their repair, leakage, projected energy and evolution diagnostics.
  MPO `scarfinder_step!` returns `(loss, energy_error, converged)`.
- Removed `gate_map`, `max_rejections`, `status`, `initial_residual` and
  `rejected` from MPO `energy_fix!`. The PXP measurements found that repair
  and projector-folded imaginary time brought no improvement, stalled
  corrections in 9–37% of iterations and took 2–3× as long. The production loop
  uses `trotter_mpos` with `gate_map` for the real-time window, followed by
  truncation and ordinary imaginary time.

### Fixed

- MPO evolution merges adjacent step boundaries in the order the operators
  act. Mixed transfer solves use a fixed unstructured seed so that virtual
  gauges do not hide the dominant eigenvalue.
- `apply!` is transactional and throws `MPOAnnihilationError` for zero
  projections, including nilpotent cells. The annihilation test distinguishes
  small finite weights on long cells from roundoff. `Blockade` transfer
  rescales per site to prevent underflow.
- `apply!(…; renormalize=true, recanonicalize=false)` normalizes the state and
  the stored Schmidt spectra.
- The controlled-layer fast path follows the stored MPO tensors, including
  construction cutoff, and falls back after an in-place tensor edit.
- ScarFinder validates Hamiltonian term sizes before evolution.
  `CanonicalFormError` identifies non-converged canonicalization.
- `truncate!` leaves the input intact if restoring the canonical form fails.
  `project!` accounts for the input norm when measuring projection loss.

### Changed

- Apply prepared `EndpointGate` and `ControlledGate` objects with `applygate!`.
  Set `op_rtol` when preparing an `EndpointGate`. The compatibility entry
  point `apply_endpoint_gate!` still accepts two-site matrices and SWAP routing.

### Notes

- `apply!` follows Orús–Vidal (contract, canonicalize, truncate all bonds) with
  the single-layer QR canonicalization of Vanderstraeten–Haegeman–Verstraete
  (Algorithms 1–2) after an exact removal of null directions. Schmidt values
  are resolved to ~1e-15 relative; the fixed-point `canonical!` resolves ~1e-8.
- `truncate_unitcell!` is unchanged; its docstring now contrasts it with
  `truncate!`.
- Intermediate imaginary-time layers skip regauging. Generic canonicalization
  builds QR/LQ isometries on demand. MPO saturation statistics count Schmidt
  values at or above the truncation floor.
