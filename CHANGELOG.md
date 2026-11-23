# Changelog

## 0.3.x

Layer MPOs and ScarFinder building blocks.

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
- `truncate!(ψ, χ)`: bond-symmetric truncation (extends `ITensorMPS.truncate!`).
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
- The MPO driver uses the MPO step, rejects a positional matrix density
  and validates Hamiltonian term sizes before mutating the state.
  `CanonicalFormError` identifies non-converged canonicalization.

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
