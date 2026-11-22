# Changelog

## Unreleased

### Fixed

- `evolve!` with MPOs merged step boundaries in the wrong order (`last·first`
  instead of `first·last`). Schedules from `trotter_mpos` were unaffected (their
  first and last layers coincide); user sequences with different end layers
  were wrong.
- `per_site_fidelity` / `inner_product` of two different states could converge
  to a wrong eigenvalue of the mixed transfer matrix when the states differed by
  a virtual gauge (identity Krylov seed); mixed transfers now start from a fixed
  unstructured seed.
- `apply!` is transactional: on failure the input is unchanged. A projector that
  annihilates the state throws `MPOAnnihilationError` (including cells whose
  tensors are nonzero but whose transfer matrix is nilpotent) instead of a LAPACK
  error or a state built from roundoff.
- `apply!(…; renormalize=true, recanonicalize=false)` now normalizes the state,
  not only the stored spectra.

### Added

- Sector-resolved update for controlled-center layers (PXP-type gates on a
  two-site cell, detected by `layer_mpo`, used by `apply!`/`evolve!`/
  `scarfinder_step!`/`energy_fix!`; `structured=false` forces the generic path).
- `scarfinder_step!` diagnostics: `evolve_max_discarded`, `evolve_discarded_sum`,
  `evolve_saturated`, `evolve_structured`, `leakage_evolved`,
  `leakage_projected`, `leakage_fixed`, `energy_fix_steps`; options
  `constraint` and `evolve_truncerr` (opt-in). `constraint_leakage`.
- `Blockade`: the nearest-neighbour blockade as an explicit constraint, with
  a positive-transfer `constraint_leakage` (exact for small weights, invariant
  under tiling and rescaling) and an exact block-copy projection;
  `iMPO(::Blockade, n)` gives the equivalent projector MPO.
- `projected_energy_density(ψ, P, terms)` and `scarfinder_step!(…;
  projected_energy=true)` (`energy_projected`, `energy_projected_error`): the
  energy of the normalized physical component `Pψ/‖Pψ‖` of the returned
  candidate. Opt-in; the loss and the ranking are unchanged.
- `scarfinder_step!` validates all static inputs (constraint compatibility,
  `target`, `terms`, energy-correction and truncation parameters) before it
  modifies the state.

### Changed

- The generic canonicalization forms the QR/LQ isometries only when needed
  (bit-identical results, fewer allocations).
- MPO-path saturation statistics count only Schmidt values at or above the
  truncation floor `max(cutoff, 1e-14)` (`chi_req`, `saturated`, `num_saturated`).

## 0.3.0

Layer MPOs and ScarFinder building blocks. The existing API is unchanged.

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

### Notes

- `apply!` follows Orús–Vidal (contract, canonicalize, truncate all bonds) with
  the single-layer QR canonicalization of Vanderstraeten–Haegeman–Verstraete
  (Algorithms 1–2) after an exact removal of null directions. Schmidt values
  are resolved to ~1e-15 relative; the fixed-point `canonical!` resolves ~1e-8.
- `truncate_unitcell!` is unchanged; its docstring now contrasts it with
  `truncate!`.
