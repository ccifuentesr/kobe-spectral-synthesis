# KOBE spectral synthesis

Spectral synthesis and stellar parameter estimation for CARMENES/KOBE merged
spectra, built on top of [iSpec](https://www.blancocuaresma.com/s/iSpec).
Given a 1D, RV-corrected spectrum, the pipeline measures the radial velocity by
cross-correlation and then fits the effective temperature and metallicity
(Teff, [M/H]) by spectral synthesis. Crucially, surface gravity is not
fitted: log g is fixed from the stellar radius and mass, where the radius comes
from the Stefan–Boltzmann law (bolometric luminosity from SED fitting + Teff)
and the mass from an empirical mass–luminosity relation (Eker et al. 2018). With log g
fixed this way and the remaining nuisance parameters (vmic, vmac, vsini) also
held fixed, the synthesis solves only for Teff and [M/H]. Stellar age can then
estimated from PARSEC isochrones.

The pipeline is tuned for **late-K dwarfs** (K5–K7 V, Teff ≈ 4000–4800 K)
observed with **CARMENES VIS** (R ≈ 94 600, 562–920 nm). The merged 1D spectra
are produced upstream by `kobe.merge_orders.py` and are already RV-corrected
(the [SERVAL](https://github.com/mzechmeister/serval) pipeline). It can be
adapted to other instruments and spectral types.

> This repository contains two scripts: `kobe.spectral_synthesis.py` and
> `kobe.merge_orders.py`, the upstream tool that turns 2D CARMENES echelle
> templates into the merged 1D input the pipeline reads.

---

## Prerequisites

### 1. iSpec + a radiative transfer code (required, external)

iSpec is **not** available on PyPI and is **not** bundled here. You must install
it separately and compile its radiative transfer backends:

- iSpec: <https://www.blancocuaresma.com/s/iSpec> (clone, build the Cython
  extensions, and download the bundled input data: line lists, model
  atmospheres, etc.).
- **Turbospectrum** is required for the default configuration
  (`SYNTH_CODE = "turbospectrum"`, with molecular line lists enabled for the
  TiO bands that shape the continuum in cool K dwarfs). SPECTRUM is supported as
  a faster, atomic-only alternative for testing (`SYNTH_CODE = "spectrum"`).

The script consumes several files that ship with the iSpec installation, under
`$KOBE_ISPEC_DIR/input/`:

| Resource | Path (relative to iSpec root) |
| --- | --- |
| Telluric CCF mask | `input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst` |
| Strong absorption lines | `input/regions/strong_lines/absorption_lines.txt` |
| Atomic line list (VALD) | `input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv` |
| Model atmospheres (MARCS GES) | `input/atmospheres/MARCS.GES/` |
| Isotopes | `input/isotopes/SPECTRUM.lst` |
| Solar abundances (Grevesse 2007) | `input/abundances/Grevesse.2007/stdatom.dat` |
| Line region masks | `input/regions/47000_GES/`, `input/regions/47000_VALD/` |
| CCF mask (K5) | `input/linelists/CCF/HARPS_SOPHIE.K5.378_680nm/mask.lst` |

### 2. Python packages

Python 3.12 is recommended. Install the direct dependencies with:

```bash
pip install -r requirements.txt
```

These are only what the script imports directly (`numpy`, `scipy`,
`matplotlib`, `astropy`, and optionally `termcolor`). iSpec brings its own
dependencies (`pandas`, `statsmodels`, `Cython`, `h5py`, …) — install those from
the iSpec `requirements.txt`.

### 3. Data files

The following data must be present (see [`MANIFEST.md`](MANIFEST.md) for exact
locations and sources). None of the large or proprietary files are committed to
git — you supply your own:

| File / directory | Size | Purpose | In repo? |
| --- | --- | --- | --- |
| `ges_lines_kobe_masked.txt` | ~12 KB | Default Fe line list (GES, 7 Teff-biasing lines masked) | yes |
| `kobe_allstars.template.csv` | tiny | Column-only template for the target catalog | yes |
| `kobe_allstars.csv` | ~120 KB | Target catalog (seeds, luminosities, flags) | **no — user must provide** |
| `gbs_spectra/` | ~26 MB | Gaia Benchmark Star reference spectra for validation | **no — user must provide** |
| `{TARGET}_merged.fits` | varies | Input spectra, one per target (in `KOBE_SPECTRA_DIR`) | **no — user must provide** |
| `parsec/parsec_isochrones.dat.txt` | ~36 MB | PARSEC v1.2S isochrones → log g, mass, age | no — download |

> The target catalog and all spectra (benchmark and per-target) are not
> distributed with this repository — you supply your own. To run the pipeline on
> your own targets, provide a `kobe_allstars.csv` matching the schema in
> `kobe_allstars.template.csv` and place your merged spectra in
> `KOBE_SPECTRA_DIR`.

#### Target catalog schema

`kobe_allstars.csv` is read once and keyed by `KOBE_id`. The script uses these
columns (missing values are tolerated and fall back to defaults or skip the
affected step):

| Column | Meaning |
| --- | --- |
| `KOBE_id` | Target identifier (matches the FITS basename and the CLI argument) |
| `Teff_mamajek`, `eTeff_mamajek` | Seed Teff [K] and its uncertainty (V–K photometric) |
| `Fe_H_marfil`, `Simbad_Fe_H` | Seed [Fe/H] [dex]; falls back to Simbad, then 0.0 |
| `eFe_H_marfil` | Uncertainty on seed [Fe/H] |
| `Lbol`, `Lberr` | Bolometric luminosity [L☉] and uncertainty (for log g / mass / age) |
| `logRHK`, `elogRHK` | Activity index (for activity flagging) |
| `M_Msol`, `eM_Msol`, `R_Rsol`, `eR_Rsol` | Catalog mass / radius [solar units] |

---

## Configuration

All machine-specific paths are read from environment variables, with fallbacks
to the original author setup. Set these before running (recommended), or edit
the defaults near the top of `kobe.spectral_synthesis.py`:

```bash
export KOBE_ISPEC_DIR=/path/to/ispec             # iSpec root (contains input/)
export KOBE_DIR=/path/to/this/repo/              # repo root (trailing slash)
export KOBE_SPECTRA_DIR=/path/to/merged_spectra  # holds {TARGET}_merged.fits
export KOBE_TEMPLATE_DIR=/path/to/templates      # merge_orders: 2D templates in / merged out
```

Synthesis behaviour is controlled by module-level constants in the script
(`SYNTH_CODE`, `USE_MOLECULES`, `LINELIST_SOURCE`, the GBS catalog, vmic/vmac
strategies, quality-flag thresholds). They are documented inline.

---

## Usage

### Producing the input spectra (`kobe.merge_orders.py`)

The pipeline expects merged 1D spectra. If you start from raw CARMENES echelle
templates (`{TARGET}_template.fits`, 61 orders × 14797 px), run the upstream
merger first. It trims order edges, resamples onto a uniform grid, blends
overlaps with an S/N-weighted ramp, applies the SERVAL stellar RV, and converts
vacuum → air wavelengths. Input and output live in `$KOBE_TEMPLATE_DIR`:

```bash
python kobe.merge_orders.py                 # process all KOBE-*_template.fits
python kobe.merge_orders.py KOBE-001        # one target
python kobe.merge_orders.py --plot KOBE-001 # also write a check plot
```

Copy or point `KOBE_SPECTRA_DIR` at the resulting `{TARGET}_merged.fits` files.
If your spectra already are merged, RV-corrected 1D FITS, you can skip this step.

### Running the pipeline (`kobe.spectral_synthesis.py`)

Run on a single target (input read from `$KOBE_SPECTRA_DIR/<TARGET>_merged.fits`):

```bash
python kobe.spectral_synthesis.py KOBE-001
```

Results are appended to `output/kobe_results.csv`. Common options:

| Option | Description |
| --- | --- |
| `--snr` | Print S/N diagnostics and exit |
| `--plot-spectrum` | Show the preprocessed spectrum and exit (no fit) |
| `--diagnostic-plots` | Write RV-check, mask-overlay, fit and PARSEC HRD plots |
| `--batch NAME [NAME ...]` | Fit several targets in parallel |
| `--workers N` | Parallel workers for `--batch` (default 1, ≤6 recommended) |
| `--gbs-test` | Validate the pipeline against Gaia Benchmark Stars |
| `--gbs-star NAME` | Restrict the GBS test to specific stars |
| `--line-subset MODE` | Fe-line diagnostic subset (default `all`) |
| `--line-subset-test` | Run all Fe-line subsets and summarize |
| `--ionization-test` | Compare [Fe/H] from Fe I vs Fe II |
| `--fe2-line-test` | Per-line Fe II diagnostics |
| `--continuum-mode MODE` | Continuum-normalization stress test |
| `--vmic-fixed KM_S` | Override the fixed cool-K microturbulence |
| `--results-csv PATH` | Override the cumulative results CSV path |

Validate the installation end-to-end against the benchmark stars:

```bash
python kobe.spectral_synthesis.py --gbs-test
```

This prints a ΔTeff / Δ[M/H] / Δvmic table against literature reference values
and does not write any calibration file. Note that `--gbs-test` requires the
benchmark reference spectra under `gbs_spectra/`, which are not distributed with
this repository (see Data files above); supply your own to use it.

---

## Pipeline overview

1. **Read & preprocess** the merged FITS spectrum (telluric cleaning, synthesis
   windowing, spline continuum normalization).
2. **Radial velocity** via cross-correlation against a K5 mask.
3. **vsini** estimate from the CCF FWHM.
4. **Spectral synthesis fit** for Teff and [M/H] (iSpec + Turbospectrum) over
   selected Fe-line regions, with log g, vmic, vmac and vsini held fixed.
5. **Derived quantities**: radius from the Stefan–Boltzmann law (SED bolometric
   luminosity + Teff), mass from the Eker 2018 mass–luminosity relation, and
   log g from that mass and radius (fixed during the fit, then refreshed with the
   fitted Teff). Stellar age from PARSEC isochrones.
6. **Quality flags** (low S/N, RV offset, bad fit, vsini discrepancy, activity).

Outputs land in `output/` (`kobe_results.csv` plus optional diagnostic PDFs).

---

## Citing

If you use this pipeline, please cite iSpec
(Blanco-Cuaresma et al. 2014; Blanco-Cuaresma 2019) and the relevant radiative
transfer code (Turbospectrum), model atmospheres (MARCS/GES) and line lists
(VALD3). The Gaia Benchmark Star reference values follow Heiter et al. 2015 and
Jofré et al. 2014.

## License

This pipeline (`kobe.spectral_synthesis.py`, `kobe.merge_orders.py` and the
accompanying files in this repository) is released under the
**GNU Affero General Public License v3.0** — see [`LICENSE`](LICENSE).

### Third-party software (not covered by this repository's copyright)

This pipeline depends on, and imports at runtime, **iSpec**, an independent
package authored by **Sergi Blanco-Cuaresma** and distributed separately under
its own AGPL-3.0 license. iSpec is *not* part of this repository, is *not*
authored by the KOBE team, and retains its own copyright. The AGPL-3.0 license
chosen here matches iSpec's to keep the combined work license-compatible. iSpec
in turn wraps third-party radiative-transfer codes (Turbospectrum, SPECTRUM),
model atmospheres (MARCS/GES) and line lists (VALD3), each under its own terms.
Please obtain iSpec from its official site and comply with its license:
<https://www.blancocuaresma.com/s/iSpec>.
