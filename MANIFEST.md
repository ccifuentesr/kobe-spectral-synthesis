# Repository manifest

What to place in the published repository, and where each item comes from.
Paths are relative to the repo root. Run from the current working copy
(`kobe_stars/`) when copying.

## Commit to git

| Item | Source | Notes |
| --- | --- | --- |
| `kobe.spectral_synthesis.py` | `kobe_stars/kobe.spectral_synthesis.py` | Main pipeline. Env-var configurable (`KOBE_ISPEC_DIR`, `KOBE_DIR`, `KOBE_SPECTRA_DIR`, with fallbacks). |
| `kobe.merge_orders.py` | `kobe_stars/kobe.merge_orders.py` | Upstream merger: 2D CARMENES templates → merged 1D input. Env-var configurable (`KOBE_TEMPLATE_DIR`). |
| `README.md` | this folder | |
| `requirements.txt` | this folder | Direct deps only; iSpec installed separately. |
| `.gitignore` | this folder | |
| `MANIFEST.md` | this folder | This file. |
| `ges_lines_kobe_masked.txt` | `kobe_stars/ges_lines_kobe_masked.txt` | Default Fe line list (~12 KB). |
| `kobe_allstars.template.csv` | this folder | Column-only catalog template (no data). |
| `LICENSE` | iSpec `LICENSE` (verbatim AGPL-3.0 text) | AGPL-3.0, chosen to match the iSpec dependency. Copyright + third-party iSpec notice are in the README. |

Suggested copy commands:

```bash
cd /Users/ccifuentesr/Library/CloudStorage/Dropbox/CODE/kobe_stars
cp kobe.spectral_synthesis.py kobe.merge_orders.py ges_lines_kobe_masked.txt \
   /Users/ccifuentesr/Library/CloudStorage/Dropbox/CODE/kobe_repo/
```

## Confidential — never commit

Proprietary KOBE program data. Excluded by `.gitignore`; the script reads them
at runtime but they are not part of the repository. The catalog schema is
documented by `kobe_allstars.template.csv` and in the README.

| Item | Size | Where it goes | Notes |
| --- | --- | --- | --- |
| `kobe_allstars.csv` | ~120 KB | repo root (local only) | Target catalog. Confidential. |
| `gbs_spectra/*.fits`, `*.fts` | ~26 MB | `gbs_spectra/` | Benchmark reference spectra. Confidential. |
| `{TARGET}_merged.fits` | varies | `KOBE_SPECTRA_DIR` (outside repo) | Per-target input, from `kobe.merge_orders.py` (SERVAL-corrected). Confidential. |

## Do NOT commit — download separately (public)

Large public data product; `.gitignore` excludes it. Document the download in
the README.

| Item | Size | Where it goes | Source |
| --- | --- | --- | --- |
| `parsec/parsec_isochrones.dat.txt` | ~36 MB | `parsec/` | PARSEC CMD service (CMD 3.x output, v1.2S tracks). |

## External, not part of this repo

- **iSpec** installation (`KOBE_ISPEC_DIR`) with its `input/` data and the
  compiled Turbospectrum/SPECTRUM backends. See README → Prerequisites.

## Not for the repo

- `kobe.spectral_synthesis_deprecated.py` — superseded; do not publish.
- All other `kobe.*.py` scripts in `kobe_stars/` — separate tools, not imported
  by the synthesis script.
