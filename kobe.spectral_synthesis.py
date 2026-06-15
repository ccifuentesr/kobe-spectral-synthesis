#!/usr/bin/env python3
#
# Part of the KOBE spectral synthesis pipeline.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for the full license text.
#
# iSpec (imported at runtime) is independent third-party software by
# Sergi Blanco-Cuaresma, distributed separately under its own AGPL-3.0 license.
#
"""
kobe.spectral_synthesis.py

Spectral synthesis and stellar parameter estimation for CARMENES/KOBE merged
spectra using iSpec. Reads from {SPECTRA_DIR}/{TARGET}_merged.fits and estimates
Teff, logg, [M/H] via CCF (RV) and spectral synthesis.

Targets are late-K dwarfs (K5–K7 V, Teff ~ 4000–4800 K) observed with
CARMENES VIS (R ~ 94600, 562–920 nm). The merged 1D spectra are produced by
kobe.merge_orders.py and are already RV-corrected (SERVAL pipeline).

Usage:
    python kobe.spectral_synthesis.py KOBE-001
"""

import os
import sys
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kobe_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must precede any pyplot import
                        # prevents macOS from opening a GUI event loop (Dock icon hang)

import numpy as np
try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text
from astropy.io import fits

# ── Paths ─────────────────────────────────────────────────────────────────────
# Configurable via environment variables; the defaults below reproduce the
# original author setup. To adapt the pipeline to your own machine, either set
# the variables (e.g. export KOBE_ISPEC_DIR=/path/to/ispec) or edit the defaults.
#   KOBE_ISPEC_DIR   — root of the iSpec installation (must contain input/)
#   KOBE_DIR         — root of this repository (catalog, line lists, parsec/, gbs_spectra/)
#   KOBE_SPECTRA_DIR — directory holding the {TARGET}_merged.fits input spectra
ISPEC_DIR         = os.environ.get('KOBE_ISPEC_DIR',   '/Users/ccifuentesr/Library/CloudStorage/Dropbox/CODE/ispec_master/')
KOBE_DIR          = os.environ.get('KOBE_DIR',         '/Users/ccifuentesr/Library/CloudStorage/Dropbox/CODE/kobe_stars/')
SPECTRA_DIR       = os.environ.get('KOBE_SPECTRA_DIR', '/Users/ccifuentesr/Library/CloudStorage/Dropbox/DATA/project_kobe/kobe_master_spectra')
KOBE_CATALOG_FILE = KOBE_DIR + 'kobe_allstars.csv'

sys.path.insert(0, os.path.abspath(ISPEC_DIR))
import ispec

# ── iSpec resource files ───────────────────────────────────────────────────────
TELLURIC_FILE      = ISPEC_DIR + "input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
STRONG_LINES_FILE  = ISPEC_DIR + "input/regions/strong_lines/absorption_lines.txt"
ATOMIC_LINELIST    = ISPEC_DIR + "input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
MODEL_ATMOS_DIR    = ISPEC_DIR + "input/atmospheres/MARCS.GES/"
ISOTOPE_FILE       = ISPEC_DIR + "input/isotopes/SPECTRUM.lst"

# ── Radiative transfer code ────────────────────────────────────────────────────
# "turbospectrum" : Turbospectrum + VALD atomic + molecular linelists (TiO, CaH…)
#                  Recommended for K5–K7 V where TiO bands affect the continuum.
# "spectrum"      : SPECTRUM + GESv6 atomic only (faster, useful for testing).
SYNTH_CODE      = "turbospectrum"
USE_MOLECULES   = True     # include TS molecular linelists (only used if SYNTH_CODE="turbospectrum")
SOLAR_ABUND_FILE   = ISPEC_DIR + "input/abundances/Grevesse.2007/stdatom.dat"

# ── Gaia Benchmark Stars catalog ───────────────────────────────────────────────
# Parameters from Heiter+15 (A&A 582, A49) / Jofré+14 (A&A 564, A133).
# Only dwarf/subgiant stars (logg > 3.5) are included — giants excluded.
# vmic: from Jofré+14 Table 4 where available, else Doyle+14 calibration.
#
# K-dwarf calibrators (KOBE-relevant, 4374–5084 K):
#   61 Cyg A  — NARVAL .fts  : BINTABLE cols AWAV[nm], FLUX_NOR (pre-normalized)
#   ε Eri     — HARPS Phase3 : BINTABLE ext SPECTRUM, cols WAVE[Å], FLUX[ADU], ERR[ADU]
#   HD 36003  — HARPS Phase3 : same structure as ε Eri
#
# format key:
#   'ispec_text'   — iSpec plain text / .txt.gz
#   'harps_phase3' — ESO Phase 3 HARPS BINTABLE (WAVE Å, FLUX ADU) → needs normalisation
#   'narval_fts'   — PolarBase NARVAL .fts BINTABLE (AWAV nm, FLUX_NOR) → pre-normalised
GBS_SPECTRA_DIR = KOBE_DIR + 'gbs_spectra/'
NARVAL_61CYGA_RESOLUTION = 65000   # NARVAL intensity mode (conservative)
HARPS_RESOLUTION = 115000          # HARPS resolving power
CARMENES_VIS_RESOLUTION = 94600    # CARMENES VIS arm
SOPHIE_HR_RESOLUTION = 75000       # SOPHIE HR mode (high-resolution)

GBS_CATALOG = {
    '61CygA': {
        'spectrum'   : GBS_SPECTRA_DIR + 'NARVAL.61cyga.fts',
        'resolution' : NARVAL_61CYGA_RESOLUTION,
        'format'     : 'narval_fts',
        'teff'       : 4374.0,   # Heiter+15
        'logg'       : 4.59,     # Heiter+15
        'MH'         : -0.33,    # Heiter+15
        'vmic'       : 1.10,     # Jofré+14
        'vsini'      : 1.50,     # Kervella+2008
        'e_teff'     : 70.0,
        'e_logg'     : 0.04,
        'e_MH'       : 0.05,
        'source'     : 'Heiter+15 / Jofré+14',
    },
    'HD156026': {
        # 36 Oph C, K5V — fills the gap between 61CygA and the K3V regime above.
        # UVES blue-arm Phase 3 product (R≈74 500, 473-684 nm) — covers KOBE window A only.
        # Reference values from MEDIAN of EW-LTE+SYNTH-Fe literature cluster:
        #   Teff: median 4541 K from N=15 entries  [1σ 4376-4585]
        #   [Fe/H]: median -0.130 dex from N=14 entries  [1σ -0.21, -0.09]
        'spectrum'   : GBS_SPECTRA_DIR + 'UVES.HD156026.fits',
        'resolution' : 74450.0,
        'format'     : 'harps_phase3',
        'teff'       : 4541.0,   # cluster median, EW-LTE+SYNTH-Fe, N=15
        'logg'       : 4.65,
        'MH'         : -0.130,   # cluster median, EW-LTE+SYNTH-Fe, N=14
        'vmic'       : 0.70,
        'vsini'      : 3.00,
        'e_teff'     : 105.0,    # half-width of 1σ literature spread
        'e_logg'     : 0.10,
        'e_MH'       : 0.06,
        'source'     : 'EW-LTE+SYNTH-Fe cluster median (1D LTE Fe-line family)',
    },
    '61CygB': {
        # HD 201092, K7V — Heiter+15 GBS calibrator.  K5V companion of 61CygA;
        # critical anchor at the cool end of the KOBE K5-K7V regime.
        # SOPHIE S1D Phase product (OHP); covers ~387-694 nm (window A only).
        'spectrum'   : GBS_SPECTRA_DIR + 'SOPHIE.61CygB.fits',
        'resolution' : SOPHIE_HR_RESOLUTION,
        'format'     : 'sophie_s1d',
        'teff'       : 4044.0,   # Heiter+15
        'logg'       : 4.67,     # Heiter+15
        'MH'         : -0.38,    # Heiter+15
        'vmic'       : 1.00,     # Jofré+14
        'vsini'      : 1.50,     # literature
        'e_teff'     : 32.0,
        'e_logg'     : 0.04,
        'e_MH'       : 0.04,
        'source'     : 'Heiter+15 / Jofré+14',
    },
    'HD88230': {
        # GJ 380, K7V — full KOBE wavelength coverage (370-1048 nm, all 5 windows).
        # NARVAL .fts (PolarBase), normalised flux.  Not in Heiter+15 GBS sample.
        # Reference values from MEDIAN of 1D-LTE Fe-line-based literature
        # (EW-LTE + SYNTH-Fe methods, methodologically comparable to KOBE):
        #   Teff: median 4086 K from N=15 entries  [1σ 4010-4212]
        #   [Fe/H]: median +0.014 dex from N=10 entries  [1σ -0.11, +0.25]
        # Literature scatter is large (-0.93 to +0.24 across all methods).
        'spectrum'   : GBS_SPECTRA_DIR + 'NARVAL.HD88230.fts',
        'resolution' : NARVAL_61CYGA_RESOLUTION,
        'format'     : 'narval_fts',
        'teff'       : 4086.0,   # cluster median, EW-LTE+SYNTH-Fe, N=15
        'logg'       : 4.62,     # consensus
        'MH'         : +0.014,   # cluster median, EW-LTE+SYNTH-Fe, N=10
        'vmic'       : 0.70,     # Doyle+14 at 4086 K
        'vsini'      : 2.00,     # slow rotator literature
        'e_teff'     : 100.0,    # half-width of 1σ literature spread
        'e_logg'     : 0.10,
        'e_MH'       : 0.18,     # half-width of 1σ literature spread
        'source'     : 'EW-LTE+SYNTH-Fe cluster median (1D LTE Fe-line family)',
    },
    'EpsInd': {
        # HD 209100, K5V — independent K5V anchor (not in Heiter+15 GBS).
        # UVES Phase 3 (R=107 200, SNR≈480, 496-707 nm) — covers KOBE window A only.
        # Reference values from MEDIAN of EW-LTE+SYNTH-Fe literature cluster:
        #   Teff: median 4682 K from N=20 entries  [1σ 4582-4754]
        #   [Fe/H]: median -0.190 dex from N=17 entries  [1σ -0.23, -0.13]
        'spectrum'   : GBS_SPECTRA_DIR + 'UVES.EpsInd.fits',
        'resolution' : 107200.0,
        'format'     : 'uves_reduced',
        'teff'       : 4682.0,   # cluster median, EW-LTE+SYNTH-Fe, N=20
        'logg'       : 4.62,
        'MH'         : -0.190,   # cluster median, EW-LTE+SYNTH-Fe, N=17
        'vmic'       : 0.85,
        'vsini'      : 1.50,
        'e_teff'     : 86.0,     # half-width of 1σ literature spread
        'e_logg'     : 0.10,
        'e_MH'       : 0.05,
        'source'     : 'EW-LTE+SYNTH-Fe cluster median (1D LTE Fe-line family)',
    },
    'BD+24_2733A': {
        # KarmnID J14257+236W, M0.0V (Teff 3900-4100 K).  Quiescent CARMENES
        # GTO reference star (Schöfer+2019).  Same instrument as KOBE targets.
        # ┌──────────────────────────────────────────────────────────────────┐
        # │ ⚠ EXTRAPOLATION BOUNDARY: M0V is at the COOL edge of KOBE        │
        # │   pipeline applicability (calibrated for K5-K7V, 3900-4400 K).   │
        # │                                                                  │
        # │   Literature [Fe/H] for this star varies from -0.10 (Marfil+20,  │
        # │   CARMENES synth) to +0.61 (Passegger+19, PHOENIX-grid) — a      │
        # │   0.7 dex spread driven by methodology, not measurement error.   │
        # │   Teff range 3800-4100 K.                                        │
        # │                                                                  │
        # │   Reference values from MEDIAN of comparable cluster, but the    │
        # │   cluster is poorly sampled at M0V (only N=2 EW-LTE+SYNTH-Fe).   │
        # │   Maldonado+20 (a frequent reference for this star) is PCA-ML,   │
        # │   not Fe-line synthesis — excluded from cluster median.          │
        # │   Treat any Δ here as informative ONLY about the cool boundary.  │
        # └──────────────────────────────────────────────────────────────────┘
        'spectrum'   : GBS_SPECTRA_DIR + 'CARMENES.J14257+236W.fits',
        'resolution' : CARMENES_VIS_RESOLUTION,
        'format'     : 'harps_phase3',
        'teff'       : 3970.0,   # cluster median, EW-LTE+SYNTH-Fe, N=2 (sparse!)
        'logg'       : 4.64,
        'MH'         : +0.010,   # cluster median, EW-LTE+SYNTH-Fe, N=2 (sparse!)
        'vmic'       : 0.70,     # K-dwarf anchor (extrapolation, see caveat above)
        'vsini'      : 2.00,
        'e_teff'     : 80.0,
        'e_logg'     : 0.10,
        'e_MH'       : 0.20,     # large uncertainty due to small N
        'source'     : 'EW-LTE+SYNTH-Fe cluster median (N=2, sparse coverage at M0V)',
    },
}

# K5 mask — appropriate for late-K dwarfs observed with CARMENES
PARSEC_ISO_FILE = KOBE_DIR + "parsec/parsec_isochrones.dat.txt"
MASK_FILE = ISPEC_DIR + "input/linelists/CCF/HARPS_SOPHIE.K5.378_680nm/mask.lst"

# ── CARMENES instrument parameters ────────────────────────────────────────────
CARMENES_RESOLUTION = 94600   # CARMENES VIS resolving power R ~ 94600

# ── Wavelength range for synthesis ────────────────────────────────────────────
# Clean spectral windows across CARMENES VIS (562–920 nm), avoiding telluric bands:
#   O₂ B-band : 686–694 nm
#   H₂O       : 714–736  nm
#   O₂ A-band : 757–772 nm
#   H₂O       : 808–836 nm
#   H₂O       : 891–920 nm  (beyond this CARMENES VIS ends)
# Hα (656 nm) and Na I D (589 nm) are excluded via STRONG_LINES_FILE.
# RV via CCF uses only the range allowed by the loaded mask (≤ 680 nm).
WAVE_MIN_NM = 562.0   # nm  — blue limit of CARMENES VIS merged spectra
WAVE_MAX_NM = 890.0   # nm  — red limit before last H₂O band

# Clean windows [wave_base_nm, wave_top_nm] — union used for spectral synthesis.
# Pixels outside these windows but inside [WAVE_MIN_NM, WAVE_MAX_NM] are excluded.
# The telluric cleaner already masks individual contaminated pixels; these windows
# add a coarse guard against the deep saturated cores of each telluric band.
SYNTH_WINDOWS_NM = [
    (562.0, 686.0),   # Blue window  — richest in Fe I / Ca I lines for K dwarfs
    (695.0, 713.0),   # Window A     — between O₂ B-band and H₂O
    (737.0, 756.0),   # Window B     — between H₂O and O₂ A-band
    (773.0, 807.0),   # Window C     — between O₂ A-band and H₂O (Ca II IRT at 849/854/866 excluded below)
    (837.0, 860.0),   # Window D     — between H₂O bands; avoids Ca II IRT (849/854/866 nm)
]

# ── Pipeline flags ─────────────────────────────────────────────────────────────
CLEAN_TELLURIC     = True   # Remove telluric-contaminated pixels
                            # (recommended when extending to redder wavelengths)
NORMALIZE_CONTINUUM = True  # Fit spline continuum over iSpec continuum regions and normalize
                            # (recommended: P4 diagnostic shows ~6% blaze residual tilt)
CONTINUUM_MODE_CHOICES = ('baseline', 'rigid', 'flex', 'local', 'none')
CONTINUUM_MODE = 'baseline'
ESTIMATE_RV        = False  # Apply RV correction (shift to rest frame).
                            # CCF RV is ALWAYS measured as a diagnostic; this flag
                            # controls whether the correction is actually applied.
RV_WARNING_KMS     = 0.5   # Print a red warning if |RV_residual| > this value (km/s)
ESTIMATE_PARAMS    = True   # Run spectral synthesis
AUTO_RESCALE_ERRORS = True  # Inflate formal parameter errors by sqrt(chi2_red) after final fit
                            # (spectrum errors are NOT modified — only parameter uncertainties)
N_RENORM_ITERATIONS  = 2    # Iterative re-normalization passes using synthetic spectrum
RENORM_MAX_CHI2RED   = 3.0  # Skip re-normalization if chi²_red exceeds this value.
                             # High chi²_red means model inadequacy dominates the
                             # observed/synthetic ratio — spline would fit spectral
                             # residuals, not continuum errors, worsening the result.

# ── Parameter-fit controls ───────────────────────────────────────────────────
# logg is always computed physically from catalog quantities:
#   logg = logg☉ + log10(M_cat/M☉) − 2·log10(R/R☉)
#   R/R☉ = √(L_cat/L☉) · (T☉/Teff)²   [Stefan-Boltzmann]
# M_cat and L_cat are catalog constants; only Teff changes between seed and final.
# logg_init uses Teff_cat; logg_final uses Teff_fit after the synthesis.
MEASURE_VSINI_CCF  = True   # Independent vsini from CCF FWHM deconvolution (self-calibrated)

# ── Broadening parameter strategy (K5–M0 V; iSpec + turbospectrum) ────────────
# At R=94 600, vsini/vmac/vmic all lie at or below the instrumental
# resolution (FWHM ~3.2 km/s).  The fit cannot disentangle them — they
# collapse into a single effective broadening.  Strategy mirrors
# Marfil+22 / SteParSyn (Tabernero+22):
#
#   · vsini — ALWAYS hard-fixed.  Slow rotators (vsini_CCF ≤ 4 km/s):
#             flat anchor at VSINI_SLOW_FIXED_KMS = 2.0 km/s.
#             Fast rotators (vsini_CCF > 4 km/s): hard-fixed at vsini_CCF.
#             Never appears in the free-parameter list.
#   · vmac  — hard-fixed at 0.0 km/s.  vmac and vsini are too degenerate
#             to be separated in FGKM (Tabernero+22 Sec. 3.1); we follow
#             the SteParSyn convention of absorbing vmac into the
#             rotational kernel (i.e. vsini=2 km/s already includes vmac
#             in quadrature).  Eliminates the zone-boundary discontinuity
#             observed in our prior scheme (KOBE-016: ΔTeff=168 K).
#   · vmic  — hard-fixed at VMIC_COOL_KDWARF = 0.7 km/s for all targets.
#             Sensitivity test (this work, N=26): 77% of targets prefer
#             0.7 in chi2_red, 0% prefer 1.1; agreement with MARFIL
#             improves by 0.06 dex vs the prior 0.9 anchor.  Magic+2014
#             3D RHD predicts ξ ≈ 0.6–0.85 km/s in this regime.
#
# Net result: with vsini, vmac, vmic all anchored, the only free
# broadening-related parameter is removed from the fit.  Free
# parameters reduce to {Teff, [Fe/H]} (logg derived physically),
# matching the SteParSyn/MARFIL architecture.
VSINI_SLOW_FIXED_KMS    = 2.0       # km/s — flat anchor for slow rotators
                                    # (Marfil+22 KOBE preliminary convention).
                                    # Empirically absorbs vmac in quadrature.
VSINI_FAST_THRESHOLD_KMS = 4.0      # If vsini_CCF > threshold → hard-fix at
                                    # vsini_CCF (real rotation regime, e.g. KOBE-004
                                    # at 7.7 km/s).  Else hard-fix at VSINI_SLOW_FIXED_KMS.
                                    # vsini is NEVER a free parameter in this pipeline.
# vmic is hard-fixed at VMIC_COOL_KDWARF for ALL KOBE targets.
# Dutra-Ferreira (2016) is calibrated for 4509–6456 K; below that it
# extrapolates to 0.38–0.49 km/s and the optimiser rail-locks at the lower
# bound, producing non-informative errors and biasing [Fe/H] low.
# CARMENES at R~94 600 cannot disentangle vmic from [Fe/H] for cool K dwarfs
# — the bimodal vmic distribution in v02 (0.90 fixed vs 0.40–0.55 free,
# with no stars between 0.57–0.89) confirms the spectroscopic signal is
# insufficient. The adopted flat anchor is 0.7 km/s.
VMIC_COOL_KDWARF    = 0.7      # km/s — flat value for K5–K7 V.
                               # Convergent evidence (this work, N=26):
                               #   · 77% of targets prefer vmic=0.7 in chi2_red
                               #     (0% prefer 1.1); slope d[Fe/H]/dvmic = -0.11
                               #     dex/(km/s), uniform across SNR and Teff.
                               #   · vmic=0.7 reduces residual vs MARFIL [Fe/H]
                               #     by ~0.06 dex compared to 0.9.
                               #   · Magic+2014 (Stagger 3D RHD) predicts
                               #     ξ ≈ 0.6–0.85 km/s for K5–K7 V.
                               #   · Dutra-Ferreira (2016) extrapolates to
                               #     ~0.5 km/s here; 0.7 is the central choice.
                               # The previous 0.9 anchor (APOGEE/GES) extrapolated
                               # from Teff > 4500 K and is inadmissible for KOBE.

# ── Line regions source ──────────────────────────────────────────────────────
# LINELIST_SOURCE: selects the iSpec line-region list used for parameter fitting.
#
#   'GES'        — Gaia-ESO curated golden list (47000_GES), ~288 lines, 480–679 nm.
#                  log(gf) from laboratory measurements (Heiter+2021); preferred for
#                  reproducibility and scientific traceability.
#
#   'GES_MASKED' — GES list with 7 lines removed that systematically bias Teff high
#                  in the GBS validation test on 61 CygA (K5V):
#                    Ca I  616.13, 616.96, 643.91, 646.26 nm  (NLTE-sensitive)
#                    Ni I  616.34 nm
#                    Fe I  616.54, 618.02 nm  (blend / single-pixel window)
#                  These lines improve χ² at Teff=4540 K vs 4374 K reference,
#                  indicating they pull the fit to higher temperatures.
#                  File: kobe_ges_lines_kobe_masked.txt (282 lines).
#
#   'VALD'       — iSpec VALD3-based list (47000_VALD), ~276 lines, 480–679 nm.
#                  log(gf) from VALD3 (mix of theoretical + experimental).
#
# The pipeline presents a known systematic offset measured on the GBS calibrator
# 61 Cyg A (K5V, Heiter+15): see GBS validation test (--gbs-star 61CygA).
# This offset should be reported as a systematic uncertainty in publications,
# not corrected numerically (single calibrator, no cross-validation available).
LINELIST_SOURCE = 'GES_MASKED'  # default: 7 Teff-biasing lines removed (see ges_lines_kobe_masked.txt)

_GES_MASKED_FILE = KOBE_DIR + 'ges_lines_kobe_masked.txt'

# GENERATE_FULL_SYNTH_DIAGNOSTIC: if True, synthesize the full observed
# wavelength range (~3000 Å with all atomic + molecular lines) after the fit
# to produce the check_spectrum.pdf visual diagnostic.  Expensive (~5–10 min
# per star in K dwarfs due to TiO/CN).  Fit parameters and errors are NOT
# affected by this step; only the diagnostic PDF.  Disable for batch runs;
# enable when debugging a specific star.  When False, the plot falls back to
# the sparse line-region synthesis produced during the fit itself.
GENERATE_FULL_SYNTH_DIAGNOSTIC = True

# ── Strong-line exclusion windows (P7) ────────────────────────────────────────
# Line regions whose wave_peak falls inside any of these intervals [nm] are
# removed before synthesis.  Strong Balmer and alkali lines are poorly
# reproduced by 1D LTE models and inflate chi-square if included.
EXCLUDE_STRONG_LINES = True
EXCLUSION_WINDOWS_NM = [
    # Lines below WAVE_MIN_NM=562 nm are never reached but kept for reference:
    # (393.0, 397.0),   # Ca II H&K — outside CARMENES VIS range
    # (422.5, 423.5),   # Ca I 422.7 nm — outside range
    # (433.5, 436.5),   # Hγ 434.05 nm — outside range
    # (484.5, 488.5),   # Hβ 486.13 nm — outside range
    # (516.5, 518.0),   # Mg I b triplet — outside range
    (587.7, 590.9),   # Na I D 589.0/589.6 nm — strong NLTE doublet
    (652.0, 668.0),   # Hα 656.28 nm — strong Balmer line, 1D LTE unreliable
]

# ── Solar reference values (IAU 2015 / Prsa+16; vmic from Doyle+14) ───────────
SOLAR_TEFF  = 5777.0
SOLAR_LOGG  = 4.4374
SOLAR_MH    = 0.00
SOLAR_VMIC  = 1.07   # km/s

# ── Initial stellar parameters (used as starting point for synthesis) ──────────
# Centred on late-K dwarfs (K5–K7 V). Adjust per target if literature values known.
INITIAL_TEFF  = 4500.0
INITIAL_LOGG  = 4.50
INITIAL_MH    = 0.0

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(KOBE_DIR, 'output')
# Systematic uncertainty floor for [Fe/H] not captured by the formal fit error.
# Three independent contributions added in quadrature:
#   · σ_lineset    ≈ 0.07 dex  — MAD across 8 line-subset modes (this work, 6 GBS)
#   · σ_continuum  ≈ 0.04 dex  — spread baseline/rigid/flex/local (literature)
#   · σ_NLTE       ≈ 0.05 dex  — Fe I NLTE corrections in K-dwarfs (Lind+12)
# Combined: √(0.07² + 0.04² + 0.05²) = 0.095 dex ≈ 0.10 dex.
# Independent of SNR; floor added in quadrature to σ_formal for total uncertainty.
SYSTEMATIC_MH_FLOOR = 0.10   # dex

RESULTS_CSV = os.path.join(OUTPUT_DIR, 'kobe_results.csv')
LINE_SUBSET_RESULTS_CSV = os.path.join(OUTPUT_DIR, 'kobe_line_subset_results.csv')
CONTINUUM_STRESS_RESULTS_CSV = os.path.join(OUTPUT_DIR, 'kobe_continuum_stress_results.csv')

# Optional Fe-line diagnostic mode.
# Production uses 'all', which excludes Fe I weak lines by default.
# 'all_diag' preserves the unfiltered line list for A/B diagnostics.
# Depths are measured on the normalized observed spectrum inside each line window:
# weak < 0.08, medium 0.08-0.18, strong >= 0.18 in normalized flux units.
LINE_SUBSET_MODE = 'all'
LINE_DEPTH_WEAK_MAX = 0.08
LINE_DEPTH_MEDIUM_MAX = 0.18
EXCLUDE_WEAK_FE1_IN_PRODUCTION = True
EXCLUDE_PROBLEMATIC_FE1_MEDIUM_IN_PRODUCTION = True
PROBLEMATIC_FE1_MEDIUM_WAVES_NM = np.array([
    579.81710, 609.66640, 646.91920,
    630.15000, 631.58110, 675.27070,
    622.67340, 679.32580,
    563.82620, 570.15440, 570.70490,
    612.79060, 623.26400, 640.80170, 641.99490,
], dtype=float)
FE1_MEDIUM_WAVE_TOL_NM = 0.003
FE2_LINE_TOL_NM = 0.003
FE2_DIAGNOSTIC_LINES = (
    ('fe2_624756', 624.7557),
    ('fe2_636946', 636.9459),
    ('fe2_643268', 643.2676),
)
FE2_LINE_TEST_MODES = tuple(label for label, _ in FE2_DIAGNOSTIC_LINES)
LINE_SUBSET_CHOICES = (
    'all',
    'all_diag',
    'fe1',
    'fe2',
    *FE2_LINE_TEST_MODES,
    'fe1_weak',
    'fe1_medium',
    'fe1_strong',
    'fe1_weak_medium',
    'fe1_blue',
    'fe1_red',
)
LINE_SUBSET_TEST_MODES = (
    'all_diag',
    'fe1',
    'fe1_weak',
    'fe1_medium',
    'fe1_strong',
    'fe1_weak_medium',
    'fe1_blue',
    'fe1_red',
)
LAST_LINE_REGION_COUNT = np.nan

# ── Chromospheric activity ────────────────────────────────────────────────────
# Stars with log R'_HK above the Teff-dependent threshold are flagged ACTIVE.
# Ca II emission biases Teff and [M/H] in spectral synthesis.
#
# Threshold calibration: Astudillo-Defru et al. 2017 (A&A 600 A13), Eq. 3.
# They give the basal (minimum-activity) logR'HK as a function of B-V color.
# We convert B-V → Teff via Pecaut & Mamajek 2013 (updated Table 5, K/M dwarfs):
#   Teff = 8907 - 6590·(B-V) + 1467·(B-V)²   [valid 3600–5200 K, B-V 0.9–1.6]
# Inverted:  B-V ≈ 3.179 - 5.647e-4·Teff + 2.928e-8·Teff²  (numerical fit)
#
# Astudillo-Defru+17 Eq. 3 (basal):
#   logR'HK_basal = 1.131·(B-V)³ − 3.991·(B-V)² + 4.833·(B-V) − 6.601
# Active threshold = basal + 0.1  (0.1 dex margin above chromospheric floor).
#
# Fallback for Teff outside [3600, 5200] K: fixed value −4.50.
ACTIVITY_LOGRPHK_THRESHOLD = -4.7  # kept as fallback / legacy constant




def activity_threshold_for_teff(teff):
    """
    Return the logR'HK active/inactive threshold appropriate for the given Teff.

    Uses the Astudillo-Defru+2017 basal logR'HK calibration (Eq. 3) plus a
    0.1 dex margin.  Valid for late-K / early-M dwarfs (Teff 3600–5200 K).
    Outside that range falls back to ACTIVITY_LOGRPHK_THRESHOLD.

    Parameters
    ----------
    teff : float or None
        Effective temperature [K].

    Returns
    -------
    threshold : float
        logR'HK value above which the star is considered active.
    bv : float or None
        B−V color used in the calculation (None if fallback used).
    """
    if teff is None or not np.isfinite(teff) or not (3600 <= teff <= 5200):
        return ACTIVITY_LOGRPHK_THRESHOLD, None

    # B-V from Teff (Pecaut & Mamajek 2013 polynomial inversion, K/M dwarfs)
    bv = 3.179 - 5.647e-4 * teff + 2.928e-8 * teff**2

    # Astudillo-Defru+17 Eq. 3 — basal logR'HK
    logrhk_basal = (1.131 * bv**3
                    - 3.991 * bv**2
                    + 4.833 * bv
                    - 6.601)

    threshold = logrhk_basal + 0.1   # 0.1 dex above basal = active
    return threshold, bv

# ── KOBE catalog (VOSA luminosities + literature seeds) ──────────────────────
_KOBE_CATALOG_CACHE = None

def _load_kobe_catalog():
    """
    Read kobe_allstars.csv once and cache a dict keyed by KOBE_id.

    Extracted fields per target:
      Lbol, Lberr   — bolometric luminosity [L☉] from VOSA
      Teff_K, eTeff_K — literature Teff [K]
      logg, elogg   — literature logg [dex]
      Fe_H, eFe_H   — literature [Fe/H] [dex]

    Returns {} if the file is missing or unreadable.
    """
    global _KOBE_CATALOG_CACHE
    if _KOBE_CATALOG_CACHE is not None:
        return _KOBE_CATALOG_CACHE

    import csv

    _KOBE_CATALOG_CACHE = {}
    if not os.path.exists(KOBE_CATALOG_FILE):
        print(colored(f"    WARNING: KOBE catalog not found: {KOBE_CATALOG_FILE}", 'magenta'))
        return _KOBE_CATALOG_CACHE

    def _float(val):
        try:
            v = float(val)
            return v if np.isfinite(v) else np.nan
        except (ValueError, TypeError):
            return np.nan

    # kobe_allstars.csv contains duplicate column names (Teff_K, logg, Fe_H
    # appear once for v02 and once for v01 results).  csv.DictReader silently
    # keeps only the LAST value for each duplicate key, which is the v01
    # column — empty for stars not yet run in v01.  We therefore read seed
    # Teff from Teff_VOSA (unique, available for all 49 targets) and seed
    # [Fe/H] from Fe_H_marfil (unique; falls back to Simbad_Fe_H).
    with open(KOBE_CATALOG_FILE, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            kid = row.get('KOBE_id', '').strip()
            if not kid:
                continue
            # Seed Teff: Mamajek (V-K photometric) — unique column, always filled
            _teff_raw  = row.get('Teff_mamajek', '').strip()
            _eteff_raw = row.get('eTeff_mamajek', '').strip()
            # Seed [Fe/H]: MARFIL spectroscopic value (unique column); fall
            # back to Simbad_Fe_H, then to NaN (pipeline will use 0.0)
            _feh_raw = (row.get('Fe_H_marfil', '').strip()
                        or row.get('Simbad_Fe_H', '').strip())
            _KOBE_CATALOG_CACHE[kid] = {
                'Lbol'    : _float(row.get('Lbol',    '')),
                'Lberr'   : _float(row.get('Lberr',   '')),
                'Teff_K'  : _float(_teff_raw),
                'eTeff_K' : _float(_eteff_raw),
                'Fe_H'    : _float(_feh_raw),
                'eFe_H'   : _float(row.get('eFe_H_marfil', '')),
                'logRHK'  : _float(row.get('logRHK',  '')),
                'elogRHK' : _float(row.get('elogRHK', '')),
                'M_Msol'  : _float(row.get('M_Msol',  '')),
                'eM_Msol' : _float(row.get('eM_Msol', '')),
                'R_Rsol'  : _float(row.get('R_Rsol',  '')),
                'eR_Rsol' : _float(row.get('eR_Rsol', '')),
            }

    return _KOBE_CATALOG_CACHE


def get_vosa_luminosity(target_name):
    """
    Return (Lbol, Lberr) in L☉ for target_name from the KOBE catalog.

    Returns None if the target is not in the catalog or values are missing.
    """
    cat = _load_kobe_catalog()
    row = cat.get(target_name)
    if row is None:
        return None
    L, eL = row['Lbol'], row['Lberr']
    if not (np.isfinite(L) and L > 0 and np.isfinite(eL) and eL >= 0):
        return None
    return (L, eL)


def get_catalog_seeds(target_name):
    """
    Return initial parameter seeds for target_name from the KOBE catalog.

    logg is NOT computed here.  Use logg_from_catalog(M_cat, L_cat, teff)
    after loading luminosity with get_vosa_luminosity().

    Returns dict with keys teff, mh, M_cat, eM_cat, or None if the
    target is not found or required fields (Teff, [Fe/H]) are missing.
    M_cat / eM_cat are np.nan when not available in the catalog.
    """
    cat = _load_kobe_catalog()
    row = cat.get(target_name)
    if row is None:
        # Case-insensitive fallback (e.g. "61cyga" → "61CygA", "kobe-056" → "KOBE-056")
        _lower = target_name.lower()
        _match = next((k for k in cat if k.lower() == _lower), None)
        if _match is not None:
            row = cat[_match]
    if row is None:
        return None
    teff = row['Teff_K']
    feh  = row['Fe_H']
    if not np.isfinite(teff):
        return None
    if not np.isfinite(feh):
        feh = 0.0  # no catalog value — use solar as seed; optimizer fits it freely
    M_cat  = row.get('M_Msol',  np.nan)
    eM_cat = row.get('eM_Msol', np.nan)
    return {'teff': teff, 'mh': feh, 'M_cat': M_cat, 'eM_cat': eM_cat}


def get_activity_flag(target_name, teff=None):
    """
    Return (logRHK, is_active, threshold, bv) for target_name.

    Uses a Teff-dependent logR'HK threshold (Astudillo-Defru+2017 basal + 0.1
    dex margin) so that the naturally higher chromospheric flux of K7/M0 dwarfs
    does not produce spurious ACTIVE flags.

    Parameters
    ----------
    target_name : str
    teff : float or None
        Effective temperature [K] for the Teff-dependent threshold.
        If None, falls back to the fixed ACTIVITY_LOGRPHK_THRESHOLD.

    Returns
    -------
    logRHK    : float or None
    is_active : bool
    threshold : float   — the threshold used
    bv        : float or None — B-V color used (None if fallback)
    """
    cat = _load_kobe_catalog()
    row = cat.get(target_name)
    if row is None:
        return (None, False, ACTIVITY_LOGRPHK_THRESHOLD, None)
    val = row.get('logRHK', np.nan)
    if not np.isfinite(val):
        return (None, False, ACTIVITY_LOGRPHK_THRESHOLD, None)
    threshold, bv = activity_threshold_for_teff(teff)
    return (val, val > threshold, threshold, bv)


# ── Quality flags ────────────────────────────────────────────────────────────

def compute_quality_flags(snr, rv_residual_kms, chi2_red,
                          vsini_synth, vsini_ccf, is_active,
                          teff=None, vmic=None, vmac=None,
                          is_active_spectral=None,
                          snr_window_a=None):
    """
    Evaluate spectrum/fit quality and return a list of human-readable flags.

    Criteria
    --------
    - S/N (global) < 30      → 'LOW_SNR'
        Note: the global SNR is estimated via MAD on a 201-pixel window
        ([_estimate_local_noise]) and is biased LOW in line-rich K-dwarf
        spectra because absorption lines contribute to the residuals
        (lines look like 'noise' to the MAD).  Threshold 30 reflects the
        operational SNR floor for [Fe/H] determination in CARMENES late-K
        masters (Marfil+22).  Lower than 30 is genuinely concerning.
    - S/N (window A, 562-686 nm) < 25  → 'LOW_SNR_FEI'
        Critical for [Fe/H] reliability since most Fe I diagnostics live
        in this band.  Independent of the global flag.
    - |RV_residual| > 1 km/s → 'RV_OFFSET'
    - chi2_red > 3          → 'BAD_FIT'
    - |vsini_synth − vsini_ccf| / max > 0.30  → 'VSINI_DISCREPANT'
    - is_active_spectral (Ca II IRT filling)   → 'ACTIVE'   [preferred]
    - is_active (catalog log R'HK)             → 'ACTIVE'   [fallback]
    - vmic and vmac are fixed pipeline constants/anchors, so no range flag is
      emitted for them.

    The spectral Ca II IRT flag (is_active_spectral) takes priority over the
    catalog-based flag (is_active) when available.  This avoids spurious ACTIVE
    labels from epoch-mismatched or calibration-uncertain catalog values.

    Returns list of flag strings (empty if all OK).
    """
    flags = []
    if snr is not None and snr < 30:
        flags.append('LOW_SNR')
    if snr_window_a is not None and snr_window_a < 25:
        flags.append('LOW_SNR_FEI')
    if rv_residual_kms is not None and abs(rv_residual_kms) > 1.0:
        flags.append('RV_OFFSET')
    if chi2_red is not None and chi2_red > 3.0:
        flags.append('BAD_FIT')
    if (vsini_synth is not None and vsini_ccf is not None
            and max(vsini_synth, vsini_ccf) > 0.5):
        disc = abs(vsini_synth - vsini_ccf) / max(vsini_synth, vsini_ccf)
        if disc > 0.30:
            flags.append('VSINI_DISCREPANT')

    # Activity flag: spectral (Ca II IRT) takes priority over catalog
    if is_active_spectral is not None:
        if is_active_spectral:
            flags.append('ACTIVE')
    elif is_active:
        flags.append('ACTIVE')

    return flags


# ── Cumulative results CSV ───────────────────────────────────────────────────

def append_results_csv(target, params, errors, quality_flags,
                       snr=None, rv_residual=None, chi2_red=None,
                       radius=None, e_radius=None):
    """
    Append one row to kobe_results.csv. Creates the file with header if absent.

    Column order: target | Teff_K eTeff_K | logg elogg |
                  Radius_Rsun eRadius_Rsun | Fe_H eFe_H |
                  vsini evsini | vmic vmac |
                  SNR RV_residual_kms chi2_red quality_flags
    """
    import csv

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_csv = RESULTS_CSV
    include_subset = LINE_SUBSET_MODE != 'all'
    include_continuum = CONTINUUM_MODE != 'baseline'

    fieldnames = [
        'target',
        'Teff_K', 'eTeff_K',
        'logg',   'elogg',
        'Radius_Rsun', 'eRadius_Rsun',
        'Fe_H',   'eFe_H',   'eFe_H_total',
        'vsini',  'evsini',
        'vmic',   'vmac',
        'SNR', 'RV_residual_kms', 'chi2_red', 'quality_flags',
    ]
    if include_subset:
        results_csv = LINE_SUBSET_RESULTS_CSV
        fieldnames.insert(1, 'LineSubset')
        fieldnames.insert(2, 'NLines')
    elif include_continuum:
        results_csv = CONTINUUM_STRESS_RESULTS_CSV
        fieldnames.insert(1, 'ContinuumMode')

    _radius_val = radius if (radius is not None and np.isfinite(radius)) else np.nan
    _e_radius_val = e_radius if (e_radius is not None and np.isfinite(e_radius)) else np.nan

    row = {
        'target'          : target,
        'Teff_K'          : f"{params.get('teff', np.nan):.2f}",
        'eTeff_K'         : f"{errors.get('teff', np.nan):.2f}",
        'logg'            : f"{params.get('logg', np.nan):.4f}",
        'elogg'           : f"{errors.get('logg', np.nan):.4f}",
        'Radius_Rsun'     : f"{_radius_val:.4f}" if np.isfinite(_radius_val) else '',
        'eRadius_Rsun'    : f"{_e_radius_val:.4f}" if np.isfinite(_e_radius_val) else '',
        'Fe_H'            : f"{params.get('MH', np.nan):.4f}",
        'eFe_H'            : f"{errors.get('MH', np.nan):.4f}",
        'eFe_H_total'      : (
            f"{np.sqrt(errors.get('MH', np.nan)**2 + SYSTEMATIC_MH_FLOOR**2):.4f}"
            if np.isfinite(errors.get('MH', np.nan)) else ''),
        'vsini'           : f"{params.get('vsini', np.nan):.4f}",
        'evsini'          : f"{errors.get('vsini', np.nan):.4f}",
        'vmic'            : f"{params.get('vmic', np.nan):.4f}",
        'vmac'            : f"{params.get('vmac', np.nan):.4f}",
        'SNR'             : f"{snr:.1f}" if snr is not None else '',
        'RV_residual_kms' : f"{rv_residual:.4f}" if rv_residual is not None else '',
        'chi2_red'        : f"{chi2_red:.4f}" if chi2_red is not None else '',
        'quality_flags'   : '|'.join(quality_flags) if quality_flags else 'OK',
    }
    if include_subset:
        row['LineSubset'] = LINE_SUBSET_MODE
        _n_lines = LAST_LINE_REGION_COUNT
        row['NLines'] = (
            f"{int(_n_lines)}"
            if np.isfinite(_n_lines) and _n_lines >= 0
            else ''
        )
    elif include_continuum:
        row['ContinuumMode'] = CONTINUUM_MODE

    import fcntl as _fcntl
    with open(results_csv, 'a+', newline='') as fh:
        _fcntl.flock(fh, _fcntl.LOCK_EX)
        try:
            needs_header = (os.fstat(fh.fileno()).st_size == 0)
            if not needs_header:
                fh.seek(0)
                reader = csv.DictReader(fh)
                if reader.fieldnames != fieldnames:
                    existing_rows = [
                        {key: old_row.get(key, '') for key in fieldnames}
                        for old_row in reader
                    ]
                    fh.seek(0)
                    fh.truncate()
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                    needs_header = False
                    print(colored(
                        "  Existing results CSV header updated to current schema.",
                        'yellow'))
                fh.seek(0, os.SEEK_END)

            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        finally:
            _fcntl.flock(fh, _fcntl.LOCK_UN)

    print(colored(f"  Results appended to {results_csv}", 'green'))


# ── PARSEC grid interpolator (module-level cache) ─────────────────────────────
_PARSEC_GRID_CACHE = None

def _build_parsec_logg_grid():
    """
    Build a 2D interpolator (Teff, [M/H]) → logg from PARSEC v1.2S isochrones.

    Strategy:
      - Read parsec_isochrones.dat.txt (CMD 3.9 output, ~55k rows)
      - Filter: label=1 (MS phase only), logg > 4.0 (exclude turnoff/SGB),
        logAge ≤ 9.9 (ages 1–8 Gyr, avoid severe turnoff at 10 Gyr)
      - For each (MH, logAge) combination, interpolate the MS track logTe → logg
        onto a common Teff grid
      - At each (Teff_grid, MH_grid) node, take the median logg across all ages
        (marginalises over the unknown stellar age)
      - Build a RegularGridInterpolator for fast (Teff, MH) → logg queries

    Returns
    -------
    interp : scipy.interpolate.RegularGridInterpolator or None
        Returns None if the file is missing or the grid cannot be built.
    """
    global _PARSEC_GRID_CACHE
    if _PARSEC_GRID_CACHE is not None:
        return _PARSEC_GRID_CACHE

    from scipy.interpolate import interp1d, RegularGridInterpolator

    if not os.path.exists(PARSEC_ISO_FILE):
        print(colored(f"    WARNING: PARSEC isochrone file not found: {PARSEC_ISO_FILE}", 'magenta'))
        return None

    # ── Read the CMD output (comment lines start with '#') ──
    data = np.genfromtxt(PARSEC_ISO_FILE, comments='#')
    # Columns:  0=Zini  1=MH  2=logAge  3=Mini  4=int_IMF  5=Mass
    #           6=logL  7=logTe  8=logg  9=label  ...
    col_mh     = data[:, 1]
    col_age    = data[:, 2]
    col_logTe  = data[:, 7]
    col_logg   = data[:, 8]
    col_label  = data[:, 9].astype(int)

    # ── Filter: MS dwarfs only, avoid turnoff contamination ──
    sel = ((col_label == 1) &          # MS phase
           (col_logg > 4.0) &          # exclude evolved/turnoff stars
           (col_age <= 9.9))           # ages 1–7.9 Gyr (avoid 10 Gyr turnoff)

    if sel.sum() < 100:
        print(colored("    WARNING: too few PARSEC MS points after filtering.", 'magenta'))
        return None

    mh_vals  = np.unique(col_mh[sel])
    age_vals = np.unique(col_age[sel])

    # ── Common Teff grid in logTe space ──
    # Range covers ~4170 K (logTe=3.62) to ~6610 K (logTe=3.82)
    n_teff = 200
    logTe_grid = np.linspace(3.62, 3.82, n_teff)

    # ── For each MH, median logg across ages at each Teff node ──
    logg_median = np.full((n_teff, len(mh_vals)), np.nan)

    for j, mh_val in enumerate(mh_vals):
        logg_all_ages = []

        for age_val in age_vals:
            mask = sel & (col_mh == mh_val) & (col_age == age_val)
            if mask.sum() < 3:
                continue

            lt = col_logTe[mask]
            lg = col_logg[mask]

            # Sort by logTe and remove duplicates
            order = np.argsort(lt)
            lt, lg = lt[order], lg[order]
            _, uniq_idx = np.unique(lt, return_index=True)
            lt, lg = lt[uniq_idx], lg[uniq_idx]

            if len(lt) < 3:
                continue

            # Interpolate this age's MS track onto the common grid
            f = interp1d(lt, lg, kind='linear', bounds_error=False, fill_value=np.nan)
            logg_all_ages.append(f(logTe_grid))

        if logg_all_ages:
            arr = np.array(logg_all_ages)  # shape (n_ages, n_teff)
            logg_median[:, j] = np.nanmedian(arr, axis=0)

    # ── Build the 2D interpolator ──
    teff_grid = 10.0 ** logTe_grid

    # Check for sufficient coverage
    finite_frac = np.isfinite(logg_median).sum() / logg_median.size
    if finite_frac < 0.3:
        print(colored(f"    WARNING: PARSEC grid has only {finite_frac:.0%} coverage. "
                      f"Falling back to polynomial.", 'magenta'))
        return None

    # Fill NaN edges via nearest-neighbour extrapolation (for border nodes)
    for j in range(len(mh_vals)):
        col = logg_median[:, j]
        finite = np.isfinite(col)
        if finite.any() and not finite.all():
            first_valid = np.argmax(finite)
            last_valid  = len(col) - 1 - np.argmax(finite[::-1])
            col[:first_valid] = col[first_valid]
            col[last_valid+1:] = col[last_valid]
            logg_median[:, j] = col

    interp = RegularGridInterpolator(
        (teff_grid, mh_vals),
        logg_median,
        method='linear',
        bounds_error=False,
        fill_value=np.nan,   # returns NaN outside grid → triggers fallback
    )

    _PARSEC_GRID_CACHE = interp
    return interp


def _logg_from_parsec_grid(teff, mh):
    """
    Query the PARSEC 2D grid interpolator for logg(Teff, [M/H]).

    Returns logg (float) or None if interpolation fails or is out of range.
    """
    interp = _build_parsec_logg_grid()
    if interp is None:
        return None

    try:
        logg = float(interp((teff, mh)))
    except Exception:
        return None

    if not np.isfinite(logg):
        return None

    return float(np.clip(logg, 3.5, 5.0))


# ── Physical logg from luminosity (Eker+2018 MLR + Stefan-Boltzmann) ────────

# Solar constants for logg calculation
LOGG_SUN    = 4.4374       # log g☉ [cgs]
TEFF_SUN    = 5772.0       # T_eff☉ [K]

def _mass_from_luminosity_eker2018(lum, e_lum=0.0):
    """
    Stellar mass from luminosity using the piecewise MLR of Eker et al. (2018),
    MNRAS, 479, 5491, Table 1.

    The six-piece classical MLR is:  log L = a · log M + b  ±  σ_MLR
    This function inverts the relevant piece for FGK MS stars.

    Parameters
    ----------
    lum   : float — luminosity in solar units [L☉]
    e_lum : float — 1σ uncertainty on luminosity [L☉]

    Returns
    -------
    mass, e_mass : floats — mass [M☉] and its 1σ uncertainty
    """
    # Eker+2018 Table 1 — piecewise MLR coefficients
    # Domain            M range         a       b        σ_MLR
    _EKER_MLR = [
        (0.179, 0.45,   2.028, -0.976,  0.076),  # ultra low mass
        (0.45,  0.72,   4.572, -0.102,  0.109),  # very low mass
        (0.72,  1.05,   5.743, -0.007,  0.129),  # low mass
        (1.05,  2.40,   4.329,  0.010,  0.140),  # intermediate mass
        (2.40,  7.00,   3.967,  0.093,  0.165),  # high mass
        (7.00, 31.00,   2.865,  1.105,  0.152),  # very high mass
    ]

    logL = np.log10(lum)

    # Find the correct domain by inverting each piece and checking mass consistency
    best_mass = None
    best_a = None
    best_sigma_mlr = None

    for m_lo, m_hi, a, b, sigma_mlr in _EKER_MLR:
        logM = (logL - b) / a
        M = 10.0 ** logM
        if m_lo <= M <= m_hi * 1.05:  # 5% tolerance at boundaries
            best_mass = M
            best_a = a
            best_sigma_mlr = sigma_mlr
            break

    if best_mass is None:
        # Fallback: use low-mass or intermediate-mass piece depending on L
        if logL < 0.0:
            best_a, b, best_sigma_mlr = 5.743, -0.007, 0.129
        else:
            best_a, b, best_sigma_mlr = 4.329,  0.010, 0.140
        logM = (logL - b) / best_a
        best_mass = 10.0 ** logM
        print(colored(f"    WARNING: L={lum:.3f} L☉ outside clean MLR domain, "
                      f"using fallback (M={best_mass:.4f} M☉)", 'magenta'))

    # Error propagation:  log M = (log L - b) / a
    # σ²_logM = (σ_logL / a)² + (σ_MLR / a)²
    sigma_logL = (e_lum / (lum * np.log(10))) if (e_lum > 0 and lum > 0) else 0.0
    sigma_logM = np.sqrt((sigma_logL / best_a) ** 2 +
                         (best_sigma_mlr / best_a) ** 2)
    e_mass = best_mass * np.log(10) * sigma_logM  # δM = M · ln(10) · σ_logM

    return float(best_mass), float(e_mass)


def logg_physical(lum, e_lum, teff, e_teff):
    """
    Surface gravity from luminosity + Teff via Eker+2018 MLR and Stefan-Boltzmann.

    Physics:
        R² = (L/L☉) · (T☉/T)⁴           [Stefan-Boltzmann in solar units]
        M  = MLR(L)                       [Eker+2018 piecewise]
        logg = logg☉ + log(M/M☉) − 2·log(R/R☉)
             = logg☉ + log(M) − log(L) + 4·log(T/T☉)

    Error propagation (analytic, accounting for M-L correlation):
        logg = logg☉ − b/a + (1/a − 1)·logL + 4·log(T/T☉)
        σ²_logg = (1/a − 1)²·σ²_logL + 16·σ²_logT + (σ_MLR/a)²

    Parameters
    ----------
    lum    : float — luminosity [L☉]
    e_lum  : float — 1σ uncertainty [L☉]
    teff   : float — effective temperature [K]
    e_teff : float — 1σ uncertainty [K]

    Returns
    -------
    logg, e_logg, mass, e_mass, radius, e_radius : floats
    """
    # Mass from Eker+2018
    mass, e_mass = _mass_from_luminosity_eker2018(lum, e_lum)

    # Radius from Stefan-Boltzmann: R/R☉ = √(L/L☉) · (T☉/T)²
    radius = np.sqrt(lum) * (TEFF_SUN / teff) ** 2
    # σ_R/R = √[ (σ_L/(2L))² + (2·σ_T/T)² ]
    e_radius = radius * np.sqrt((e_lum / (2.0 * lum)) ** 2 +
                                (2.0 * e_teff / teff) ** 2) if (e_lum > 0 or e_teff > 0) else 0.0

    # logg
    logg = LOGG_SUN + np.log10(mass) - 2.0 * np.log10(radius)

    # Error propagation (analytic)
    # Identify which Eker piece was used (re-derive a and σ_MLR)
    _, e_mass_check = _mass_from_luminosity_eker2018(lum, 0.0)
    # Get a and σ_MLR from the mass value
    _EKER_MLR = [
        (0.179, 0.45,  2.028, -0.976, 0.076),
        (0.45,  0.72,  4.572, -0.102, 0.109),
        (0.72,  1.05,  5.743, -0.007, 0.129),
        (1.05,  2.40,  4.329,  0.010, 0.140),
        (2.40,  7.00,  3.967,  0.093, 0.165),
        (7.00, 31.00,  2.865,  1.105, 0.152),
    ]
    a_mlr, sigma_mlr = 5.743, 0.129  # default low mass
    for m_lo, m_hi, a, b, sig in _EKER_MLR:
        if m_lo <= mass <= m_hi * 1.05:
            a_mlr, sigma_mlr = a, sig
            break

    sigma_logL = (e_lum / (lum * np.log(10))) if (e_lum > 0 and lum > 0) else 0.0
    sigma_logT = (e_teff / (teff * np.log(10))) if (e_teff > 0 and teff > 0) else 0.0

    e_logg = np.sqrt(
        (1.0 / a_mlr - 1.0) ** 2 * sigma_logL ** 2 +
        16.0 * sigma_logT ** 2 +
        (sigma_mlr / a_mlr) ** 2
    )

    return (float(np.clip(logg, 3.0, 5.5)), float(e_logg),
            float(mass), float(e_mass),
            float(radius), float(e_radius))


def logg_from_catalog(M_cat, L_cat, teff, e_teff=0.0, e_L=0.0):
    """
    Surface gravity from catalog mass and Stefan-Boltzmann radius.

    R/R☉  = √(L_cat/L☉) · (T☉/teff)²
    logg   = logg☉ + log10(M_cat/M☉) − 2·log10(R/R☉)

    Error propagation (M_cat assumed exact; uncertainty from L and Teff):
        logg = logg☉ + log10(M) − log10(L) + 4·log10(T/T☉)
        σ_logg = √[ (σ_L / (L·ln10))² + (4·σ_T / (T·ln10))² ]

    Returns
    -------
    logg    : float — clipped to [3.0, 5.5]
    e_logg  : float — propagated 1σ uncertainty
    radius  : float — derived R/R☉
    """
    import math
    radius = math.sqrt(L_cat) * (TEFF_SUN / teff) ** 2
    logg   = float(np.clip(
        LOGG_SUN + math.log10(M_cat) - 2.0 * math.log10(radius),
        3.0, 5.5))
    sig_logL = (e_L   / (L_cat * math.log(10))) if (e_L   > 0 and L_cat > 0) else 0.0
    sig_logT = (e_teff / (teff  * math.log(10))) if (e_teff > 0 and teff  > 0) else 0.0
    e_logg   = math.sqrt(sig_logL ** 2 + (4.0 * sig_logT) ** 2)
    return logg, e_logg, radius


# ── Mass and age estimation for evolved stars (logg_mode='free') ─────────

def mass_from_parsec_isochrones(teff, e_teff, lum, e_lum, mh=0.0,
                                logg=None):
    """
    Interpolate stellar mass and age from PARSEC v1.2S isochrones.

    For each isochrone (logAge, [M/H]_nearest), find the MS point closest
    to (Teff, L) in the HR diagram and read off the mass.  The best-fit
    age is the isochrone that minimises the distance; the uncertainty
    spans the range of ages whose MS passes within the error box.

    For evolved stars (logg < 3.8), labels 1-3 (MS+SGB+RGB) are used;
    for MS stars, only label=1 to avoid SGB contamination biasing the
    age estimate towards younger isochrones.

    Returns
    -------
    mass, e_mass : float — mass [M☉] and half-range of compatible masses
    age_gyr, age_lo, age_hi : float — best age and range [Gyr]
    """
    if not os.path.exists(PARSEC_ISO_FILE):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    data = np.genfromtxt(PARSEC_ISO_FILE, comments='#')
    col_mh    = data[:, 1]
    col_age   = data[:, 2]   # logAge
    col_mini  = data[:, 3]
    col_mass  = data[:, 5]   # current mass
    col_logL  = data[:, 6]
    col_logTe = data[:, 7]
    col_label = data[:, 9].astype(int)

    # Select nearest [M/H]
    mh_vals = np.unique(col_mh)
    mh_near = mh_vals[np.argmin(np.abs(mh_vals - mh))]

    # Age grid: logAge 8.0 to 10.15 (100 Myr to 14 Gyr)
    age_vals = np.unique(col_age)
    age_vals = age_vals[(age_vals >= 8.0) & (age_vals <= 10.15)]

    logL_target = np.log10(lum)
    logTe_target = np.log10(teff)

    # Normalisation scales for distance metric
    # Floors account for systematic uncertainties in isochrone physics
    # (opacities, convective overshooting) which limit the positioning
    # of theoretical isochrones to ~0.01 dex in logTe and ~0.03 dex in logL.
    sigma_logTe = max(e_teff / (teff * np.log(10)), 0.01) if e_teff > 0 else 0.01
    sigma_logL  = max(e_lum / (lum * np.log(10)), 0.03) if e_lum > 0 else 0.05

    # Evolutionary phase filter: MS-only for dwarfs, MS+SGB+RGB for giants
    evolved = (logg is not None and logg < 3.8)
    max_label = 3 if evolved else 1

    best_dist = np.inf
    best_mass = np.nan
    best_age  = np.nan
    compatible_masses = []
    compatible_ages   = []

    for logAge in age_vals:
        sel = ((col_mh == mh_near) &
               (col_age == logAge) &
               (col_label >= 1) & (col_label <= max_label))
        if sel.sum() < 5:
            continue

        logTe_iso = col_logTe[sel]
        logL_iso  = col_logL[sel]
        mass_iso  = col_mass[sel]

        # Normalised distance from target to each isochrone point
        d_logTe = (logTe_iso - logTe_target) / sigma_logTe
        d_logL  = (logL_iso - logL_target) / sigma_logL
        dist = np.sqrt(d_logTe**2 + d_logL**2)

        idx_min = np.argmin(dist)
        d_min = dist[idx_min]

        if d_min < best_dist:
            best_dist = d_min
            best_mass = mass_iso[idx_min]
            best_age  = 10.0**(logAge - 9.0)  # Gyr

        # "Compatible" = passes within 2σ error ellipse
        if d_min < 2.0:
            compatible_masses.append(mass_iso[idx_min])
            compatible_ages.append(10.0**(logAge - 9.0))

    if len(compatible_masses) >= 2:
        e_mass = (max(compatible_masses) - min(compatible_masses)) / 2.0
        age_lo = min(compatible_ages)
        age_hi = max(compatible_ages)
    else:
        e_mass = 0.1 * best_mass if np.isfinite(best_mass) else np.nan
        age_lo = best_age
        age_hi = best_age

    # Enforce minimum age uncertainty: at least ±20% or ±0.5 Gyr
    if np.isfinite(best_age) and age_lo == age_hi:
        age_margin = max(0.2 * best_age, 0.5)
        age_lo = max(0.1, best_age - age_margin)
        age_hi = best_age + age_margin

    return float(best_mass), float(e_mass), float(best_age), float(age_lo), float(age_hi)


def mass_from_logg_radius(logg, e_logg, radius, e_radius):
    """
    Stellar mass from spectroscopic logg and Stefan-Boltzmann radius.

    M/M☉ = 10^(logg - logg☉) · (R/R☉)²

    Parameters
    ----------
    logg, e_logg     : float — spectroscopic logg and its 1σ error
    radius, e_radius : float — radius [R☉] from Stefan-Boltzmann and its 1σ error

    Returns
    -------
    mass, e_mass : float — mass [M☉] and 1σ uncertainty
    """
    mass = 10.0**(logg - LOGG_SUN) * radius**2

    # Error propagation:  M = 10^(g-g☉) · R²
    # σ_M/M = √[ (ln10 · σ_logg)² + (2 · σ_R/R)² ]
    rel_err = np.sqrt((np.log(10) * e_logg)**2 +
                      (2.0 * e_radius / radius)**2) if radius > 0 else np.nan
    e_mass = mass * rel_err

    return float(mass), float(e_mass)


def age_from_parsec_isochrones(teff, e_teff, lum, e_lum, mh=0.0, logg=None):
    """
    Estimate stellar age from position in PARSEC HR diagram.

    Wrapper: calls mass_from_parsec_isochrones and returns only age info.

    Returns
    -------
    age_gyr, age_lo, age_hi : float — best-fit age and compatible range [Gyr]
    """
    _, _, age_gyr, age_lo, age_hi = mass_from_parsec_isochrones(
        teff, e_teff, lum, e_lum, mh=mh, logg=logg)
    return float(age_gyr), float(age_lo), float(age_hi)


def vmic_zone_strategy(teff, logg=4.6):
    """
    vmic strategy for KOBE late-K dwarfs: hard-fixed at VMIC_COOL_KDWARF
    for all targets regardless of Teff.

    Rationale: CARMENES at R~94 600 cannot disentangle vmic from [Fe/H] for
    cool K dwarfs (3800–4600 K).  The adopted flat anchor is 0.7 km/s.

    Returns
    -------
    centre   : float — fixed value [km/s]  (= VMIC_COOL_KDWARF)
    lo, hi   : float — hard bounds (equal to centre — not free)
    is_fixed : bool  — always True
    """
    v = VMIC_COOL_KDWARF
    return v, v, v, True


def vmac_zone_anchor(teff):
    """
    vmac anchor: always 0.0 km/s (absorbed into vsini).

    Rationale (Tabernero+2022, SteParSyn): in FGKM stars at R ~ 10⁵, vmac
    and vsini are too degenerate to be separated.  SteParSyn/MARFIL combine
    both into a single rotation kernel (V_broad).  We follow the same
    convention: vmac is set to zero and the empirical vsini value (2 km/s
    for slow rotators, vsini_CCF for fast rotators) absorbs the residual
    macroturbulent broadening in quadrature.

    The previous zone scheme (0.00 / 0.50 / 0.75 km/s by Teff) introduced
    discontinuities in the chi² landscape near the zone boundaries.  In
    the vmic sensitivity test (this work, N=26) two stars (KOBE-016,
    KOBE-020) showed Teff jumps of 168 K and 25 K respectively when the
    fit crossed a zone threshold — pure artefact.

    The teff argument is retained for backwards compatibility with callers
    but is unused.

    Returns
    -------
    float — 0.0 km/s
    """
    return 0.00
def build_initial_parameters(teff, logg, mh):
    """
    Build the initial parameter dictionary used by iSpec model_spectrum.

    vmic: fixed cool-K dwarf empirical value.
    vmac: hard-fixed at 0.0 km/s (not in free-params list).
    See vmic_zone_strategy() and vmac_zone_anchor() for anchors.
    """
    vmic_c, _, _, _ = vmic_zone_strategy(float(teff), float(logg))
    vmac_c = vmac_zone_anchor(float(teff))
    return {
        "teff"                 : float(teff),
        "logg"                 : float(logg),
        "MH"                   : float(mh),
        "alpha"                : ispec.determine_abundance_enchancements(float(mh)),
        "vmic"                 : vmic_c,
        "vmac"                 : vmac_c,
        "vsini"                : 2.0,
        "limb_darkening_coeff" : 0.6,
        "R"                    : CARMENES_RESOLUTION,
        "vrad"                 : 0.0,
    }


INITIAL_PARAMETERS = build_initial_parameters(INITIAL_TEFF, INITIAL_LOGG, INITIAL_MH)

# Parameters to fit — rebuilt after CLI parsing in __main__.
# Strategy for K5–M0 V: vmic fixed, vmac hard-fixed at 0.0 km/s,
# vsini hard-fixed from the CCF strategy.
FREE_PARAMS = ["teff", "MH", "vsini"]  # placeholder; overwritten below

# Per-target globals — set in __main__ (single-target) or in _synthesis_worker (batch).
# Declared here so that `main()` can always resolve them as module-level names.
M_CAT    = float('nan')
eM_CAT   = float('nan')
L_CAT    = float('nan')
eL_CAT   = float('nan')
LUMINOSITY = None

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr_in_window(spectrum, wave_min=None, wave_max=None):
    """
    Compute S/N as median(flux) / median(err) either globally or in a window.
    Wavelengths are in nm.
    """
    if wave_min is None or wave_max is None:
        flux = spectrum['flux']
        err = spectrum['err']
    else:
        mask = (spectrum['waveobs'] >= wave_min) & (spectrum['waveobs'] <= wave_max)
        if mask.sum() == 0:
            return np.nan
        flux = spectrum['flux'][mask]
        err = spectrum['err'][mask]

    med_flux = np.nanmedian(flux)
    med_err = np.nanmedian(err)
    if (not np.isfinite(med_flux)) or (not np.isfinite(med_err)) or med_err <= 0:
        return np.nan
    return med_flux / med_err


def print_snr_summary(spectrum, title="SNR summary"):
    """
    Print S/N diagnostics for the full spectrum, the blue region, and 550-600 nm.
    """
    snr_global = compute_snr_in_window(spectrum)
    snr_blue = compute_snr_in_window(spectrum, 400.0, 500.0)
    snr_550_600 = compute_snr_in_window(spectrum, 550.0, 600.0)

    print(colored(f"\n{'='*60}", 'blue'))
    print(colored(f"  {title}", 'blue'))
    print(colored(f"{'='*60}", 'blue'))
    print(f"  Global median S/N : {snr_global:.1f}")
    print(f"  Blue S/N (400-500 nm) : {snr_blue:.1f}")
    print(f"  S/N (550-600 nm) : {snr_550_600:.1f}")
    print(colored(f"{'='*60}\n", 'blue'))

def diagnostic_check_rv(spectrum, target_name='target', output_dir='.'):
    """
    Temporary diagnostic function to understand RV issues.
    Checks the position of strong absorption lines (Hα, Na I D, etc.)
    to verify if the spectrum is truly in rest frame or misaligned.
    """
    import matplotlib.pyplot as plt
    
    # Expected rest wavelengths (nm)
    rest_lines = {
        'Hα':         656.28,
        'Na I D1':    588.995,
        'Na I D2':    589.592,
        'Hβ':         486.13,
        'Hγ':         434.05,
    }
    
    print(colored(f"\n{'='*60}", 'blue'))
    print(colored(f"  RV DIAGNOSTIC CHECK", 'blue'))
    print(colored(f"{'='*60}", 'blue'))
    
    # Find absorption minima near known lines.
    # Hα uses a tight window (±0.3 nm) to avoid picking up adjacent features,
    # especially in active stars where the core is filled by chromospheric emission.
    search_radius = {'Hα': 0.3, 'Na I D1': 0.4, 'Na I D2': 0.4, 'Hγ': 0.5}
    # Depth threshold above which Hα core is considered emission-filled / unreliable
    HALPHA_EMISSION_THRESHOLD = 0.85

    for line_name, rest_wave in rest_lines.items():
        hw = search_radius.get(line_name, 1.5)  # half-width in nm
        mask = (spectrum['waveobs'] >= rest_wave - hw) & \
               (spectrum['waveobs'] <= rest_wave + hw)
        if mask.sum() < 10:
            print(f"  {line_name:10s} @ {rest_wave:.2f} nm : OUT OF RANGE")
            continue

        flux_window = spectrum['flux'][mask]
        wave_window = spectrum['waveobs'][mask]

        # Parabolic refinement around the pixel minimum for sub-pixel accuracy
        min_idx = np.argmin(flux_window)
        if 1 <= min_idx <= len(flux_window) - 2:
            y0, y1, y2 = flux_window[min_idx - 1], flux_window[min_idx], flux_window[min_idx + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                dx = 0.5 * (y0 - y2) / denom  # fractional pixel offset
                pixel_scale = wave_window[min_idx] - wave_window[min_idx - 1]
                measured_wave = wave_window[min_idx] + dx * pixel_scale
            else:
                measured_wave = wave_window[min_idx]
        else:
            measured_wave = wave_window[min_idx]
        measured_depth = flux_window[min_idx]

        # Flag Hα if core depth is too shallow (emission filling)
        if line_name == 'Hα' and measured_depth > HALPHA_EMISSION_THRESHOLD:
            print(colored(
                f"  {line_name:10s} @ {rest_wave:.2f} nm  →  "
                f"SKIPPED (depth={measured_depth:.3f} > {HALPHA_EMISSION_THRESHOLD:.2f}; "
                f"core likely emission-filled in active star)",
                'yellow'
            ))
            continue

        # Calculate velocity shift
        vel_shift = (measured_wave - rest_wave) / rest_wave * 299792.458  # km/s

        color = 'green' if abs(vel_shift) < 2.0 else 'yellow' if abs(vel_shift) < 5.0 else 'red'
        print(colored(
            f"  {line_name:10s} @ {rest_wave:.2f} nm  →  "
            f"measured {measured_wave:.2f} nm  "
            f"(depth={measured_depth:.3f}, Δv={vel_shift:+.1f} km/s)",
            color
        ))
    
    print(colored(f"{'='*60}\n", 'blue'))
    
    # Plot diagnostic
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, (line_name, rest_wave) in enumerate(rest_lines.items()):
        ax = axes[idx]
        mask = (spectrum['waveobs'] >= rest_wave - 2.0) & \
               (spectrum['waveobs'] <= rest_wave + 2.0)
        if mask.sum() > 10:
            ax.plot(spectrum['waveobs'][mask]*10, spectrum['flux'][mask], 'k-', lw=1)
            ax.axvline(rest_wave*10, color='red', linestyle='--', alpha=0.7, label='Rest frame')
            ax.set_title(f'{line_name} ({rest_wave:.2f} nm)')
            ax.set_xlabel('Wavelength [Å]')
            ax.set_ylabel('Normalized flux')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'{line_name}\nOUT OF RANGE', 
                   ha='center', va='center', transform=ax.transAxes)
    
    axes[-1].axis('off')
    plt.tight_layout()
    outpath = os.path.join(output_dir, f'{target_name}_diagnostic_rv_check.png')
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(colored(f"    Diagnostic plot saved: {outpath}", 'cyan'))


def measure_caii_irt_filling(spectrum, fill_threshold=0.55, hw_nm=0.15):
    """
    Measure Ca II IRT core filling as a spectral activity indicator.

    Uses the two IRT lines within the synthesis window (837-860 nm):
      849.802 nm  and  854.209 nm
    (866.214 nm falls outside the window and is skipped.)

    A line is considered emission-filled — and the star flagged as active —
    when the normalized core flux exceeds fill_threshold.  For inactive K
    dwarfs the IRT cores typically reach 0.10–0.40; chromospheric filling
    raises them above ~0.50–0.55.

    Parameters
    ----------
    spectrum      : iSpec spectrum table (normalized, telluric-cleaned)
    fill_threshold: normalized flux above which a core is flagged as filled
    hw_nm         : half-width of the search window around each line [nm]

    Returns
    -------
    is_active_spectral : bool   — True if ≥1 line core is filled
    core_depths        : dict   — {wavelength_nm: core_flux} for measured lines
    n_measured         : int    — number of lines actually found in spectrum
    """
    irt_lines_nm = [849.802, 854.209]
    core_depths = {}

    for lam in irt_lines_nm:
        mask = ((spectrum['waveobs'] >= lam - hw_nm) &
                (spectrum['waveobs'] <= lam + hw_nm))
        if mask.sum() < 3:
            continue
        core_depths[lam] = float(np.min(spectrum['flux'][mask]))

    if not core_depths:
        return False, core_depths, 0

    is_active_spectral = any(d > fill_threshold for d in core_depths.values())
    return is_active_spectral, core_depths, len(core_depths)


def check_mask_overlay(target_name, output_dir='.'):
    """
    Optional visual check: overlay the CCF mask lines on the observed spectrum.
    Saves to output directory (no interactive window).
    """
    import matplotlib.pyplot as plt

    fits_file = os.path.join(SPECTRA_DIR, f"{target_name}_merged.fits")
    if not os.path.exists(fits_file):
        print(colored(f"ERROR: File not found: {fits_file}", 'red'))
        sys.exit(1)

    spectrum = read_cafe_spectrum(fits_file)
    # Use the full spectrum (no synthesis-window restriction) so the zoom
    # window falls within CARMENES VIS coverage (~520–960 nm).

    wave_nm = spectrum['waveobs']
    flux = spectrum['flux']

    good = np.isfinite(wave_nm) & np.isfinite(flux)
    wave_nm = wave_nm[good]
    flux = flux[good]

    scale = np.nanpercentile(flux, 95)
    if (not np.isfinite(scale)) or (scale == 0):
        scale = np.nanmedian(np.abs(flux))
    if (not np.isfinite(scale)) or (scale == 0):
        scale = 1.0
    flux_norm = flux / scale

    ccf_mask = ispec.read_cross_correlation_mask(MASK_FILE)
    mask_wave_nm = ccf_mask['wave_peak']
    mask_depth = ccf_mask['depth']

    # Zoom-in window — centred at 6200 Å, within CARMENES VIS + K5 mask coverage
    zoom_center_aa = 6200.0
    zoom_width_aa  = 80.0
    zoom_min_nm = (zoom_center_aa - 0.5 * zoom_width_aa) / 10.0
    zoom_max_nm = (zoom_center_aa + 0.5 * zoom_width_aa) / 10.0

    in_zoom_spec = (wave_nm >= zoom_min_nm) & (wave_nm <= zoom_max_nm)
    wave_nm = wave_nm[in_zoom_spec]
    flux_norm = flux_norm[in_zoom_spec]

    in_zoom_mask = (mask_wave_nm >= zoom_min_nm) & (mask_wave_nm <= zoom_max_nm)
    mask_wave_nm = mask_wave_nm[in_zoom_mask]
    mask_depth = mask_depth[in_zoom_mask]

    if len(wave_nm) == 0:
        print(colored("ERROR: No spectrum points in zoom window.", 'red'))
        return

    print(colored(f"\nCHECK MASK MODE: {target_name}", 'blue'))
    print(f"  Mask file     : {os.path.basename(os.path.dirname(MASK_FILE))}")
    print(f"  Zoom range    : {wave_nm.min()*10:.1f} – {wave_nm.max()*10:.1f} Å")
    print(f"  Mask lines in range: {len(mask_wave_nm)}")

    depth_max = np.nanmax(mask_depth) if len(mask_depth) else 1.0
    if (not np.isfinite(depth_max)) or (depth_max <= 0):
        depth_max = 1.0

    plt.figure(figsize=(14, 6))
    plt.plot(wave_nm * 10, flux_norm, color='black', lw=0.8, label='Observed spectrum (scaled)')

    y_top = np.nanmax(flux_norm)
    y_bottom = np.nanmin(flux_norm)
    flux_span = max(y_top - y_bottom, 1e-6)
    y_margin = 0.06 * flux_span
    y_min = y_bottom - y_margin
    y_max = y_top + y_margin

    # Draw full-height mask lines for clear visual comparison
    plt.vlines(mask_wave_nm * 10, y_min, y_max, color='red', alpha=0.60, lw=0.9,
               label='MASK_FILE lines')

    plt.ylim(y_min, y_max)

    # Keep full y-axis range (no manual clipping)
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Flux (scaled)')
    plt.title(f'{target_name} — observed spectrum + CCF mask ({os.path.basename(os.path.dirname(MASK_FILE))})')
    plt.legend(loc='upper right')
    plt.tight_layout()
    outpath = os.path.join(output_dir, f'{target_name}_check_mask.pdf')
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(colored(f"    Mask overlay saved: {outpath}", 'cyan'))


def plot_hrd_parsec(target_name, teff, e_teff, lum, e_lum, mh=0.0,
                    output_file=None):
    """
    HR diagram: L vs Teff with PARSEC isochrones as context.

    Reads PARSEC_ISO_FILE and plots isochrones at the closest [M/H] to the
    target, for ages 1–10 Gyr.  The target is overplotted with error bars.

    Parameters
    ----------
    target_name : str
    teff, e_teff : float — effective temperature [K] and 1σ error
    lum, e_lum   : float — luminosity [L☉] and 1σ error
    mh           : float — [M/H] of the target (to select nearest isochrone set)
    output_file  : str or None — output PDF path (auto-generated if None)
    """
    import matplotlib.pyplot as plt

    if output_file is None:
        output_file = f"{target_name}_hrd_parsec.pdf"

    if not os.path.exists(PARSEC_ISO_FILE):
        print(colored(f"    WARNING: PARSEC file not found, skipping HRD plot.", 'magenta'))
        return

    # ── Read PARSEC data ──
    data = np.genfromtxt(PARSEC_ISO_FILE, comments='#')
    col_mh    = data[:, 1]
    col_age   = data[:, 2]   # logAge
    col_mini  = data[:, 3]   # initial mass — monotonic along each isochrone
    col_logL  = data[:, 6]
    col_logTe = data[:, 7]
    col_label = data[:, 9].astype(int)

    # Select nearest MH
    mh_vals = np.unique(col_mh)
    mh_near = mh_vals[np.argmin(np.abs(mh_vals - mh))]

    # Ages to plot (logAge → label)
    age_grid = [
        (9.0,  "1 Gyr"),
        (9.3,  "2 Gyr"),
        (9.5,  "3 Gyr"),
        (9.7,  "5 Gyr"),
        (9.85, "7 Gyr"),
        (9.95, "9 Gyr"),
        (10.0, "10 Gyr"),
    ]

    # ── Find best-match isochrone ─────────────────────────────────────────
    # The closest MS isochrone to the target (Teff, L) in normalised log space.
    all_log_ages = np.unique(col_age)
    log_teff_target = np.log10(teff)
    log_lum_target  = np.log10(max(lum, 1e-6))
    best_dist    = np.inf
    best_log_age = all_log_ages[0]
    for _la in all_log_ages:
        _sel = ((col_mh == mh_near) & (col_age == _la) & (col_label == 1))
        if _sel.sum() < 3:
            continue
        _dist = np.min(np.sqrt(
            ((col_logTe[_sel] - log_teff_target) / 0.015) ** 2 +
            ((col_logL[_sel]  - log_lum_target)  / 0.25)  ** 2
        ))
        if _dist < best_dist:
            best_dist    = _dist
            best_log_age = _la
    age_best_gyr = 10.0 ** (best_log_age - 9.0)

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, ls=':', lw=0.5, color='0.78', zorder=0)

    # Gray gradient: youngest → darkest, oldest → lightest
    age_grid_gyr = [1, 2, 3, 5, 7, 9, 10, 13]
    n_iso        = len(age_grid_gyr)
    # index 0 = 1 Gyr (darkest 0.12), index n-1 = 13 Gyr (lightest 0.72)
    gray_levels  = np.linspace(0.12, 0.72, n_iso)

    for k, age_gyr in enumerate(age_grid_gyr):
        log_age_target = np.log10(age_gyr * 1e9)
        la = all_log_ages[np.argmin(np.abs(all_log_ages - log_age_target))]
        is_best = np.isclose(la, best_log_age, atol=0.06)

        iso_color = 'black' if is_best else str(gray_levels[k])
        iso_lw    = 2.0     if is_best else 1.0
        iso_alpha = 1.0     if is_best else 0.85
        iso_z     = 4       if is_best else 2

        # MS track
        sel_ms = ((col_mh == mh_near) & (col_age == la) & (col_label == 1))
        teff_ms = lum_ms = None
        if sel_ms.sum() >= 3:
            order  = np.argsort(col_mini[sel_ms])
            teff_ms = 10.0 ** col_logTe[sel_ms][order]
            lum_ms  = 10.0 ** col_logL[sel_ms][order]
            ax.plot(teff_ms, lum_ms,
                    color=iso_color, lw=iso_lw, alpha=iso_alpha, zorder=iso_z)

        # Evolved extension (SGB / low RGB) — dashed
        sel_ev = ((col_mh == mh_near) & (col_age == la) &
                  (col_label >= 2) & (col_label <= 3))
        if sel_ev.sum() >= 2:
            order_ev = np.argsort(col_mini[sel_ev])
            teff_ev  = 10.0 ** col_logTe[sel_ev][order_ev]
            lum_ev   = 10.0 ** col_logL[sel_ev][order_ev]
            ax.plot(teff_ev, lum_ev,
                    color=iso_color, lw=iso_lw * 0.6, ls='--',
                    alpha=iso_alpha * 0.55, zorder=iso_z - 1)
            if teff_ms is not None:
                ax.plot([teff_ms[-1], teff_ev[0]], [lum_ms[-1], lum_ev[0]],
                        color=iso_color, lw=iso_lw * 0.6, ls='--',
                        alpha=iso_alpha * 0.55, zorder=iso_z - 1)

        # Age label at MSTO (brightest MS point) for non-best isochrones
        if not is_best and teff_ms is not None:
            idx_top = np.argmax(lum_ms)
            ax.text(teff_ms[idx_top], lum_ms[idx_top] * 1.06,
                    f"{age_gyr} Gyr",
                    color=iso_color, fontsize=8, fontweight='bold',
                    va='bottom', ha='center', clip_on=True)

    # ── Plot the exact best-match isochrone if it falls between grid ages ──
    # (already plotted above if grid age ≈ best; this handles intermediate ages)
    if not any(np.isclose(
            all_log_ages[np.argmin(np.abs(all_log_ages - np.log10(g * 1e9)))],
            best_log_age, atol=0.06) for g in age_grid_gyr):
        sel_ms_b = ((col_mh == mh_near) & (col_age == best_log_age) & (col_label == 1))
        if sel_ms_b.sum() >= 3:
            order_b   = np.argsort(col_mini[sel_ms_b])
            teff_ms_b = 10.0 ** col_logTe[sel_ms_b][order_b]
            lum_ms_b  = 10.0 ** col_logL[sel_ms_b][order_b]
            ax.plot(teff_ms_b, lum_ms_b, color='black', lw=2.0, zorder=4)
        sel_ev_b = ((col_mh == mh_near) & (col_age == best_log_age) &
                    (col_label >= 2) & (col_label <= 3))
        if sel_ev_b.sum() >= 2 and sel_ms_b.sum() >= 3:
            order_evb = np.argsort(col_mini[sel_ev_b])
            teff_ev_b = 10.0 ** col_logTe[sel_ev_b][order_evb]
            lum_ev_b  = 10.0 ** col_logL[sel_ev_b][order_evb]
            ax.plot(teff_ev_b, lum_ev_b, color='black', lw=1.2, ls='--',
                    alpha=0.55, zorder=3)
            ax.plot([teff_ms_b[-1], teff_ev_b[0]],
                    [lum_ms_b[-1], lum_ev_b[0]],
                    color='black', lw=1.2, ls='--', alpha=0.55, zorder=3)

    # ── Target point ──────────────────────────────────────────────────────
    ax.errorbar(teff, lum, xerr=e_teff, yerr=e_lum,
                fmt='o', color='red', ms=9, mec='red', mew=1.0,
                ecolor='red', elinewidth=1.5, capsize=0, zorder=10)

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlabel(r"$T_{\rm eff}$  [K]", fontsize=14)
    ax.set_ylabel(r"$L\,/\,L_\odot$", fontsize=14)
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.tick_params(labelsize=11)

    teff_margin = max(700, int(0.18 * teff / 100) * 100)
    teff_hi = min(8000, teff + teff_margin)
    teff_lo = max(3200, teff - teff_margin)
    lum_lo  = max(0.003, lum / 12.0)
    lum_hi  = min(300.0, lum * 25.0)
    ax.set_xlim(teff_hi, teff_lo)
    ax.set_ylim(lum_lo,  lum_hi)

    # ── Corner annotations ────────────────────────────────────────────────
    # Target name — bottom left
    ax.text(0.04, 0.05, target_name, transform=ax.transAxes,
            fontsize=20, fontweight='bold', color='black',
            va='bottom', ha='left', zorder=11)
    # Best age — bottom right
    ax.text(0.96, 0.05, f"{age_best_gyr:.1f} Gyr", transform=ax.transAxes,
            fontsize=14, fontweight='bold', color='black',
            va='bottom', ha='right', zorder=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(colored(f"    HRD plot saved: {output_file}", 'green'))
    plt.close(fig)


def check_spectrum_fit(observed_spectrum, synthetic_spectrum, output_file="check_spectrum.pdf"):
    """
    Diagnostic plot: observed vs synthetic in the GES line regions used for the fit.

    Layout: 5 wavelength panels (left: spectrum, right: residuals).
    Full spectrum shown in light gray for context; fit segments highlighted
    in black (observed) and red (synthetic).  RMS computed only on fit-segment
    pixels, giving an honest measure of fit quality.
    """
    import matplotlib.pyplot as plt

    # ── Load the same line regions used in the actual fit ──────────────────
    _plot_tag = "VALD" if SYNTH_CODE == "turbospectrum" else "GES"
    line_regions = ispec.read_line_regions(
        ISPEC_DIR + f"input/regions/47000_{_plot_tag}/{SYNTH_CODE}_synth_good_for_params_all.txt"
    )
    if EXCLUDE_STRONG_LINES:
        def _in_excl(wp):
            return any(lo <= wp <= hi for lo, hi in EXCLUSION_WINDOWS_NM)
        line_regions = line_regions[
            ~np.array([_in_excl(w) for w in line_regions['wave_peak']])
        ]

    # Build a boolean mask: True for pixels inside any fit segment (±0.25 nm margin)
    segment_mask = np.zeros(len(observed_spectrum), dtype=bool)
    margin = 0.25  # nm — same as create_segments_around_lines
    for lr in line_regions:
        wb = lr['wave_base'] - margin
        wt = lr['wave_top'] + margin
        segment_mask |= ((observed_spectrum['waveobs'] >= wb) &
                         (observed_spectrum['waveobs'] <= wt))

    # ── Diagnostic wavelength windows — derived from SYNTH_WINDOWS_NM ────
    # Long windows are split into ≤PANEL_WIDTH nm sub-panels so that every
    # panel contains real CARMENES data and none are empty.
    PANEL_WIDTH = 25.0  # nm per sub-panel
    regions = []
    for wb, wt in SYNTH_WINDOWS_NM:
        width = wt - wb
        if width <= PANEL_WIDTH:
            regions.append((wb, wt))
        else:
            n = int(np.ceil(width / PANEL_WIDTH))
            edges = np.linspace(wb, wt, n + 1)
            for j in range(n):
                regions.append((float(edges[j]), float(edges[j + 1])))

    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 2,
                             figsize=(14, max(2.5 * n_regions, 6)),
                             gridspec_kw={"width_ratios": [3, 1]})

    global_res_seg = []  # accumulate segment residuals across all panels

    for i, (wmin, wmax) in enumerate(regions):

        win = ((observed_spectrum['waveobs'] >= wmin) &
               (observed_spectrum['waveobs'] <= wmax))

        wave     = observed_spectrum['waveobs'][win]
        flux_obs = observed_spectrum['flux'][win]
        flux_syn = synthetic_spectrum['flux'][win]
        seg      = segment_mask[win]

        residuals = flux_obs - flux_syn

        # ── Spectrum panel ────────────────────────────────────────────────
        ax_sp = axes[i, 0]
        # Full spectrum in light gray (context)
        ax_sp.plot(wave * 10, flux_obs, color='0.80', lw=0.6, zorder=1)
        ax_sp.plot(wave * 10, flux_syn, color='#ffaaaa', lw=0.6, zorder=1)

        # Fit segments: overplot in solid black/red
        wave_seg = np.where(seg, wave, np.nan)
        ax_sp.plot(wave_seg * 10, np.where(seg, flux_obs, np.nan),
                   'k', lw=1.0, zorder=3, label="Observed (fit segments)")
        ax_sp.plot(wave_seg * 10, np.where(seg, flux_syn, np.nan),
                   'r', lw=1.0, zorder=3, label="Synthetic (fit segments)")

        # Shade fit segments lightly
        seg_diff = np.diff(seg.astype(int), prepend=0, append=0)
        starts = np.where(seg_diff == 1)[0]
        ends   = np.where(seg_diff == -1)[0]
        for s, e in zip(starts, ends):
            if s < len(wave) and e <= len(wave):
                ax_sp.axvspan(wave[s] * 10, wave[min(e, len(wave)-1)] * 10,
                              color='blue', alpha=0.06, zorder=0)

        n_seg = seg.sum()
        ax_sp.set_ylabel("Flux")
        ax_sp.set_title(f"{wmin:.0f}–{wmax:.0f} nm  ({n_seg} px in fit)")
        if i == 0:
            ax_sp.legend(fontsize=7, loc='lower left')

        # ── Residual panel ────────────────────────────────────────────────
        ax_res = axes[i, 1]
        # Full residuals in light gray
        ax_res.plot(wave * 10, residuals, color='0.80', lw=0.5, zorder=1)
        # Segment residuals in blue
        ax_res.plot(wave_seg * 10, np.where(seg, residuals, np.nan),
                    'b', lw=0.8, zorder=3)
        ax_res.axhline(0, color='k', ls='--', lw=0.8)

        if n_seg > 0:
            rms_seg = np.sqrt(np.mean(residuals[seg] ** 2))
            rms_all = np.sqrt(np.mean(residuals ** 2))
            ax_res.set_title(f"RMS fit={rms_seg:.4f}\n(full={rms_all:.4f})",
                             fontsize=9)
            global_res_seg.append(residuals[seg])
        else:
            ax_res.set_title("No fit segments", fontsize=9)

        ax_res.set_xlabel("Wavelength [Å]")

    axes[-1, 0].set_xlabel("Wavelength [Å]")

    # Global RMS across all fit segments
    if global_res_seg:
        all_res = np.concatenate(global_res_seg)
        rms_global = np.sqrt(np.mean(all_res ** 2))
        fig.suptitle(f"Spectrum fit diagnostic  —  Global RMS (fit segments) = {rms_global:.4f}",
                     fontsize=12, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"    Diagnostic spectrum plot saved: {output_file}")
    plt.close(fig)


def generate_full_synthetic_spectrum(observed_spectrum, params,
                                     modeled_layers_pack, atomic_linelist,
                                     isotopes, solar_abundances):
    """
    Generate a full-range synthetic spectrum on the same wavelength grid
    as the observed spectrum, using best-fit stellar parameters.
    """
    target = {
        "teff": float(params["teff"]),
        "logg": float(params["logg"]),
        "MH": float(params["MH"]),
    }

    if not ispec.valid_atmosphere_target(modeled_layers_pack, target):
        raise ValueError(f"Best-fit parameters out of atmosphere grid range: {target}")

    atmosphere_layers = ispec.interpolate_atmosphere_layers(
        modeled_layers_pack,
        target,
        code=SYNTH_CODE
    )

    synthetic_flux = ispec.generate_spectrum(
        observed_spectrum["waveobs"],
        atmosphere_layers,
        params["teff"],
        params["logg"],
        params["MH"],
        params["alpha"],
        atomic_linelist,
        isotopes,
        solar_abundances,
        None,
        microturbulence_vel=params["vmic"],
        macroturbulence=params["vmac"],
        vsini=params["vsini"],
        limb_darkening_coeff=params["limb_darkening_coeff"],
        R=params["R"],
        verbose=0,
        code=SYNTH_CODE,
        use_molecules=(SYNTH_CODE == "turbospectrum" and USE_MOLECULES),
    )

    synthetic_spectrum = observed_spectrum.copy()
    synthetic_spectrum["flux"] = synthetic_flux
    return synthetic_spectrum

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Wavelength window selection
# ─────────────────────────────────────────────────────────────────────────────

def apply_synth_windows(spectrum, windows=None):
    """
    Keep only pixels that fall within one of the clean synthesis windows,
    discarding telluric-band gaps and beyond-range pixels.

    Parameters
    ----------
    spectrum : iSpec recarray (waveobs in nm)
    windows  : list of (wave_base_nm, wave_top_nm) tuples.
               Defaults to SYNTH_WINDOWS_NM.

    Returns
    -------
    filtered spectrum recarray
    n_windows_used : int
    total_pixels   : int
    """
    if windows is None:
        windows = SYNTH_WINDOWS_NM

    mask = np.zeros(len(spectrum), dtype=bool)
    for wb, wt in windows:
        mask |= (spectrum['waveobs'] >= wb) & (spectrum['waveobs'] <= wt)
    return spectrum[mask], len(windows), int(mask.sum())


def apply_rv_window(spectrum):
    """Keep only pixels within [WAVE_MIN_NM, 680 nm] — the range covered by
    the K5 CCF mask — for the RV cross-correlation step."""
    wf = ispec.create_wavelength_filter(spectrum, wave_base=WAVE_MIN_NM, wave_top=680.0)
    return spectrum[wf]


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_cafe_spectrum(fits_file):
    """
    Read a CAFE combined spectrum FITS and return an iSpec-compatible
    structured array with fields: waveobs [nm], flux, err.

    The CAFE pipeline stores wavelengths in Angstroms → divide by 10.

    FLUX_STD is the epoch-to-epoch scatter (not the photon noise of the
    combined spectrum). The proper per-pixel error is estimated directly
    from the combined flux using a MAD-based local sliding window, which
    gives the true noise of the combined spectrum independent of N_epochs.
    """
    with fits.open(fits_file) as f:
        wave_aa = f['WAVELENGTH'].data.copy()   # Angstroms
        flux    = f['FLUX'].data.copy()

    wave_nm = wave_aa / 10.0   # Å → nm

    # Build iSpec structured array
    spectrum = np.recarray(len(wave_nm),
                           dtype=[('waveobs', float),
                                  ('flux',    float),
                                  ('err',     float)])
    spectrum['waveobs'] = wave_nm
    spectrum['flux']    = flux.astype(float)

    # Remove pixels with non-finite flux (NaN/Inf from bad orders or edges)
    good = np.isfinite(spectrum['flux']) & np.isfinite(spectrum['waveobs'])
    spectrum = spectrum[good]

    # Estimate per-pixel noise from the combined flux using a local MAD window.
    # This avoids using FLUX_STD (epoch scatter, ~1.8x too large) and gives
    # the true noise of the combined spectrum.
    err = _estimate_local_noise(spectrum['flux'], window=201)
    # Conservative floor: 1/500 of median flux (= S/N floor of 500)
    floor = np.nanmedian(np.abs(spectrum['flux'])) / 500.0
    spectrum['err'] = np.where((err > 0) & np.isfinite(err), err, floor)

    return spectrum


def _estimate_local_noise(flux, window=201):
    """
    Estimate per-pixel noise via a sliding MAD window on the flux array.

    For each pixel i, the local residuals relative to a running median are
    computed over a window of ±half pixels; the MAD of those residuals gives
    the local σ.  Edge pixels use the nearest valid σ.

    Parameters
    ----------
    flux    : 1-D float array, already stripped of NaN
    window  : int (odd), sliding window width in pixels

    Returns
    -------
    err : 1-D float array, same length as flux
    """
    from scipy.ndimage import median_filter

    flux  = np.asarray(flux, dtype=np.float64)
    trend = median_filter(flux, size=window, mode='reflect')
    residuals = flux - trend

    half = window // 2
    n    = len(flux)
    err  = np.empty(n)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        seg = residuals[lo:hi]
        if len(seg) < 5:
            err[i] = np.nan
            continue
        mad = np.median(np.abs(seg - np.median(seg)))
        err[i] = 1.4826 * mad   # MAD → σ

    # Fill edge NaNs by interpolation from nearest valid value
    valid = np.isfinite(err) & (err > 0)
    if valid.sum() < 2:
        err[:] = np.nanmedian(np.abs(residuals))
    else:
        err = np.where(valid, err,
                       np.interp(np.arange(n), np.where(valid)[0], err[valid]))
    return err


# ─────────────────────────────────────────────────────────────────────────────
# Telluric cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_telluric_regions(spectrum):
    """
    Identify and remove pixels contaminated by telluric absorption lines
    using a cross-correlation with a synthetic telluric mask.
    Returns the cleaned spectrum and the measured telluric barycentric velocity.
    If the CCF cannot find a peak (e.g. weak tellurics in a combined spectrum)
    the original spectrum is returned unchanged with bv = 0.
    """
    telluric_ll = ispec.read_telluric_linelist(TELLURIC_FILE, minimum_depth=0.0)

    try:
        models, _ = ispec.cross_correlate_with_mask(
            spectrum, telluric_ll,
            lower_velocity_limit=-100, upper_velocity_limit=100,
            velocity_step=0.5, mask_depth=0.01,
            fourier=False, only_one_peak=True
        )
        bv     = float(np.round(models[0].mu(),  4))
        bv_err = float(np.round(models[0].emu(), 4))
    except (IndexError, Exception) as e:
        print(colored(f"    WARNING: Telluric CCF failed ({e}). Skipping telluric masking.", 'magenta'))
        return spectrum, 0.0, 0.0

    # Keep only the 25% deepest telluric lines for masking
    dfilter = telluric_ll['depth'] > np.percentile(telluric_ll['depth'], 75)
    tfilter = ispec.create_filter_for_regions_affected_by_tellurics(
        spectrum['waveobs'], telluric_ll[dfilter],
        min_velocity=-bv - 30.0, max_velocity=-bv + 30.0
    )
    clean = spectrum[~tfilter]
    return clean, bv, bv_err


# ─────────────────────────────────────────────────────────────────────────────
# Continuum normalization
# ─────────────────────────────────────────────────────────────────────────────

def _continuum_fit_kwargs(mode):
    """
    Return iSpec continuum-fit parameters for stress-testing normalization.
    """
    profiles = {
        'baseline': dict(degree=2, nknots=5, order="median+max",
                         median_wave_range=0.1, max_wave_range=1.0),
        'rigid':    dict(degree=2, nknots=3, order="median+max",
                         median_wave_range=0.2, max_wave_range=2.0),
        'flex':     dict(degree=3, nknots=10, order="median+max",
                         median_wave_range=0.08, max_wave_range=0.7),
        'local':    dict(degree=2, nknots=14, order="median+max",
                         median_wave_range=0.05, max_wave_range=0.5),
    }
    if mode not in profiles:
        raise ValueError(f"Unsupported continuum mode '{mode}'")
    return profiles[mode]


def normalize_spectrum_spline(spectrum, mode=None):
    """
    Fit a spline continuum to the spectrum and normalize.

    Strategy:
      - The spectrum is already pipeline-normalized; this is a residual correction
        for blaze residuals and order-edge artifacts.
      - Uses iSpec continuum regions (iron-line-free windows) as anchor points
      - Excludes strong lines (Hα, Na I D, Hβ, Ca II, etc.) via STRONG_LINES_FILE
      - baseline mode uses degree=2, nknots=5
      - stress modes alter only the continuum model, not fit physics
      - err is divided by the same continuum model to preserve S/N

    Returns
    -------
    normalized : structured array (waveobs, flux, err) with flux ≈ 1.0
    continuum_model : iSpec continuum model (for iSpec model_spectrum)
    """
    if mode is None:
        mode = CONTINUUM_MODE
    if mode == 'none':
        return spectrum.copy(), None

    strong_lines = ispec.read_line_regions(STRONG_LINES_FILE)
    fit_kwargs = _continuum_fit_kwargs(mode)

    continuum_model = ispec.fit_continuum(
        spectrum,
        from_resolution=CARMENES_RESOLUTION,
        model="Splines",
        degree=fit_kwargs['degree'],
        nknots=fit_kwargs['nknots'],
        order=fit_kwargs['order'],
        median_wave_range=fit_kwargs['median_wave_range'],
        max_wave_range=fit_kwargs['max_wave_range'],
        ignore=strong_lines,
        automatic_strong_line_detection=True,
        strong_line_probability=0.5,
        use_errors_for_fitting=True,
    )

    normalized = ispec.normalize_spectrum(
        spectrum, continuum_model, consider_continuum_errors=False
    )

    # Divide err by the continuum to keep S/N consistent after normalization.
    # iSpec's normalize_spectrum only divides flux, not err.
    cont_values = continuum_model(spectrum['waveobs'])
    cont_values = np.where((cont_values > 0) & np.isfinite(cont_values),
                           cont_values, 1.0)
    normalized['err'] = spectrum['err'] / cont_values

    return normalized, continuum_model


def renormalize_with_synthetic(observed, synthetic, nknots=3):
    """
    Refine continuum normalization using the observed/synthetic ratio.

    After a first parameter fit, residual continuum errors show up as a smooth
    deviation of the ratio observed/synthetic from unity.  A low-order spline
    is fitted to this ratio and divided out, producing a better-normalized
    spectrum for the next synthesis iteration.

    Parameters
    ----------
    observed   : iSpec structured array (waveobs, flux, err)
    synthetic  : iSpec structured array with same waveobs grid
    nknots     : int, number of interior knots for the correction spline

    Returns
    -------
    corrected  : iSpec structured array with refined normalization
    """
    from scipy.interpolate import LSQUnivariateSpline

    ratio = observed['flux'] / synthetic['flux']
    wave = observed['waveobs']

    # Use only pixels where synthetic flux is significant and ratio is sensible
    good = (np.isfinite(ratio) & (synthetic['flux'] > 0.05) &
            (ratio > 0.5) & (ratio < 2.0))

    if good.sum() < 100:
        print(colored("    WARNING: Too few good pixels for re-normalization.", 'magenta'))
        return observed

    # Sigma-clip outliers (deep line cores, emission artifacts)
    med = np.nanmedian(ratio[good])
    mad = np.nanmedian(np.abs(ratio[good] - med))
    sigma = 1.4826 * mad
    clip = good & (np.abs(ratio - med) < 3 * sigma)

    if clip.sum() < 100:
        print(colored("    WARNING: Too few pixels after sigma-clip.", 'magenta'))
        return observed

    # Interior knots evenly spaced (excluding edges)
    wave_clip = wave[clip]
    knot_spacing = (wave_clip.max() - wave_clip.min()) / (nknots + 1)
    knots = np.linspace(wave_clip.min() + knot_spacing,
                        wave_clip.max() - knot_spacing, nknots)

    spline = LSQUnivariateSpline(wave_clip, ratio[clip], knots, k=3)
    correction = spline(wave)

    # Safety: clamp extreme corrections (should not happen with few knots)
    correction = np.where((correction > 0.8) & (correction < 1.2) &
                          np.isfinite(correction), correction, 1.0)

    corrected = observed.copy()
    corrected['flux'] = observed['flux'] / correction
    corrected['err'] = observed['err'] / correction

    max_dev = np.nanmax(np.abs(correction - 1.0))
    print(f"    Continuum correction: max |deviation| = {max_dev:.4f} ({max_dev*100:.1f}%)")

    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Radial velocity
# ─────────────────────────────────────────────────────────────────────────────

def estimate_rv_ccf(spectrum, lower=-200, upper=200):
    """
    Measure the radial velocity by cross-correlating the spectrum with the
    G2 binary mask.  Returns (rv, e_rv) in km/s.
    """
    mask = ispec.read_cross_correlation_mask(MASK_FILE)
    models, ccf = ispec.cross_correlate_with_mask(
        spectrum, mask,
        lower_velocity_limit=lower, upper_velocity_limit=upper
    )
    rv   = float(models[0].mu())
    e_rv = float(models[0].emu())
    return rv, e_rv


def estimate_vsini_ccf(spectrum, params,
                        modeled_layers_pack, atomic_linelist,
                        isotopes, solar_abundances,
                        velocity_step=0.5):
    """
    Independent vsini estimate from self-calibrated CCF FWHM deconvolution.

    Method:
      1. Cross-correlate observed spectrum with G2 mask → FWHM_obs
      2. Generate synthetic spectrum with same params but vsini ≈ 0 → CCF → FWHM_0
         (FWHM_0 encodes ALL non-rotational broadening: instrumental at R=94600,
          vmac, vmic, thermal, intrinsic line widths — no external calibrations needed)
      3. Generate synthetic spectrum with vsini = 10 km/s → CCF → FWHM_10
         (used to calibrate the self-calibrated FWHM²–vsini² proportionality constant K)
      4. Calibration:  K = vsini_ref / sqrt(FWHM_10² − FWHM_0²)
         vsini_ccf = K × sqrt(FWHM_obs² − FWHM_0²)

    The 2-point calibration (steps 2–3) accounts exactly for the non-Gaussian
    rotational kernel shape and the specific mask+instrument combination,
    avoiding the need for theoretical correction factors (Santos+02, Gray 2005).

    Parameters
    ----------
    spectrum : recarray
        Observed spectrum (normalized, RV-corrected).
    params : dict
        Fitted stellar parameters (teff, logg, MH, alpha, vmic, vmac, vsini, …).
    modeled_layers_pack, atomic_linelist, isotopes, solar_abundances
        iSpec data for synthetic spectrum generation.
    velocity_step : float
        CCF velocity step in km/s (default 0.5; finer than the 1.0 used for RV).

    Returns
    -------
    vsini_ccf : float — vsini [km/s]
    e_vsini   : float — propagated uncertainty [km/s]
    info      : dict  — diagnostic quantities (fwhm_obs, fwhm_0, fwhm_10, K, sig_obs, …)
    """
    VSINI_REF = 10.0   # reference vsini for calibration point [km/s]
    VSINI_ZERO = 0.1   # effectively zero rotation [km/s]

    mask = ispec.read_cross_correlation_mask(MASK_FILE)

    # ── 1. Observed CCF — dual velocity range for empirical error ───────
    # Run CCF at ±200 km/s (wide) and ±150 km/s (narrow); the difference
    # in the resulting FWHM gives an empirical reproducibility estimate for
    # the Gaussian fit, far more reliable than the formal covariance esig().
    models_obs_wide, _ = ispec.cross_correlate_with_mask(
        spectrum, mask,
        lower_velocity_limit=-200, upper_velocity_limit=200,
        velocity_step=velocity_step)
    models_obs_narrow, _ = ispec.cross_correlate_with_mask(
        spectrum, mask,
        lower_velocity_limit=-150, upper_velocity_limit=150,
        velocity_step=velocity_step)

    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    sig_obs      = abs(float(models_obs_wide[0].sig()))
    fwhm_obs     = sig_obs * fwhm_factor                        # primary measurement (wide)
    fwhm_obs_nrw = abs(float(models_obs_narrow[0].sig())) * fwhm_factor
    e_fwhm_obs   = abs(fwhm_obs - fwhm_obs_nrw)                # empirical FWHM reproducibility
    # Floor: at high vsini the wide/narrow fits converge to identical FWHM,
    # giving e_fwhm=0.  A minimum of 0.05 km/s ensures a finite vsini
    # uncertainty is always reported.
    e_fwhm_obs = max(e_fwhm_obs, 0.05)

    # ── 2. Synthetic CCF at vsini ≈ 0 ──────────────────────────────────
    # Restrict synthesis to the K5 mask wavelength range (562–680 nm).
    # The CCF only uses this window anyway; synthesizing the full multi-window
    # spectrum (562–860 nm) wastes ~3× computation for no gain.
    spectrum_ccf = apply_rv_window(spectrum)
    params_zero = dict(params)
    params_zero['vsini'] = VSINI_ZERO
    print(colored(f"      Generating synthetic (vsini={VSINI_ZERO} km/s) for FWHM_0...", 'yellow'))
    synth_zero = generate_full_synthetic_spectrum(
        spectrum_ccf, params_zero,
        modeled_layers_pack, atomic_linelist, isotopes, solar_abundances)
    models_z, _ = ispec.cross_correlate_with_mask(
        synth_zero, mask,
        lower_velocity_limit=-200, upper_velocity_limit=200,
        velocity_step=velocity_step)
    sig_0 = abs(float(models_z[0].sig()))
    fwhm_0 = sig_0 * 2.0 * np.sqrt(2.0 * np.log(2.0))

    # ── 3. Synthetic CCF at vsini = VSINI_REF ─────────────────────────
    params_ref = dict(params)
    params_ref['vsini'] = VSINI_REF
    print(colored(f"      Generating synthetic (vsini={VSINI_REF} km/s) for calibration...", 'yellow'))
    synth_ref = generate_full_synthetic_spectrum(
        spectrum_ccf, params_ref,
        modeled_layers_pack, atomic_linelist, isotopes, solar_abundances)
    models_r, _ = ispec.cross_correlate_with_mask(
        synth_ref, mask,
        lower_velocity_limit=-200, upper_velocity_limit=200,
        velocity_step=velocity_step)
    sig_ref = abs(float(models_r[0].sig()))
    fwhm_ref = sig_ref * 2.0 * np.sqrt(2.0 * np.log(2.0))

    # ── 4. Calibrate and measure vsini ─────────────────────────────────
    denom2 = fwhm_ref**2 - fwhm_0**2
    if denom2 <= 0:
        print(colored("      WARNING: calibration failed (FWHM_ref ≤ FWHM_0). "
                       "Cannot measure vsini from CCF.", 'red'))
        return np.nan, np.nan, {}

    K = VSINI_REF / np.sqrt(denom2)    # calibration constant

    fwhm2_rot = fwhm_obs**2 - fwhm_0**2
    if fwhm2_rot <= 0:
        vsini_ccf = 0.0
        e_vsini = np.nan
        print(colored("      Rotation unresolved (FWHM_obs ≤ FWHM_0).", 'cyan'))
    else:
        vsini_ccf = K * np.sqrt(fwhm2_rot)
        # Error propagation using empirical FWHM reproducibility:
        #   vsini = K * sqrt(FWHM_obs² - FWHM_0²)
        #   d(vsini)/d(FWHM_obs) = K * FWHM_obs / sqrt(FWHM_obs² - FWHM_0²)
        # e_FWHM_obs = |FWHM_wide − FWHM_narrow|  (reproducibility, not formal covariance)
        if e_fwhm_obs > 0 and vsini_ccf > 0:
            e_vsini = K * fwhm_obs * e_fwhm_obs / np.sqrt(fwhm2_rot)
        else:
            e_vsini = np.nan

    info = {
        'fwhm_obs':     fwhm_obs,
        'fwhm_obs_nrw': fwhm_obs_nrw,
        'e_fwhm_obs':   e_fwhm_obs,
        'fwhm_0':       fwhm_0,
        'fwhm_ref':     fwhm_ref,
        'K':            K,
        'sig_obs':      sig_obs,
        'sig_0':        sig_0,
        'sig_ref':      sig_ref,
        'vsini_ref':    VSINI_REF,
    }

    return vsini_ccf, e_vsini, info


# ─────────────────────────────────────────────────────────────────────────────
# Spectral synthesis
# ─────────────────────────────────────────────────────────────────────────────

def _line_note_array(line_regions):
    """Return line-region notes as stripped strings, or empty strings if absent."""
    if 'note' not in line_regions.dtype.names:
        return np.array([''] * len(line_regions), dtype=object)
    notes = []
    for val in line_regions['note']:
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='ignore')
        notes.append(str(val).strip())
    return np.array(notes, dtype=object)


def _line_depths_from_observed_spectrum(line_regions, spectrum):
    """
    Estimate local observed line depth as 1 - min(flux) inside each line window.

    This is a diagnostic classifier only; it is intentionally based on the
    normalized observed spectrum so weak/medium/strong subsets probe the actual
    CARMENES data quality and local continuum/blend behaviour.
    """
    depths = np.full(len(line_regions), np.nan, dtype=float)
    wave = spectrum['waveobs']
    flux = spectrum['flux']
    for idx, row in enumerate(line_regions):
        mask = ((wave >= row['wave_base']) & (wave <= row['wave_top']) &
                np.isfinite(flux))
        if np.any(mask):
            depths[idx] = max(0.0, 1.0 - float(np.nanmin(flux[mask])))
    return depths


def _line_species_mask(line_regions, species):
    """Return mask for a species note such as 'Fe 1' or 'Fe 2'."""
    wanted = species.lower().replace('  ', ' ').strip()
    notes = _line_note_array(line_regions)
    return np.array([
        note.lower().replace('  ', ' ').strip() == wanted
        for note in notes
    ], dtype=bool)


def _fe1_depth_masks(line_regions, spectrum):
    """Return Fe I/depth masks used by production and diagnostic filters."""
    is_fe1 = _line_species_mask(line_regions, 'Fe 1')
    depths = _line_depths_from_observed_spectrum(line_regions, spectrum)
    finite = np.isfinite(depths)
    is_fe1_weak = is_fe1 & finite & (depths < LINE_DEPTH_WEAK_MAX)
    is_fe1_medium = (is_fe1 & finite &
                     (depths >= LINE_DEPTH_WEAK_MAX) &
                     (depths < LINE_DEPTH_MEDIUM_MAX))
    is_fe1_strong = is_fe1 & finite & (depths >= LINE_DEPTH_MEDIUM_MAX)
    return is_fe1, depths, is_fe1_weak, is_fe1_medium, is_fe1_strong


def _problematic_fe1_medium_mask(line_regions, is_fe1_medium):
    """Return mask for medium Fe I lines excluded from the production list."""
    if not EXCLUDE_PROBLEMATIC_FE1_MEDIUM_IN_PRODUCTION:
        return np.zeros(len(line_regions), dtype=bool)
    if len(PROBLEMATIC_FE1_MEDIUM_WAVES_NM) == 0:
        return np.zeros(len(line_regions), dtype=bool)
    delta = np.abs(
        line_regions['wave_peak'][:, np.newaxis] -
        PROBLEMATIC_FE1_MEDIUM_WAVES_NM[np.newaxis, :]
    )
    return is_fe1_medium & np.any(delta <= FE1_MEDIUM_WAVE_TOL_NM, axis=1)


def _fe2_line_mode_wave(mode):
    """Return the Fe II wavelength for an individual-line diagnostic mode."""
    for label, wave_nm in FE2_DIAGNOSTIC_LINES:
        if mode == label:
            return wave_nm
    return None


def read_selected_line_regions_file(code=None):
    """Read the configured line-region source without applying subset filters."""
    if code is None:
        code = SYNTH_CODE

    _ges_linelist  = (ISPEC_DIR
                      + f"input/regions/47000_GES/{code}_synth_good_for_params_all.txt")
    _vald_linelist = (ISPEC_DIR
                      + f"input/regions/47000_VALD/{code}_synth_good_for_params_all.txt")

    _src = LINELIST_SOURCE.upper()
    if _src == 'GES':
        print(f"    Line regions : iSpec GES  480-679 nm  [{_ges_linelist}]")
        return ispec.read_line_regions(_ges_linelist)
    if _src == 'GES_MASKED':
        print(f"    Line regions : GES_MASKED 480-679 nm  [{_GES_MASKED_FILE}]")
        return ispec.read_line_regions(_GES_MASKED_FILE)
    if _src == 'VALD':
        print(f"    Line regions : iSpec VALD 480-679 nm  [{_vald_linelist}]")
        return ispec.read_line_regions(_vald_linelist)

    print(f"    Line regions : fallback VALD 480-679 nm  [{_vald_linelist}]")
    return ispec.read_line_regions(_vald_linelist)


def apply_line_subset_filter(line_regions, spectrum):
    """Apply the optional Fe I line-subset diagnostic filter."""
    mode = LINE_SUBSET_MODE.lower()
    if mode == 'all_diag':
        return line_regions

    is_fe1, _depths, is_fe1_weak, is_fe1_medium, is_fe1_strong = \
        _fe1_depth_masks(line_regions, spectrum)
    is_fe2 = _line_species_mask(line_regions, 'Fe 2')
    is_problematic_medium = _problematic_fe1_medium_mask(
        line_regions, is_fe1_medium)

    if mode == 'all':
        if (not EXCLUDE_WEAK_FE1_IN_PRODUCTION and
                not EXCLUDE_PROBLEMATIC_FE1_MEDIUM_IN_PRODUCTION):
            return line_regions
        keep = ~(is_fe1_weak | is_problematic_medium)
    elif mode == 'fe1':
        keep = is_fe1
    elif mode == 'fe2':
        keep = is_fe2
    elif mode in FE2_LINE_TEST_MODES:
        wave_nm = _fe2_line_mode_wave(mode)
        keep = is_fe2 & (np.abs(line_regions['wave_peak'] - wave_nm)
                         <= FE2_LINE_TOL_NM)
    elif mode in ('fe1_weak', 'fe1_medium', 'fe1_strong', 'fe1_weak_medium'):
        if mode == 'fe1_weak':
            keep = is_fe1_weak
        elif mode == 'fe1_medium':
            keep = is_fe1_medium
        elif mode == 'fe1_strong':
            keep = is_fe1_strong
        else:
            keep = is_fe1_weak | is_fe1_medium
    elif mode == 'fe1_blue':
        keep = is_fe1 & (line_regions['wave_peak'] < 620.0)
    elif mode == 'fe1_red':
        keep = is_fe1 & (line_regions['wave_peak'] >= 620.0)
    else:
        raise ValueError(f"Unsupported LINE_SUBSET_MODE='{LINE_SUBSET_MODE}'")

    filtered = line_regions[keep]
    print(f"    Line subset : {LINE_SUBSET_MODE}  "
          f"kept {len(filtered)} / {len(line_regions)} regions")
    if mode == 'all':
        n_weak = int(np.sum(is_fe1_weak & ~keep))
        n_medium = int(np.sum(is_problematic_medium & ~keep))
        if n_weak or n_medium:
            print(f"    Fe I production mask: removed {n_weak} weak + "
                  f"{n_medium} problematic medium regions")
    if len(filtered) == 0:
        raise RuntimeError(f"Line subset '{LINE_SUBSET_MODE}' left no valid regions.")
    return filtered


def estimate_stellar_parameters(spectrum, initial_params, free_params,
                                atomic_linelist, isotopes,
                                modeled_layers_pack, solar_abundances,
                                max_iterations=20, vmic_range=None):
    """
    Fit stellar parameters via spectral synthesis
    using the SPECTRUM radiative transfer code through iSpec.

    The spectrum should already be continuum-normalized before calling this
    function (via normalize_spectrum_spline). A fixed continuum of 1.0 is
    then passed to model_spectrum, which is correct for a pre-normalized input.
    Returns (params, errors, status, stats, synth_spec).
    """
    global LAST_LINE_REGION_COUNT
    LAST_LINE_REGION_COUNT = np.nan

    # Spectrum is pre-normalized: pass a fixed continuum = 1.0
    continuum_model = ispec.fit_continuum(
        spectrum, fixed_value=1.0, model="Fixed value"
    )

    code = SYNTH_CODE

    line_regions_file = read_selected_line_regions_file(code=code)

    line_regions_file = apply_line_subset_filter(line_regions_file, spectrum)

    # P7: remove line regions that fall inside strong-line exclusion windows
    # (Hα, Hβ, Hγ, Na I D, Ca II H&K).  These lines are badly reproduced by
    # 1D LTE models and would dominate χ² if included.
    if EXCLUDE_STRONG_LINES:
        def _in_exclusion(wave_peak):
            return any(lo <= wave_peak <= hi for lo, hi in EXCLUSION_WINDOWS_NM)
        n_before = len(line_regions_file)
        line_regions_file = line_regions_file[
            ~np.array([_in_exclusion(w) for w in line_regions_file['wave_peak']])
        ]
        n_removed = n_before - len(line_regions_file)
        if n_removed:
            print(f"    Strong-line exclusion: removed {n_removed} / {n_before} "
                  f"line regions (Hα/Hβ/Hγ/Na I D).")

    # Filter line regions to observed pixels — must fall entirely within one of
    # the synthesis windows (telluric gaps between windows have no data).
    n_before_range = len(line_regions_file)
    window_mask = np.zeros(len(line_regions_file), dtype=bool)
    for wb, wt in SYNTH_WINDOWS_NM:
        window_mask |= (
            (line_regions_file['wave_base'] >= wb) &
            (line_regions_file['wave_top']  <= wt)
        )
    line_regions_file = line_regions_file[window_mask]
    win_str = ', '.join(f"{wb*10:.0f}–{wt*10:.0f}" for wb, wt in SYNTH_WINDOWS_NM)
    print(f"    Window filter: {len(line_regions_file)} / {n_before_range} line regions "
          f"within synthesis windows [{win_str} Å].")

    # Filter line regions to the actual wavelength coverage of THIS spectrum.
    # The KOBE line regions file covers 562–860 nm, but GBS star spectra may
    # have different edge coverage or bad-pixel gaps; adjust_linemasks raises
    # "This should not happen" when a region window has no valid structure.
    _spec_wmin = np.nanmin(spectrum['waveobs'])
    _spec_wmax = np.nanmax(spectrum['waveobs'])
    _in_range = (
        (line_regions_file['wave_base'] >= _spec_wmin) &
        (line_regions_file['wave_top']  <= _spec_wmax)
    )

    def _has_sufficient_data(row):
        _m = ((spectrum['waveobs'] >= row['wave_base']) &
              (spectrum['waveobs'] <= row['wave_top']))
        _f = spectrum['flux'][_m]
        return (len(_f) >= 3 and
                int(np.sum(np.isfinite(_f) & (_f > 0))) >= 3)

    _has_data = np.array([_has_sufficient_data(r) for r in line_regions_file],
                         dtype=bool)
    _valid = _in_range & _has_data
    _n_dropped = int((~_valid).sum())
    if _n_dropped:
        print(f"    Coverage filter: removed {_n_dropped} / {len(line_regions_file)} "
              f"line regions with insufficient data in this spectrum "
              f"[{_spec_wmin:.3f}–{_spec_wmax:.3f} nm].")
    line_regions_file = line_regions_file[_valid]

    # Robust per-region adjust_linemasks: iSpec's __assert_structure can
    # raise "This should not happen" on degenerate flux patterns (flat,
    # monotonic, or pathological peak/valley counts).  Process each region
    # individually and silently skip those that fail.
    _adjusted_parts = []
    _n_adjust_failed = 0
    for _idx in range(len(line_regions_file)):
        _single = line_regions_file[_idx:_idx + 1].copy()
        try:
            _adj = ispec.adjust_linemasks(spectrum, _single, max_margin=0.5)
            _adjusted_parts.append(_adj)
        except Exception:
            _n_adjust_failed += 1
    if _n_adjust_failed:
        print(f"    adjust_linemasks: skipped {_n_adjust_failed} / "
              f"{len(line_regions_file)} regions (peak-structure assertion).")
    if not _adjusted_parts:
        raise RuntimeError("No valid line regions survived adjust_linemasks — "
                           "check spectrum quality and line regions file.")
    line_regions = np.concatenate(_adjusted_parts)
    LAST_LINE_REGION_COUNT = len(line_regions)
    segments     = ispec.create_segments_around_lines(line_regions, margin=0.25)

    # Enforce vmic bounds inside iSpec: override modeled_layers_pack[7]['vmic']
    # temporarily so __create_param_structure sees the fixed sensitivity range
    # instead of the default (0, 50) km/s from the atmosphere grid.
    _ranges = modeled_layers_pack[7]
    _orig_vmic_range = _ranges.get('vmic')
    if vmic_range is not None and 'vmic' in free_params:
        _ranges['vmic'] = tuple(vmic_range)
    try:
        obs_spec, synth_spec, params, errors, abunds, loggf, status, stats = \
            ispec.model_spectrum(
                spectrum,
                continuum_model,
                modeled_layers_pack,
                atomic_linelist,
                isotopes,
                solar_abundances,
                None,   # free_abundances
                None,   # linelist_free_loggf
                initial_params["teff"],
                initial_params["logg"],
                initial_params["MH"],
                initial_params["alpha"],
                initial_params["vmic"],
                initial_params["vmac"],
                initial_params["vsini"],
                initial_params["limb_darkening_coeff"],
                initial_params["R"],
                initial_params["vrad"],
                free_params,
                segments=segments,
                linemasks=line_regions,
                enhance_abundances=True,
                use_errors=True,
                max_iterations=max_iterations,
                code=code
            )
    finally:
        if vmic_range is not None:
            if _orig_vmic_range is None:
                _ranges.pop('vmic', None)
            else:
                _ranges['vmic'] = _orig_vmic_range

    return params, errors, status, stats, synth_spec

# ─────────────────────────────────────────────────────────────────────────────
# GBS Teff-dependent calibration
# ─────────────────────────────────────────────────────────────────────────────

def _read_gbs_fits(path, fmt):
    """
    Read a GBS spectrum from a non-iSpec FITS file.

    Supported formats
    -----------------
    'harps_phase3'
        ESO Phase 3 HARPS 1D product.  Data in extension 'SPECTRUM' (BINTABLE),
        columns WAVE [Å → nm], FLUX [ADU], ERR [ADU].  Not normalised.
        Also handles UVES Phase 3 products that expose FLUX/ERR columns.

    'uves_reduced'
        ESO Phase 3 UVES product where only FLUX_REDUCED / ERR_REDUCED (ADU)
        are provided.  Same WAVE [Å] column.

    'narval_fts'
        PolarBase NARVAL .fts file.  Data in extension 1 (BINTABLE),
        columns AWAV [nm], FLUX_NOR (already normalised), FLUX_ERR.

    'sophie_s1d'
        OHP SOPHIE S1D resampled product.  Flux in PrimaryHDU as a 1-D image;
        wavelength reconstructed from WCS (CRVAL1, CRPIX1, CDELT1) in Å.
        No per-pixel error array; we set a constant err = median(|flux|)/100.

    Returns
    -------
    numpy structured array with fields (waveobs [nm], flux, err)
    compatible with iSpec functions.
    """
    from astropy.io import fits as afits

    dtype = np.dtype([('waveobs', float), ('flux', float), ('err', float)])

    with afits.open(path, memmap=False) as hdul:
        if fmt == 'harps_phase3':
            tbl = hdul['SPECTRUM'].data
            wave_nm = tbl['WAVE'][0].astype(float) / 10.0   # Å → nm
            flux    = tbl['FLUX'][0].astype(float)
            err     = tbl['ERR'][0].astype(float)
            # Replace non-positive/NaN errors with local noise estimate
            bad = ~((err > 0) & np.isfinite(err))
            if bad.any():
                err[bad] = np.nanmedian(np.abs(flux)) / 200.0

        elif fmt == 'uves_reduced':
            tbl = hdul['SPECTRUM'].data
            wave_nm = tbl['WAVE'][0].astype(float) / 10.0
            flux    = tbl['FLUX_REDUCED'][0].astype(float)
            err     = tbl['ERR_REDUCED'][0].astype(float)
            bad = ~((err > 0) & np.isfinite(err))
            if bad.any():
                err[bad] = np.nanmedian(np.abs(flux)) / 200.0

        elif fmt == 'sophie_s1d':
            hdr  = hdul[0].header
            flux = np.asarray(hdul[0].data, dtype=float)
            n    = len(flux)
            cr1  = float(hdr.get('CRVAL1', 0.0))
            cd1  = float(hdr.get('CDELT1', 1.0))
            cp1  = float(hdr.get('CRPIX1', 1.0))
            wave_aa = cr1 + (np.arange(n) - cp1 + 1) * cd1
            wave_nm = wave_aa / 10.0
            err_const = np.nanmedian(np.abs(flux)) / 100.0   # assume SNR ≈ 100
            err = np.full(n, err_const, dtype=float)
            bad = ~np.isfinite(flux)
            if bad.any():
                flux[bad] = 0.0

        elif fmt == 'narval_fts':
            tbl  = hdul[1].data
            wave_nm = tbl['AWAV'].astype(float)     # already in nm
            flux    = tbl['FLUX_NOR'].astype(float)
            err     = tbl['FLUX_ERR'].astype(float)
            bad = ~((err > 0) & np.isfinite(err))
            if bad.any():
                err[bad] = np.nanmedian(np.abs(flux)) / 200.0

        else:
            raise ValueError(f"_read_gbs_fits: unknown format '{fmt}'")

    # Sort by wavelength (NARVAL may be reverse-ordered)
    order = np.argsort(wave_nm)
    wave_nm, flux, err = wave_nm[order], flux[order], err[order]

    spec = np.zeros(len(wave_nm), dtype=dtype)
    spec['waveobs'] = wave_nm
    spec['flux']    = flux
    spec['err']     = err
    return spec


def _prepare_gbs_spectrum(star_key):
    """
    Load, degrade, cut, RV-correct and normalise a GBS spectrum.

    Returns the normalised spectrum ready for synthesis, or None on failure.
    Resolution degradation uses ispec.convolve_spectrum (Gaussian convolution).
    RV is measured from the G2 CCF mask and corrected unconditionally —
    GBS spectra may or may not be pre-corrected depending on the source.
    """
    gbs = GBS_CATALOG[star_key]
    path = gbs['spectrum']
    from_res = gbs['resolution']
    fmt = gbs.get('format', 'ispec_text')

    print(colored(f"\n  Loading {star_key}: {os.path.basename(path)}", 'yellow'))
    if not os.path.exists(path):
        print(colored(f"    ERROR: file not found: {path}", 'red'))
        return None

    if fmt == 'ispec_text':
        spec = ispec.read_spectrum(path)
    else:
        spec = _read_gbs_fits(path, fmt)
    print(f"    Pixels: {len(spec)}  "
          f"λ = {spec['waveobs'].min()*10:.0f}–{spec['waveobs'].max()*10:.0f} Å")

    # ── Degrade resolution → CARMENES ────────────────────────────────────────
    if from_res > CARMENES_RESOLUTION:
        print(colored(f"    Degrading R: {from_res} → {CARMENES_RESOLUTION}...", 'yellow'))
        spec = ispec.convolve_spectrum(
            spec, to_resolution=CARMENES_RESOLUTION, from_resolution=from_res)

    # ── Cut to synthesis windows ─────────────────────────────────────────────
    spec, _, n_pix = apply_synth_windows(spec)
    print(f"    After windows: {n_pix} pixels  "
          f"({spec['waveobs'].min():.1f}–{spec['waveobs'].max():.1f} nm, "
          f"{len(SYNTH_WINDOWS_NM)} windows)")

    # ── Ensure valid error column ─────────────────────────────────────────────
    if not np.any((spec['err'] > 0) & np.isfinite(spec['err'])):
        err = _estimate_local_noise(spec['flux'], window=201)
        floor = np.nanmedian(np.abs(spec['flux'])) / 500.0
        spec['err'] = np.where((err > 0) & np.isfinite(err), err, floor)

    snr = compute_snr_in_window(spec)
    print(f"    Median S/N : {snr:.0f}")

    # ── RV correction (always applied for GBS spectra) ────────────────────────
    try:
        rv, e_rv = estimate_rv_ccf(spec)
        print(colored(f"    RV = {rv:+.2f} ± {e_rv:.2f} km/s  → correcting...", 'cyan'))
        spec = ispec.correct_velocity(spec, rv)
    except Exception as exc:
        print(colored(f"    WARNING: RV correction failed ({exc}). Proceeding uncorrected.", 'red'))

    # ── Normalise continuum (same method as science targets) ──────────────────
    if NORMALIZE_CONTINUUM and CONTINUUM_MODE != 'none':
        print(colored(f"    Normalising continuum ({CONTINUUM_MODE})...", 'yellow'))
        try:
            spec, _ = normalize_spectrum_spline(spec, mode=CONTINUUM_MODE)
        except Exception as exc:
            print(colored(f"    WARNING: normalisation failed ({exc}).", 'red'))
    else:
        print(colored(f"    Continuum normalisation skipped ({CONTINUUM_MODE}).", 'magenta'))

    return spec


def _fit_gbs_star(star_key, spec, atomic_ll, isotopes, layers_pack, solar_abund):
    """
    Run the same anchored pipeline as science targets on a GBS spectrum.

    Same scheme as _run_parameter_fit in main():
      Step 0: Teff + logg fixed (reference) → fit [M/H] (anchored)
      Step 1: Teff free, logg fixed, [M/H] fixed → fit Teff, vsini

    Returns (delta_mh, params, errors, stats).
    delta_mh = [M/H]_anchored − [M/H]_literature
    """
    gbs = GBS_CATALOG[star_key]
    teff_ref = gbs['teff']
    logg_ref = gbs['logg']
    mh_ref   = gbs['MH']
    vmic_ref = gbs['vmic']

    # vmic is fixed at the K-dwarf pipeline value; vmac is fixed at 0.0 km/s.
    vmic_c, _, _, _ = vmic_zone_strategy(teff_ref, logg_ref)
    vmac_c = vmac_zone_anchor(teff_ref)

    vsini_ref   = gbs.get('vsini', None)   # fixed if catalogue provides it
    vsini_seed  = vsini_ref if vsini_ref is not None else 2.0

    seed = {
        "teff"                 : teff_ref,
        "logg"                 : logg_ref,
        "MH"                   : mh_ref,
        "alpha"                : ispec.determine_abundance_enchancements(mh_ref),
        "vmic"                 : vmic_c,
        "vmac"                 : vmac_c,
        "vsini"                : vsini_seed,
        "limb_darkening_coeff" : 0.6,
        "R"                    : CARMENES_RESOLUTION, # match degraded GBS spectrum resolution
        "vrad"                 : 0.0,
    }

    # --- Step 0: Anchored [M/H] (Teff + logg fixed; vmac/vmic fixed) ---
    # vsini is fixed when the catalogue supplies a reference value (e.g. 61CygA=1.5 km/s)
    # to avoid broadening degeneracies at low vsini.
    anchor_free = ["MH"] if vsini_ref is not None else ["MH", "vsini"]

    print(colored(
        f"    Step 0 — Anchored [M/H]: Teff={teff_ref:.0f} K (fixed)  "
        f"logg={logg_ref:.3f} (fixed)  [M/H]={mh_ref:+.3f} (seed)  "
        f"vmac={vmac_c:.2f} (fixed)  vmic={vmic_c:.2f} (fixed)  "
        f"free: {', '.join(anchor_free)}", 'cyan'))

    anch_params, anch_errors, _, _, _ = estimate_stellar_parameters(
        spec, seed, anchor_free,
        atomic_ll, isotopes, layers_pack, solar_abund,
        max_iterations=20)

    anchored_MH = anch_params['MH']
    print(colored(
        f"    → [M/H] = {anchored_MH:+.4f} ± {anch_errors['MH']:.4f}  "
        f"(ref: {mh_ref:+.3f}, δ = {anchored_MH - mh_ref:+.4f} dex)",
        'green'))

    # --- Step 1: Main fit (Teff free, logg fixed, [M/H] fixed, vmac/vmic fixed) ---
    seed1 = dict(seed)
    seed1['MH']    = anchored_MH
    seed1['alpha'] = ispec.determine_abundance_enchancements(anchored_MH)
    seed1['vsini'] = vsini_ref if vsini_ref is not None else anch_params['vsini']

    # logg fixed, [M/H] fixed, vmac/vmic fixed; vsini follows the GBS reference.
    main_free = ["teff"] if vsini_ref is not None else ["teff", "vsini"]

    print(colored(
        f"    Step 1 — Main fit: Teff={teff_ref:.0f} K (seed, free)  "
        f"logg={logg_ref:.2f} (fixed)  [M/H]={anchored_MH:+.3f} (fixed)  "
        f"free: {', '.join(main_free)}", 'cyan'))

    params, errors, status, stats, _ = estimate_stellar_parameters(
        spec, seed1, main_free,
        atomic_ll, isotopes, layers_pack, solar_abund,
        max_iterations=20)

    # Restore anchored [M/H] into results
    params['MH'] = anchored_MH
    errors['MH'] = anch_errors['MH']

    # Error inflation
    if AUTO_RESCALE_ERRORS:
        med_sig = float(np.nanmedian(spec['err']))
        rms_val = float(np.nanmedian(np.abs(
            spec['flux'] - np.nanmedian(spec['flux']))))
        if np.isfinite(rms_val) and med_sig > 0:
            scale = max(rms_val / med_sig, 1.0)
            for key in errors:
                errors[key] *= scale

    delta_mh = float(anchored_MH) - mh_ref
    teff_delta = params['teff'] - teff_ref

    print(colored(
        f"\n    ── {star_key} results ──\n"
        f"    Teff  = {params['teff']:.4f} ± {errors['teff']:.4f} K"
        f"  (ref: {teff_ref:.0f}, Δ = {teff_delta:+.0f} K)\n"
        f"    logg  = {logg_ref:.4f} (fixed)\n"
        f"    [M/H] = {anchored_MH:.4f} ± {errors['MH']:.4f}"
        f"  (ref: {mh_ref:+.3f}, δ = {delta_mh:+.4f} dex)\n"
        f"    vmic  = {params['vmic']:.4f} ± {errors.get('vmic', 0):.4f} km/s\n"
        f"    vsini = {params['vsini']:.4f} ± {errors.get('vsini', 0):.4f} km/s",
        'cyan'))

    return delta_mh, params, errors, stats
def run_gbs_test(star_filter=None):
    """
    Pipeline validation: run the full synthesis on GBS stars and print
    a comparison table (fitted vs. reference parameters).

    Parameters
    ----------
    star_filter : list of str or None
        If given, only process these GBS_CATALOG keys.
        If None, process all stars.
    """
    star_keys = list(GBS_CATALOG.keys())
    if star_filter:
        unknown = [s for s in star_filter if s not in GBS_CATALOG]
        if unknown:
            print(colored(
                f"  ERROR: unknown GBS star(s): {', '.join(unknown)}\n"
                f"  Available: {', '.join(star_keys)}", 'red'))
            return {}
        star_keys = star_filter

    print(colored(f"\n{'='*60}", 'magenta'))
    print(colored("  GBS PIPELINE VALIDATION TEST", 'magenta'))
    print(colored(f"  Stars: {', '.join(star_keys)}", 'magenta'))
    print(colored(f"{'='*60}", 'magenta'))

    # Load iSpec resources
    print(colored("  Loading iSpec atomic data...", 'yellow'))
    atomic_ll   = ispec.read_atomic_linelist(ATOMIC_LINELIST)
    isotopes    = ispec.read_isotope_data(ISOTOPE_FILE)
    layers_pack = ispec.load_modeled_layers_pack(MODEL_ATMOS_DIR)
    solar_abund = ispec.read_solar_abundances(SOLAR_ABUND_FILE)
    print(colored("  Done.", 'green'))

    results = {}

    for star_key in star_keys:
        gbs = GBS_CATALOG[star_key]
        print(colored(f"\n{'─'*60}", 'magenta'))
        print(colored(f"  [{star_key}]  "
                      f"Teff_ref={gbs['teff']:.0f} K  "
                      f"logg_ref={gbs['logg']:.2f}  "
                      f"[M/H]_ref={gbs['MH']:+.2f}  "
                      f"({gbs['source']})", 'magenta'))

        spec = _prepare_gbs_spectrum(star_key)
        if spec is None:
            print(colored(f"  SKIPPING {star_key} — spectrum unavailable.", 'red'))
            continue

        delta_mh, params, errors, _stats = _fit_gbs_star(
            star_key, spec, atomic_ll, isotopes, layers_pack, solar_abund)

        results[star_key] = {
            'teff_ref' : gbs['teff'],
            'teff_fit' : params['teff'],
            'e_teff'   : errors.get('teff', np.nan),
            'logg_ref' : gbs['logg'],
            'mh_ref'   : gbs['MH'],
            'mh_fit'   : params['MH'],
            'e_mh'     : errors.get('MH', np.nan),
            'vmic_ref' : gbs.get('vmic', np.nan),
            'vmic_fit' : params.get('vmic', np.nan),
            'vsini_fit': params.get('vsini', np.nan),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(colored(f"\n{'='*72}", 'green'))
    print(colored("  GBS VALIDATION SUMMARY", 'green'))
    print(colored(f"{'='*72}", 'green'))
    print(colored(
        f"  {'Star':<10} {'Teff_ref':>8} {'Teff_fit':>8} {'ΔTeff':>7} "
        f"{'[M/H]_ref':>9} {'[M/H]_fit':>9} {'Δ[M/H]':>7} "
        f"{'vmic':>5}", 'cyan'))
    print(colored(f"  {'─'*68}", 'cyan'))

    delta_teffs = []
    delta_mhs   = []
    for k, r in results.items():
        dt = r['teff_fit'] - r['teff_ref']
        dm = r['mh_fit']   - r['mh_ref']
        delta_teffs.append(dt)
        delta_mhs.append(dm)
        print(colored(
            f"  {k:<10} {r['teff_ref']:8.0f} {r['teff_fit']:8.0f} {dt:+7.0f} "
            f"{r['mh_ref']:+9.2f} {r['mh_fit']:+9.3f} {dm:+7.3f} "
            f"{r['vmic_fit']:5.2f}", 'cyan'))

    if delta_teffs:
        _mean_dt  = np.mean(delta_teffs)
        _std_dt   = np.std(delta_teffs)
        _mean_dm  = np.mean(delta_mhs)
        _std_dm   = np.std(delta_mhs)
        _mad_dt   = np.mean(np.abs(delta_teffs))
        _mad_dm   = np.mean(np.abs(delta_mhs))
        print(colored(f"  {'─'*68}", 'cyan'))
        print(colored(
            f"  {'Mean':10} {'':>8} {'':>8} {_mean_dt:+7.0f} "
            f"{'':>9} {'':>9} {_mean_dm:+7.3f}", 'green'))
        print(colored(
            f"  {'Std':10} {'':>8} {'':>8} {_std_dt:7.0f} "
            f"{'':>9} {'':>9} {_std_dm:7.3f}", 'green'))
        print(colored(
            f"  {'MAD':10} {'':>8} {'':>8} {_mad_dt:7.0f} "
            f"{'':>9} {'':>9} {_mad_dm:7.3f}", 'green'))

    print(colored(f"{'='*72}\n", 'green'))
    return results


def _extract_stat_scalar(stats_obj, keys=("rms_residual", "rms", "chi2", "reduced_chi2")):
    """Extract a first finite scalar from an iSpec stats object."""
    for key in keys:
        val = None
        try:
            if hasattr(stats_obj, "keys") and key in stats_obj:
                val = stats_obj[key]
        except Exception:
            pass
        if val is None:
            try:
                if hasattr(stats_obj, key):
                    val = getattr(stats_obj, key)
            except Exception:
                pass
        if val is None:
            continue
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size > 0 and np.isfinite(arr[0]):
            return float(arr[0])
    return np.nan


def run_gbs_fe2_line_test(star_filter=None):
    """
    Validate Fe I, all Fe II, and individual Fe II lines on GBS spectra.

    This is the GBS analogue of --fe2-line-test for KOBE targets. Results are
    printed only, because GBS rows use literature references rather than the
    science-target cumulative CSV schema.
    """
    global LINE_SUBSET_MODE

    star_keys = list(GBS_CATALOG.keys())
    if star_filter:
        unknown = [s for s in star_filter if s not in GBS_CATALOG]
        if unknown:
            print(colored(
                f"  ERROR: unknown GBS star(s): {', '.join(unknown)}\n"
                f"  Available: {', '.join(star_keys)}", 'red'))
            return 1
        star_keys = star_filter

    modes = ('fe1', 'fe2', *FE2_LINE_TEST_MODES)
    print(colored(f"\n{'='*72}", 'magenta'))
    print(colored("  GBS FE II LINE-BY-LINE DIAGNOSTIC", 'magenta'))
    print(colored(f"  Stars   : {', '.join(star_keys)}", 'magenta'))
    print(colored(f"  Subsets : {', '.join(modes)}", 'magenta'))
    print(colored(f"{'='*72}", 'magenta'))

    print(colored("  Loading iSpec atomic data...", 'yellow'))
    atomic_ll   = ispec.read_atomic_linelist(ATOMIC_LINELIST)
    isotopes    = ispec.read_isotope_data(ISOTOPE_FILE)
    layers_pack = ispec.load_modeled_layers_pack(MODEL_ATMOS_DIR)
    solar_abund = ispec.read_solar_abundances(SOLAR_ABUND_FILE)
    print(colored("  Done.", 'green'))

    previous_mode = LINE_SUBSET_MODE
    rows = []
    n_err = 0
    try:
        for star_key in star_keys:
            spec = _prepare_gbs_spectrum(star_key)
            if spec is None:
                n_err += 1
                continue

            for mode in modes:
                LINE_SUBSET_MODE = mode
                print(colored(
                    f"\n[gbs-fe2-line] {star_key} mode={mode}",
                    'yellow'))
                try:
                    _delta, params, errors, stats = _fit_gbs_star(
                        star_key, spec,
                        atomic_ll, isotopes, layers_pack, solar_abund)
                    rows.append({
                        'star': star_key,
                        'mode': mode,
                        'n_lines': LAST_LINE_REGION_COUNT,
                        'mh_ref': GBS_CATALOG[star_key]['MH'],
                        'mh': params.get('MH', np.nan),
                        'e_mh': errors.get('MH', np.nan),
                        'teff': params.get('teff', np.nan),
                        'fit_score': _extract_stat_scalar(stats),
                    })
                except Exception as exc:
                    n_err += 1
                    print(colored(
                        f"  ERROR: {star_key} {mode} failed: {exc}",
                        'red'))
    finally:
        LINE_SUBSET_MODE = previous_mode

    _print_fe2_line_metadata()

    print(colored(f"\n{'='*104}", 'green'))
    print(colored(
        "  GBS FE II LINE-BY-LINE SUMMARY  (Δline = [Fe/H]line - [Fe/H]FeI)",
        'green'))
    print(colored(f"{'='*104}", 'green'))
    print(colored(
        f"  {'Star':<10} {'Line':<11} {'N':>3} {'Ref':>8} "
        f"{'FeI':>8} {'FeII_all':>9} {'FeII_line':>10} "
        f"{'Δline':>8} {'Δref':>8} {'Teff':>7} {'χ²/RMS':>8}",
        'cyan'))
    print(colored(f"  {'─'*100}", 'cyan'))

    by_key = {(r['star'], r['mode']): r for r in rows}
    for star_key in star_keys:
        fe1 = by_key.get((star_key, 'fe1'))
        fe2 = by_key.get((star_key, 'fe2'))
        feh1 = fe1['mh'] if fe1 else np.nan
        feh2 = fe2['mh'] if fe2 else np.nan
        mh_ref = GBS_CATALOG[star_key]['MH']
        for mode in FE2_LINE_TEST_MODES:
            row = by_key.get((star_key, mode))
            if row is None:
                print(colored(
                    f"  {star_key:<10} {_format_fe2_line_label(mode):<11} missing",
                    'yellow'))
                continue
            dline = row['mh'] - feh1 if np.isfinite(feh1) else np.nan
            dref = row['mh'] - mh_ref if np.isfinite(row['mh']) else np.nan
            n_str = (f"{int(row['n_lines'])}"
                     if np.isfinite(row['n_lines']) else '')
            print(colored(
                f"  {star_key:<10} {_format_fe2_line_label(mode):<11} "
                f"{n_str:>3} {mh_ref:8.3f} "
                f"{feh1:8.3f} {feh2:9.3f} {row['mh']:10.3f} "
                f"{dline:8.3f} {dref:8.3f} "
                f"{row['teff']:7.0f} {row['fit_score']:8.4f}",
                'cyan'))

    return n_err




# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(target_name, diagnostic_plots=False, output_dir='.'):
    fits_file = os.path.join(SPECTRA_DIR, f"{target_name}_merged.fits")
    if not os.path.exists(fits_file):
        print(colored(f"ERROR: File not found: {fits_file}", 'red'))
        sys.exit(1)

    print(colored(f"\n{'='*60}", 'cyan'))
    print(colored(f"  Target : {target_name}", 'cyan'))
    print(colored(f"  File   : {fits_file}", 'cyan'))
    print(colored(f"  Mask   : {os.path.basename(os.path.dirname(MASK_FILE))}", 'cyan'))
    print(colored(f"  R      : {CARMENES_RESOLUTION}", 'cyan'))
    print(colored(
        f"  logg   : physical — M_cat + SB(L_cat, Teff)  "
        f"[logg_init from Teff_cat; logg_final from Teff_fit]", 'cyan'))
    print(colored(f"{'='*60}", 'cyan'))

    # ── Chromospheric activity flag ────────────────────────────────────────────
    # Use catalog Teff seed for the Teff-dependent threshold (Astudillo-Defru+17).
    _cat = _load_kobe_catalog()
    _teff_seed = _cat.get(target_name, {}).get('Teff_K', None)
    logRHK, is_active, _act_thresh, _act_bv = get_activity_flag(target_name, teff=_teff_seed)
    if logRHK is not None:
        flag_str = "weak activity flag" if is_active else "inactive"
        color = 'yellow' if is_active else 'green'
        _bv_str = f", B-V={_act_bv:.3f}" if _act_bv is not None else " (fallback)"
        print(colored(
            f"  log R'HK = {logRHK:.2f}  ({flag_str}; "
            f"Teff-dep. threshold {_act_thresh:.2f}{_bv_str})",
            color))
        if is_active:
            print(colored(
                "  NOTE: catalog activity is treated as a weak, non-contemporaneous flag.",
                'yellow'))

    # ------------------------------------------------------------------
    # 1) Read full spectrum
    # ------------------------------------------------------------------
    print(colored(f"\n[1] Reading spectrum...", 'yellow'))
    spectrum = read_cafe_spectrum(fits_file)

    # Optional sanity checks (after reading spectrum)
    if diagnostic_plots:
        print(colored("\n  [Diagnostic] RV check...", 'yellow'))
        diagnostic_check_rv(spectrum, target_name=target_name, output_dir=output_dir)
        print(colored("  [Diagnostic] Mask overlay...", 'yellow'))
        check_mask_overlay(target_name, output_dir=output_dir)

    snr_est = compute_snr_in_window(spectrum)
    snr_window_a = compute_snr_in_window(spectrum, 562.0, 686.0)
    print(f"    Total pixels : {len(spectrum)}")
    print(f"    λ range      : {spectrum['waveobs'].min()*10:.1f} – "
          f"{spectrum['waveobs'].max()*10:.1f} Å")
    print(f"    Median flux  : {np.nanmedian(spectrum['flux']):.4f}")
    print(f"    Median σ     : {np.nanmedian(spectrum['err']):.5f}  → S/N ≈ {snr_est:.0f}  "
          f"(window A 562-686 nm: S/N ≈ {snr_window_a:.0f})")

    # ------------------------------------------------------------------
    # 2) Clean telluric contamination
    # ------------------------------------------------------------------
    if CLEAN_TELLURIC:
        print(colored(f"\n[2] Cleaning telluric regions (full spectrum)...", 'yellow'))
        spectrum, bv, bv_err = clean_telluric_regions(spectrum)
        print(f"    Telluric BV : {bv:.4f} ± {bv_err:.4f} km/s")
        print(f"    Pixels after telluric clean : {len(spectrum)}")
    else:
        print(colored(f"\n[2] Telluric cleaning skipped.", 'magenta'))

    # ------------------------------------------------------------------
    # 3) Apply synthesis windows (multi-window, avoids telluric bands)
    # ------------------------------------------------------------------
    win_str = ', '.join(f"{wb:.0f}–{wt:.0f}" for wb, wt in SYNTH_WINDOWS_NM)
    print(colored(f"\n[3] Applying synthesis windows ({len(SYNTH_WINDOWS_NM)} windows)...", 'yellow'))
    print(f"    Windows [nm]: {win_str}")
    spectrum_synth, n_win, n_pix = apply_synth_windows(spectrum)
    print(f"    Pixels in synthesis windows : {n_pix}  "
          f"({spectrum_synth['waveobs'].min()*10:.0f}–"
          f"{spectrum_synth['waveobs'].max()*10:.0f} Å, {n_win} windows)")

    # ------------------------------------------------------------------
    # 4) Normalize continuum (on full multi-window spectrum)
    # ------------------------------------------------------------------
    if NORMALIZE_CONTINUUM and CONTINUUM_MODE != 'none':
        print(colored(f"\n[4] Normalizing continuum ({CONTINUUM_MODE})...", 'yellow'))
        pre_median = np.nanmedian(spectrum_synth['flux'])
        spectrum_synth, _ = normalize_spectrum_spline(spectrum_synth, mode=CONTINUUM_MODE)
        post_median = np.nanmedian(spectrum_synth['flux'])
        snr_post = np.nanmedian(spectrum_synth['flux']) / np.nanmedian(spectrum_synth['err'])
        print(f"    Median flux before : {pre_median:.4f}  →  after : {post_median:.4f}")
        print(f"    Median σ (post)    : {np.nanmedian(spectrum_synth['err']):.5f}  → S/N ≈ {snr_post:.0f}")
    else:
        print(colored(f"\n[4] Continuum normalization skipped ({CONTINUUM_MODE}).", 'magenta'))

    # ------------------------------------------------------------------
    # 4b) Ca II IRT spectral activity check (849.8 + 854.2 nm)
    # ------------------------------------------------------------------
    _irt_active, _irt_depths, _irt_n = measure_caii_irt_filling(spectrum_synth)
    if _irt_n == 0:
        print(colored("  Ca II IRT : lines out of range — spectral activity flag unavailable",
                      'yellow'))
        is_active_spectral = None
    else:
        is_active_spectral = _irt_active
        _irt_lines_str = "  ".join(
            f"{lam:.3f} nm → core={d:.3f}" for lam, d in _irt_depths.items()
        )
        flag_str  = "ACTIVE ⚠ (Ca II IRT filling)" if _irt_active else "inactive"
        flag_col  = 'red' if _irt_active else 'green'
        print(colored(f"  Ca II IRT : {flag_str}", flag_col))
        print(f"    {_irt_lines_str}")
        if _irt_active and not is_active:
            print(colored(
                "  NOTE: spectral Ca II IRT flags activity not seen in catalog log R'HK "
                "— epoch mismatch or catalog uncertainty.", 'yellow'))
        elif not _irt_active and is_active:
            print(colored(
                "  NOTE: catalog log R'HK flags activity but Ca II IRT cores appear normal "
                "— may be quiescent at this epoch.", 'yellow'))

    # ------------------------------------------------------------------
    # 5) Measure RV residual (CCF on blue window only — mask limit 680 nm)
    # ------------------------------------------------------------------
    print(colored(f"\n[5] Measuring RV residual via CCF (K5 mask, ≤680 nm)...", 'yellow'))
    try:
        spectrum_rv = apply_rv_window(spectrum_synth)
        rv, e_rv = estimate_rv_ccf(spectrum_rv)
        rv_flag = abs(rv) > RV_WARNING_KMS
        rv_color = 'red' if rv_flag else 'cyan'
        print(colored(f"    RV = {rv:.4f} ± {e_rv:.4f} km/s  "
                      f"(CCF on {len(spectrum_rv)} px, 562–680 nm)", rv_color))

        if ESTIMATE_RV:
            spectrum_synth = ispec.correct_velocity(spectrum_synth, rv)

    except Exception as exc:
        print(colored(f"    CCF failed: {exc}", 'red'))
        rv, e_rv = np.nan, np.nan

    spectrum = spectrum_synth   # from here on 'spectrum' is the multi-window array

    # ------------------------------------------------------------------
    # 6) Parameter estimation
    # ------------------------------------------------------------------
    if ESTIMATE_PARAMS:

        max_iter = 20

        print(colored(f"\n[6] Loading iSpec atomic data...", 'yellow'))
        atomic_ll   = ispec.read_atomic_linelist(ATOMIC_LINELIST)
        isotopes    = ispec.read_isotope_data(ISOTOPE_FILE)
        layers_pack = ispec.load_modeled_layers_pack(MODEL_ATMOS_DIR)
        solar_abund = ispec.read_solar_abundances(SOLAR_ABUND_FILE)

        print(colored(f"    Done.", 'green'))

        print(colored(f"\n[7] Fitting stellar parameters...", 'yellow'))

        def _extract_fit_score(stats_obj):
            """Return a scalar goodness-of-fit score (lower is better) or NaN."""
            keys = ["chi2", "reduced_chi2", "chi_square", "rms", "rms_residual"]

            for key in keys:
                val = None
                try:
                    if hasattr(stats_obj, "keys") and key in stats_obj:
                        val = stats_obj[key]
                except Exception:
                    pass
                if val is None:
                    try:
                        if hasattr(stats_obj, key):
                            val = getattr(stats_obj, key)
                    except Exception:
                        pass
                if val is None:
                    continue

                arr = np.asarray(val, dtype=float).ravel()
                if arr.size > 0 and np.isfinite(arr[0]):
                    return float(arr[0])
            return np.nan

        def _extract_stat_value(stats_obj, keys):
            """Extract first finite scalar for any key in stats_obj."""
            for key in keys:
                val = None
                try:
                    if hasattr(stats_obj, "keys") and key in stats_obj:
                        val = stats_obj[key]
                except Exception:
                    pass
                if val is None:
                    try:
                        if hasattr(stats_obj, key):
                            val = getattr(stats_obj, key)
                    except Exception:
                        pass
                if val is None:
                    continue
                arr = np.asarray(val, dtype=float).ravel()
                if arr.size > 0 and np.isfinite(arr[0]):
                    return float(arr[0])
            return np.nan


        _W = 60  # section banner width

        def _banner(title, color='blue'):
            print(colored(f"\n{'─'*_W}", color))
            print(colored(f"  {title}", color))
            print(colored(f"{'─'*_W}", color))

        def _run_parameter_fit(current_spectrum):
            """
            Full parameter-fit pipeline for K5–M0 V dwarfs:
              -1. Pre-fit vsini_CCF measurement (using catalog seeds). Drives the
                  hard-fixed vsini anchor downstream.
               0. Anchored [M/H] pass (Teff+logg fixed from catalog; vmac fixed;
                  vmic fixed; vsini hard-fixed).
               1. Main fit (MH fixed from step 0; Teff free; logg derived;
                  vmac/vmic/vsini fixed).
               2. Physical logg — single computation (no iteration).
               3. Error inflation (sqrt chi²_red) + systematic floors.
            """
            _catalog_seeds = get_catalog_seeds(target_name)
            _seed_teff = (_catalog_seeds['teff'] if _catalog_seeds is not None
                          else INITIAL_TEFF)

            # -------------------------------------------------------------
            # -1. Pre-fit vsini_CCF using catalog seeds (or defaults)
            # -------------------------------------------------------------
            _pre_vsini_ccf   = np.nan
            _pre_e_vsini_ccf = np.nan
            _pre_ccf_info    = {}

            if _catalog_seeds is not None:
                _pre_params = build_initial_parameters(
                    _catalog_seeds['teff'],
                    INITIAL_LOGG,
                    _catalog_seeds['mh'])
            else:
                _pre_params = dict(INITIAL_PARAMETERS)

            if MEASURE_VSINI_CCF:
                _banner("Pre-fit vsini_CCF  (drives hard-fixed vsini anchor)", 'yellow')
                try:
                    _pre_vsini_ccf, _pre_e_vsini_ccf, _pre_ccf_info = \
                        estimate_vsini_ccf(
                            current_spectrum, _pre_params,
                            layers_pack, atomic_ll, isotopes, solar_abund,
                            velocity_step=0.5)
                    _e_str = (f" ± {_pre_e_vsini_ccf:.3f}"
                              if np.isfinite(_pre_e_vsini_ccf) else "")
                    if np.isfinite(_pre_vsini_ccf):
                        print(colored(
                            f"  vsini_CCF = {_pre_vsini_ccf:.3f}{_e_str} km/s  "
                            f"(seed Teff={_pre_params['teff']:.0f} K)", 'cyan'))
                    else:
                        print(colored(
                            "  vsini_CCF unresolved — fit will run with vsini free.",
                            'yellow'))
                except Exception as _exc:
                    print(colored(
                        f"  Pre-fit vsini_CCF failed ({_exc}); "
                        f"fit will run with vsini free.", 'yellow'))

            # vsini strategy (Marfil+22 KOBE convention): vsini is ALWAYS
            # hard-fixed — never a free parameter.
            #   · vsini_CCF > VSINI_FAST_THRESHOLD_KMS → fast rotator,
            #     fix at the measured vsini_CCF (real rotation, e.g.
            #     KOBE-004 at 7.7 km/s).
            #   · otherwise → fix at VSINI_SLOW_FIXED_KMS (2.0 km/s),
            #     a flat empirical anchor that absorbs vmac in quadrature.
            #
            # Rationale: at R = 94 600, vsini at the few-km/s level is
            # unresolvable and degenerate with vmic via the curve of growth
            # (this work: dvsini/dvmic = -1.33 km/s per (km/s) with σ=0.15
            # across 26 targets).  Letting vsini float channels vmic
            # variations into [Fe/H].  KOBE-004 (vsini_CCF=7.7 km/s,
            # rotation-pinned) shows Δ[Fe/H]/Δvmic at half the typical
            # value — direct empirical confirmation that fixing vsini
            # collapses the systematic.
            _vsini_hard_fix = True   # always
            if (np.isfinite(_pre_vsini_ccf) and
                    _pre_vsini_ccf > VSINI_FAST_THRESHOLD_KMS):
                _vsini_init = float(_pre_vsini_ccf)
                _fix_reason = (f"fast rotator: CCF > "
                               f"{VSINI_FAST_THRESHOLD_KMS:.1f} km/s")
            else:
                _vsini_init = VSINI_SLOW_FIXED_KMS
                _fix_reason = (f"slow rotator: flat anchor "
                               f"{VSINI_SLOW_FIXED_KMS:.1f} km/s "
                               f"(CCF={_pre_vsini_ccf:.2f} km/s)"
                               if np.isfinite(_pre_vsini_ccf)
                               else f"flat anchor {VSINI_SLOW_FIXED_KMS:.1f} km/s "
                                    f"(CCF unavailable)")
            print(colored(
                f"  vsini strategy: HARD-FIX at {_vsini_init:.3f} km/s "
                f"({_fix_reason})", 'cyan'))

            def _strip_vsini(free_list):
                """Remove 'vsini' from a free-params list (always hard-fixed)."""
                return [p for p in free_list if p != "vsini"]

            # -------------------------------------------------------------
            # 0. Anchored [M/H] pass: Teff + logg fixed → fit [M/H]
            # -------------------------------------------------------------
            # The catalog seeds (SteParSyn) provide an independent Teff and
            # logg.  Fixing both breaks the Teff–[M/H] degeneracy and yields
            # a cleaner metallicity, especially for active late-K dwarfs.
            # vmac is hard-fixed at 0.0 km/s throughout.
            _anchored_MH = None
            _anch_errors = {}
            if _catalog_seeds is not None:
                _anchor_teff = _catalog_seeds['teff']
                _anchor_logg = INITIAL_LOGG
                _anchor_mh   = _catalog_seeds['mh']

                _anchor_seed = build_initial_parameters(
                    _anchor_teff, _anchor_logg, _anchor_mh)
                _anchor_seed['vsini'] = _vsini_init

                _, _vmic_lo, _vmic_hi, _vmic_fixed = vmic_zone_strategy(
                    _anchor_teff, _anchor_logg)

                # Step 0 free list: MH only. vmic/vmac/vsini are fixed.
                _anchor_free = _strip_vsini(["MH", "vsini"])

                _banner("Anchored [M/H] pass  —  Teff + logg fixed (catalog)", 'magenta')
                print(colored(
                    f"  Teff={_anchor_teff:.0f} K (fixed)  "
                    f"logg={_anchor_logg:.2f} (fixed)  "
                    f"[M/H]={_anchor_mh:+.2f} (seed)  "
                    f"vmac={_anchor_seed['vmac']:.2f} (fixed)  "
                    f"vmic={_anchor_seed['vmic']:.2f} (fixed)  "
                    f"vsini={_vsini_init:.2f} "
                    f"({'fixed' if _vsini_hard_fix else 'init'})  "
                    f"free: {', '.join(_anchor_free)}",
                    'cyan'))

                _anch_params, _anch_errors, _, _, _ = estimate_stellar_parameters(
                    current_spectrum,
                    _anchor_seed,
                    _anchor_free,
                    atomic_ll, isotopes, layers_pack, solar_abund,
                    max_iterations=max_iter,
                    vmic_range=(_vmic_lo, _vmic_hi),
                )
                _anchored_MH = _anch_params['MH']
                print(colored(
                    f"  → [M/H] = {_anchored_MH:+.3f} ± {_anch_errors['MH']:.3f}  "
                    f"(anchored; Teff={_anchor_teff:.0f} K, logg={_anchor_logg:.2f})",
                    'green'))
            else:
                print(colored(
                    "  No catalog seeds available — skipping anchored [M/H] pass.",
                    'yellow'))

            # -------------------------------------------------------------
            # 1. Main fit (Teff with anchored [M/H], vmac/vmic/vsini fixed;
            #    logg is derived physically after the fit)
            # -------------------------------------------------------------
            # Rebuild INITIAL_PARAMETERS from catalog seed so fixed anchors
            # use the catalog Teff (not the global default).
            _main_initial = build_initial_parameters(
                _seed_teff,
                INITIAL_LOGG,
                (_catalog_seeds['mh'] if _catalog_seeds is not None
                 else INITIAL_MH))
            _main_initial['vsini'] = _vsini_init

            # Fixed vmic strategy for main fit.
            _, _vmic_lo, _vmic_hi, _vmic_fixed = vmic_zone_strategy(
                _seed_teff, INITIAL_LOGG)

            # FREE_PARAMS sets the baseline; remove MH if anchored and
            # always remove vsini because it is hard-fixed.
            _main_free = list(FREE_PARAMS)
            if _anchored_MH is not None:
                if "MH" in _main_free:
                    _main_free.remove("MH")
                _main_initial['MH'] = _anchored_MH
                _main_initial['alpha'] = ispec.determine_abundance_enchancements(
                    _anchored_MH)
            _main_free = _strip_vsini(_main_free)

            _banner("Main fit  —  Teff fit, logg derived, [M/H] anchored", 'blue')
            _mh_tag = f"[M/H]={_main_initial['MH']:+.2f} (anchored, fixed)" \
                      if _anchored_MH is not None \
                      else f"[M/H]={_main_initial['MH']:+.2f}"
            _vmic_tag = f"vmic={_main_initial['vmic']:.2f} (fixed)"
            print(colored(
                f"  Teff={_main_initial['teff']:.0f} K  "
                f"logg={_main_initial['logg']:.2f}  "
                f"{_mh_tag}  "
                f"vmac={_main_initial['vmac']:.2f} (fixed)  "
                f"{_vmic_tag}  "
                f"vsini={_main_initial['vsini']:.2f} "
                f"({'fixed' if _vsini_hard_fix else 'free'})  "
                f"free: {', '.join(_main_free)}",
                'cyan'))
            params, errors, status, stats, sparse_synth = estimate_stellar_parameters(
                current_spectrum,
                _main_initial,
                _main_free,
                atomic_ll, isotopes, layers_pack, solar_abund,
                max_iterations=max_iter,
                vmic_range=(_vmic_lo, _vmic_hi),
            )
            # Propagate fixed values into results (iSpec omits them from errors).
            if _vsini_hard_fix:
                params['vsini'] = _vsini_init
                errors['vsini'] = float(_pre_e_vsini_ccf) \
                    if np.isfinite(_pre_e_vsini_ccf) else 0.5
            # vmac is always fixed: ensure errors reflect this (0 from fit).
            errors.setdefault('vmac', 0.0)
            if _anchored_MH is not None:
                params['MH'] = _anchored_MH
                errors['MH'] = _anch_errors['MH']
            print(colored(
                f"  → Teff={params['teff']:.0f} K  logg={params['logg']:.2f}  "
                f"[M/H]={params['MH']:+.2f}  vmic={params['vmic']:.2f}  "
                f"vsini={params['vsini']:.2f} km/s",
                'green'))

            # --- 5. Inflate formal errors by sqrt(chi2_red) ---
            # Standard correction: if chi2_red > 1, formal covariance matrix
            # underestimates uncertainties.  sigma_param *= sqrt(chi2_red).
            # Spectrum errors are preserved (no rescaling of data).
            if AUTO_RESCALE_ERRORS:
                rms_final = _extract_stat_value(stats, ["rms_residual", "rms"])
                med_sigma_final = float(np.nanmedian(current_spectrum['err']))
                if (np.isfinite(rms_final) and np.isfinite(med_sigma_final)
                        and med_sigma_final > 0 and rms_final > 0):
                    chi2_red_scale = rms_final / med_sigma_final
                else:
                    chi2_red_scale = 1.0

                if chi2_red_scale > 1.0:
                    for key in errors:
                        errors[key] *= chi2_red_scale
                    print(colored(
                        f"\n    Parameter errors inflated by sqrt(chi2_red) = "
                        f"{chi2_red_scale:.3f}  (chi2_red ≈ {chi2_red_scale**2:.2f})",
                        'yellow'))
                else:
                    print(colored(
                        f"\n    sqrt(chi2_red) = {chi2_red_scale:.3f} (≤1, no inflation needed)",
                        'green'))

            # --- 8. Systematic errors in quadrature ---
            # Formal (chi2-inflated) errors measure statistical precision only.
            # Systematic uncertainties from 1D-LTE model limitations
            # (incomplete line lists, TiO/CaH opacities, atmosphere model
            # discretisation) are irreducible regardless of data quality.
            # Combined in quadrature: e_total = sqrt(e_stat² + e_sys²)
            # (Jofré+14, Blanco-Cuaresma+14, Tsantaki+13).
            _SYS_TEFF  = 60.0    # K    — 1D-LTE systematic floor
            _SYS_MH    = 0.05    # dex
            _SYS_VSINI = 0.5     # km/s
            _SYS_VMIC  = 0.10    # km/s
            _raw_errors = dict(errors)   # statistical-only errors, for display
            _applied_floors = {}         # key → sys value added in quadrature
            _floors_applied = []
            for _key, _sys in [('teff', _SYS_TEFF), ('MH', _SYS_MH),
                                ('vsini', _SYS_VSINI), ('vmic', _SYS_VMIC)]:
                if _key in errors:
                    _e_stat = errors[_key]
                    _e_total = np.sqrt(_e_stat ** 2 + _sys ** 2)
                    _applied_floors[_key] = _sys
                    _floors_applied.append(
                        f"{_key}: √({_e_stat:.2f}²+{_sys:.2f}²)={_e_total:.2f}")
                    errors[_key] = _e_total
            if _floors_applied:
                print(colored(
                    f"    Systematic errors added in quadrature: "
                    f"{', '.join(_floors_applied)}",
                    'yellow'))

            return params, errors, status, stats, sparse_synth, current_spectrum, \
                _raw_errors, _applied_floors, \
                _pre_vsini_ccf, _pre_e_vsini_ccf, _pre_ccf_info

        params, errors, status, stats, sparse_synthetic_spectrum, spectrum_final, \
            _raw_errors, _applied_floors, \
            _prefit_vsini_ccf, _prefit_e_vsini_ccf, _prefit_ccf_info = \
            _run_parameter_fit(spectrum)

        # --- Update logg with fitted Teff (R and logg change; M_cat and L_cat fixed) ---
        # logg_init was computed from Teff_cat; now recompute with Teff_fit.
        _banner("logg_final  (M_cat + SB(L_cat, Teff_fit))", 'magenta')
        try:
            logg_final, e_logg_final, radius_final = logg_from_catalog(
                M_CAT, L_CAT, params['teff'], e_teff=errors.get('teff', 60.0), e_L=eL_CAT)
            params['logg'] = logg_final
            errors['logg'] = e_logg_final
            params['_radius'] = radius_final
            _delta_logg = logg_final - INITIAL_LOGG
            print(colored(
                f"  logg_final = {logg_final:.3f} ± {e_logg_final:.3f}  "
                f"(R_final={radius_final:.3f} R☉,  Teff_fit={params['teff']:.0f} K,  "
                f"M_cat={M_CAT:.3f} M☉)  Δlogg={_delta_logg:+.4f} vs init",
                'green'))
            if logg_final < 3.8:
                print(colored(
                    f"  ⚠ logg_final = {logg_final:.4f} < 3.8 — "
                    f"star may be evolved (SGB/RGB). Verify catalog M_cat and L_cat.",
                    'yellow'))
        except Exception as exc:
            print(colored(f"  WARNING: logg_final computation failed: {exc}", 'magenta'))

        # ------------------------------------------------------------
        # Spectrum fit diagnostic
        if diagnostic_plots:
            print(colored(f"\n{'='*60}", 'blue'))
            print(colored("  SPECTRUM FIT DIAGNOSTIC", 'blue'))
            print(colored(f"{'='*60}", 'blue'))

            # Use the re-normalized spectrum (spectrum_final) for the diagnostic,
            # NOT the original spectrum — the fit was optimized against this version.
            if GENERATE_FULL_SYNTH_DIAGNOSTIC:
                print(colored("  Generating full synthetic spectrum for diagnostic...", "yellow"))
                try:
                    synthetic_spectrum = generate_full_synthetic_spectrum(
                        spectrum_final,
                        params,
                        layers_pack,
                        atomic_ll,
                        isotopes,
                        solar_abund,
                    )
                except Exception as exc:
                    print(colored(f"    Full synthesis failed ({exc}). Using model_spectrum synthetic output.", "red"))
                    synthetic_spectrum = sparse_synthetic_spectrum
            else:
                print(colored("  Skipping full synthesis (GENERATE_FULL_SYNTH_DIAGNOSTIC=False); "
                              "using sparse line-region synthesis for plot.", "yellow"))
                synthetic_spectrum = sparse_synthetic_spectrum

            print(colored("  Creating diagnostic plot...", "yellow"))

            check_spectrum_fit(
                spectrum_final,
                synthetic_spectrum,
                output_file=os.path.join(output_dir, f"{target_name}_check_spectrum.pdf")
            )

        # ------------------------------------------------------------------
        # HRD plot with PARSEC isochrones (if luminosity is available)
        # ------------------------------------------------------------------
        if diagnostic_plots and LUMINOSITY is not None:
            print(colored(f"\n  Generating HRD plot with PARSEC isochrones...", 'yellow'))
            try:
                plot_hrd_parsec(
                    target_name,
                    params['teff'], errors.get('teff', 60.0),
                    LUMINOSITY[0], LUMINOSITY[1],
                    mh=params['MH'],
                    output_file=os.path.join(output_dir, f"{target_name}_hrd_parsec.pdf")
                )
            except Exception as exc:
                print(colored(f"    HRD plot failed: {exc}", 'red'))

        # ------------------------------------------------------------------
        # 8) vsini_CCF — use the pre-fit measurement (already performed
        #    inside _run_parameter_fit using catalog seeds).  A second
        #    measurement using fitted params would change vsini_CCF by
        #    < 0.05 km/s for K5–M0 V (negligible), so we avoid the extra
        #    ~5 min cost and keep a single self-consistent value.
        # ------------------------------------------------------------------
        vsini_ccf    = _prefit_vsini_ccf
        e_vsini_ccf  = _prefit_e_vsini_ccf
        ccf_info     = _prefit_ccf_info or {}
        if MEASURE_VSINI_CCF and np.isfinite(vsini_ccf):
            _e_str = f" ± {e_vsini_ccf:.4f}" if np.isfinite(e_vsini_ccf) else ""
            print(colored(
                f"\n[8] vsini_CCF = {vsini_ccf:.4f}{_e_str} km/s  "
                f"(pre-fit; strategy applied: "
                f"FIXED at "
                f"{'CCF' if vsini_ccf > VSINI_FAST_THRESHOLD_KMS else f'{VSINI_SLOW_FIXED_KMS:.1f}'})",
                'cyan'))

        # ------------------------------------------------------------------
        # Print results
        # ------------------------------------------------------------------
        print(colored(f"\n{'='*60}", 'green'))
        print(colored(f"  Results for {target_name}", 'green'))
        print(colored(f"{'='*60}", 'green'))

        # Helper: format raw error with optional floor annotation in parentheses.
        # Shows the pre-floor value; if a floor was applied, appends "(floor X unit)".
        def _fmt_err(key, fmt, unit=''):
            e_total = errors.get(key, np.nan)
            e_stat  = _raw_errors.get(key, e_total)
            s = f"{e_total:{fmt}}{unit}"
            if key in _applied_floors:
                e_sys = _applied_floors[key]
                s += f"  (e_stat={e_stat:{fmt}}{unit}, e_sys={e_sys:.4g}{unit})"
            return s

        # ── Teff (green — derived) ─────────────────────────────────────────
        print(colored(
            f"  Teff   = {params['teff']:.1f} ± {_fmt_err('teff', '.1f', ' K')}",
            'green'))

        # ── logg (green — physical, updated from Teff_fit) ────────────────
        # Collect Radius / Age lines to print at the end (blue — secondary).
        _secondary_lines = []   # list of (text, color) to print after vsini

        _e_logg_str = f" ± {errors['logg']:.3f}" if 'logg' in errors else ""
        _logg_warn  = "  ⚠ logg < 3.8" if params['logg'] < 3.8 else ""
        _logg_color = 'yellow' if params['logg'] < 3.8 else 'green'
        print(colored(
            f"  logg   = {params['logg']:.3f}{_e_logg_str}  "
            f"(M_cat={M_CAT:.3f} M☉,  R_final={params.get('_radius', np.nan):.3f} R☉,  "
            f"logg_init={INITIAL_LOGG:.3f}){_logg_warn}",
            _logg_color))

        # Radius from logg_final computation (already stored in params['_radius'])
        _radius_final = params.get('_radius', np.nan)
        _e_radius_final = (_radius_final * np.sqrt(
            (eL_CAT / (2.0 * L_CAT))**2 +
            (2.0 * errors.get('teff', 60.0) / params['teff'])**2
        ) if (np.isfinite(_radius_final) and L_CAT > 0
              and (eL_CAT > 0 or errors.get('teff', 0) > 0)) else np.nan)
        if np.isfinite(_radius_final):
            _secondary_lines.append((
                f"  Radius = {_radius_final:.3f}"
                f"{f' ± {_e_radius_final:.3f}' if np.isfinite(_e_radius_final) else ''}"
                f" R☉  (SB,  Teff_fit={params['teff']:.0f} K,  L_cat={L_CAT:.4f} L☉)",
                'blue'))
        _secondary_lines.append((
            f"  Mass   = {M_CAT:.3f}"
            f"{f' ± {eM_CAT:.3f}' if np.isfinite(eM_CAT) else ''}"
            f" M☉  (catalog)",
            'blue'))

        # PARSEC age (uses Teff_fit + L_cat + logg_final)
        if LUMINOSITY is not None:
            try:
                mh_for_age = params['MH']
                age_gyr, age_lo, age_hi = age_from_parsec_isochrones(
                    params['teff'], errors.get('teff', 60.0),
                    LUMINOSITY[0], LUMINOSITY[1], mh=mh_for_age,
                    logg=params['logg'])
                if np.isfinite(age_gyr):
                    _secondary_lines.append((
                        f"  Age    = {age_gyr:.1f} Gyr"
                        f"  (range {age_lo:.1f}–{age_hi:.1f} Gyr, PARSEC HRD)",
                        'blue'))
            except Exception as exc:
                _secondary_lines.append((f"    PARSEC age failed: {exc}", 'magenta'))

        # ── [M/H] (green — derived) ────────────────────────────────────────
        mh_raw = params['MH']
        _e_mh_formal = errors.get('MH', np.nan)
        _e_mh_total = (np.sqrt(_e_mh_formal**2 + SYSTEMATIC_MH_FLOOR**2)
                       if np.isfinite(_e_mh_formal) else np.nan)
        print(colored(
            f"  [M/H]  = {mh_raw:.3f} ± {_fmt_err('MH', '.3f', ' dex')}  (formal)",
            'green'))
        if np.isfinite(_e_mh_total):
            print(colored(
                f"           ± {_e_mh_total:.3f} dex  "
                f"(total, ⊕{SYSTEMATIC_MH_FLOOR:.2f} dex systematic floor)",
                'green'))

        # ── vmic ──────────────────────────────────────────────────────────
        _vmic_val = params.get('vmic', np.nan)
        _fitted_teff = params.get('teff', INITIAL_TEFF)
        print(colored(
            f"  vmic   = {_vmic_val:.3f} km/s  (FIXED)",
            'green'))

        # ── vmac (hard-FIXED at 0.0 km/s anchor) ──────────────────────────
        _vmac_val = params.get('vmac', np.nan)
        _vmac_anchor = vmac_zone_anchor(_fitted_teff)
        print(colored(
            f"  vmac   = {_vmac_val:.3f} km/s  "
            f"(FIXED; anchor {_vmac_anchor:.2f} km/s)",
            'green'))

        # ── vsini (blue — unreliable secondary) ───────────────────────────
        print(colored(
            f"  vsini  = {params['vsini']:.2f} ± {_fmt_err('vsini', '.2f', ' km/s')}  (synthesis)",
            'blue'))
        if MEASURE_VSINI_CCF and np.isfinite(vsini_ccf):
            e_str = f" ± {e_vsini_ccf:.2f}" if np.isfinite(e_vsini_ccf) else ""
            print(colored(
                f"  vsini  = {vsini_ccf:.2f}{e_str} km/s  (CCF deconvolution)",
                'blue'))

        # ── Age / Mass / Radius (blue — catalog / secondary) ──────────────
        if _secondary_lines:
            for _line, _col in _secondary_lines:
                print(colored(_line, _col))

        # ── Quality flags & cumulative results CSV ─────────────────────────────
        chi2_red_val = _extract_stat_value(stats, ["rms_residual", "rms"])
        quality_flags = compute_quality_flags(
            snr=snr_est,
            rv_residual_kms=rv,
            chi2_red=chi2_red_val,
            vsini_synth=params.get('vsini', np.nan),
            vsini_ccf=vsini_ccf,
            is_active=is_active,
            teff=params.get('teff', np.nan),
            vmic=params.get('vmic', np.nan),
            vmac=params.get('vmac', np.nan),
            is_active_spectral=is_active_spectral,
            snr_window_a=snr_window_a,
        )
        if quality_flags:
            print(colored(f"  Flags   : {', '.join(quality_flags)}", 'yellow'))

        append_results_csv(
            target=target_name,
            params=params,
            errors=errors,
            quality_flags=quality_flags,
            snr=snr_est,
            rv_residual=rv,
            chi2_red=chi2_red_val,
            radius=_radius_final,
            e_radius=_e_radius_final,
        )

        print(colored(f"{'='*60}\n", 'green'))

    else:
        print(colored(f"\n[6-7] Parameter estimation skipped.", 'magenta'))


# ─────────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────────

# ── Parallel batch worker ─────────────────────────────────────────────────────

def _synthesis_worker(args_tuple):
    """
    Module-level worker for multiprocessing batch runs.

    Receives a tuple (target_name, diagnostic_plots, output_dir), sets all
    per-target globals, then delegates to main().  Returns a 3-tuple:
        (target_name, 'OK'|'ERROR', None|traceback_string)

    Uses fork-based multiprocessing (context='fork') so iSpec / Turbospectrum
    are already loaded in memory — no 30 s re-import cost per worker.
    """
    import traceback as _tb

    target_name, diagnostic_plots, output_dir = args_tuple

    # ── Declare all per-target globals we are about to mutate ────────────────
    global INITIAL_TEFF, INITIAL_LOGG, INITIAL_MH, INITIAL_PARAMETERS
    global FREE_PARAMS
    global M_CAT, eM_CAT, L_CAT, eL_CAT, LUMINOSITY

    # ── Catalog seeds ─────────────────────────────────────────────────────────
    seeds = get_catalog_seeds(target_name)
    if seeds is None:
        print(colored(
            f"[batch] ERROR: {target_name} not found in KOBE catalog — skipped.",
            'red'))
        return (target_name, 'ERROR', f"{target_name} not found in KOBE catalog")

    INITIAL_TEFF = seeds['teff']
    INITIAL_MH   = seeds['mh']
    M_CAT        = seeds['M_cat']
    eM_CAT       = seeds['eM_cat']

    if not (np.isfinite(M_CAT) and M_CAT > 0):
        return (target_name, 'ERROR',
                f"catalog mass missing or invalid (M_cat={M_CAT})")

    LUMINOSITY = get_vosa_luminosity(target_name)
    if LUMINOSITY is None:
        return (target_name, 'ERROR',
                f"luminosity missing in KOBE catalog")
    L_CAT, eL_CAT = LUMINOSITY

    INITIAL_LOGG, _e_logg, _R = logg_from_catalog(
        M_CAT, L_CAT, INITIAL_TEFF, e_teff=0.0, e_L=eL_CAT)

    INITIAL_PARAMETERS = build_initial_parameters(
        INITIAL_TEFF, INITIAL_LOGG, INITIAL_MH)

    FREE_PARAMS = ["teff", "MH", "vsini"]

    print(colored(
        f"\n[batch] {target_name}: "
        f"Teff={INITIAL_TEFF:.0f} K, [Fe/H]={INITIAL_MH:+.2f}, "
        f"logg_init={INITIAL_LOGG:.3f}", 'cyan'))

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        main(target_name,
             diagnostic_plots=diagnostic_plots,
             output_dir=output_dir)
        return (target_name, 'OK', None)
    except SystemExit as exc:
        msg = f"Pipeline aborted (SystemExit code={exc.code})"
        print(colored(f"[batch] {target_name}: {msg}", 'red'))
        return (target_name, 'ERROR', msg)
    except Exception:
        tb = _tb.format_exc()
        print(colored(f"[batch] {target_name}: unhandled exception\n{tb}", 'red'))
        return (target_name, 'ERROR', tb)


def _run_targets_batch(targets, n_workers, diagnostic_plots, output_dir, label="batch"):
    """Run one pipeline pass over targets, optionally in forked workers."""
    import multiprocessing as _mp

    n_workers = max(1, min(n_workers, 6))
    print(colored(
        f"\n[{label}] {len(targets)} target(s)  ·  {n_workers} worker(s)", 'blue'))
    for t in targets:
        print(colored(f"         {t}", 'blue'))

    worker_args = [(t, diagnostic_plots, output_dir) for t in targets]
    if n_workers == 1:
        results = [_synthesis_worker(a) for a in worker_args]
    else:
        ctx = _mp.get_context('fork')
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_synthesis_worker, worker_args)

    print(colored(f"\n{'─'*50}", 'blue'))
    print(colored(f"  {label} summary", 'blue'))
    print(colored(f"{'─'*50}", 'blue'))
    n_ok = sum(1 for _, s, _ in results if s == 'OK')
    n_err = len(results) - n_ok
    for tgt, status, msg in results:
        _col = 'green' if status == 'OK' else 'red'
        _msg = f"  {'✓' if status == 'OK' else '✗'}  {tgt:<30s}  {status}"
        if msg and status != 'OK':
            _msg += f"  ({msg.splitlines()[0][:60]})"
        print(colored(_msg, _col))
    print(colored(f"\n  {n_ok}/{len(results)} completed successfully.", 'blue'))
    return results


def _safe_float(value):
    try:
        val = float(value)
        return val if np.isfinite(val) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _summarize_line_subset_results(targets, csv_path=LINE_SUBSET_RESULTS_CSV):
    """Print latest line-subset rows and deltas against all_diag."""
    import csv
    from collections import OrderedDict

    if not os.path.exists(csv_path):
        print(colored(f"  No line-subset CSV found: {csv_path}", 'yellow'))
        return

    targets = set(targets)
    latest = OrderedDict()
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('target') not in targets:
                continue
            subset = row.get('LineSubset', '')
            if subset not in LINE_SUBSET_TEST_MODES:
                continue
            latest[(row['target'], subset)] = row

    if not latest:
        print(colored("  No matching line-subset rows to summarize.", 'yellow'))
        return

    print(colored(f"\n{'='*84}", 'green'))
    print(colored("  LINE-SUBSET SUMMARY  (latest rows; deltas vs all_diag)", 'green'))
    print(colored(f"{'='*84}", 'green'))
    print(colored(
        f"  {'Target':<10} {'Subset':<16} {'N':>4} "
        f"{'Teff':>7} {'[Fe/H]':>8} {'Δ[Fe/H]':>8} "
        f"{'χ²_red':>8} {'Δχ²':>8}",
        'cyan'))
    print(colored(f"  {'─'*80}", 'cyan'))

    for target in sorted(targets):
        base = latest.get((target, 'all_diag'))
        base_feh = _safe_float(base.get('Fe_H')) if base else np.nan
        base_chi = _safe_float(base.get('chi2_red')) if base else np.nan
        for subset in LINE_SUBSET_TEST_MODES:
            row = latest.get((target, subset))
            if row is None:
                continue
            feh = _safe_float(row.get('Fe_H'))
            chi = _safe_float(row.get('chi2_red'))
            d_feh = feh - base_feh if np.isfinite(base_feh) else np.nan
            d_chi = chi - base_chi if np.isfinite(base_chi) else np.nan
            print(colored(
                f"  {target:<10} {subset:<16} "
                f"{row.get('NLines', ''):>4} "
                f"{_safe_float(row.get('Teff_K')):7.0f} "
                f"{feh:8.3f} {d_feh:8.3f} "
                f"{chi:8.4f} {d_chi:8.4f}",
                'cyan'))


def _run_line_subset_test(targets, n_workers, diagnostic_plots, output_dir):
    """Run all diagnostic line subsets for the given science targets."""
    global LINE_SUBSET_MODE

    print(colored(f"\n{'='*72}", 'magenta'))
    print(colored("  LINE-SUBSET DIAGNOSTIC TEST", 'magenta'))
    print(colored(f"  Targets : {', '.join(targets)}", 'magenta'))
    print(colored(f"  Subsets : {', '.join(LINE_SUBSET_TEST_MODES)}", 'magenta'))
    print(colored(f"  Output  : {LINE_SUBSET_RESULTS_CSV}", 'magenta'))
    print(colored(f"{'='*72}", 'magenta'))

    all_results = []
    previous_mode = LINE_SUBSET_MODE
    try:
        for subset in LINE_SUBSET_TEST_MODES:
            LINE_SUBSET_MODE = subset
            print(colored(
                f"\n[line-subset] mode={subset}  → {LINE_SUBSET_RESULTS_CSV}",
                'yellow'))
            results = _run_targets_batch(
                targets, n_workers, diagnostic_plots, output_dir,
                label=f"line-subset:{subset}")
            all_results.extend((subset, *r) for r in results)
    finally:
        LINE_SUBSET_MODE = previous_mode

    _summarize_line_subset_results(targets)
    n_err = sum(1 for _, _, status, _ in all_results if status != 'OK')
    return all_results, n_err


def _summarize_ionization_results(targets, csv_path=LINE_SUBSET_RESULTS_CSV):
    """Print latest Fe I vs Fe II diagnostic rows and ionization difference."""
    import csv
    from collections import OrderedDict

    if not os.path.exists(csv_path):
        print(colored(f"  No line-subset CSV found: {csv_path}", 'yellow'))
        return

    targets = set(targets)
    latest = OrderedDict()
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('target') not in targets:
                continue
            subset = row.get('LineSubset', '')
            if subset in ('fe1', 'fe2'):
                latest[(row['target'], subset)] = row

    print(colored(f"\n{'='*86}", 'green'))
    print(colored("  IONIZATION SUMMARY  (latest rows; Δion = [Fe/H]FeI - [Fe/H]FeII)", 'green'))
    print(colored(f"{'='*86}", 'green'))
    print(colored(
        f"  {'Target':<10} {'N_FeI':>5} {'N_FeII':>6} "
        f"{'FeI':>8} {'FeII':>8} {'Δion':>8} "
        f"{'TeffI':>7} {'TeffII':>7} {'χ²I':>7} {'χ²II':>7}",
        'cyan'))
    print(colored(f"  {'─'*82}", 'cyan'))

    deltas = []
    for target in sorted(targets):
        fe1 = latest.get((target, 'fe1'))
        fe2 = latest.get((target, 'fe2'))
        if fe1 is None or fe2 is None:
            print(colored(f"  {target:<10} missing fe1/fe2 result", 'yellow'))
            continue

        feh1 = _safe_float(fe1.get('Fe_H'))
        feh2 = _safe_float(fe2.get('Fe_H'))
        dion = feh1 - feh2 if np.isfinite(feh1) and np.isfinite(feh2) else np.nan
        if np.isfinite(dion):
            deltas.append(dion)
        print(colored(
            f"  {target:<10} "
            f"{fe1.get('NLines', ''):>5} {fe2.get('NLines', ''):>6} "
            f"{feh1:8.3f} {feh2:8.3f} {dion:8.3f} "
            f"{_safe_float(fe1.get('Teff_K')):7.0f} "
            f"{_safe_float(fe2.get('Teff_K')):7.0f} "
            f"{_safe_float(fe1.get('chi2_red')):7.4f} "
            f"{_safe_float(fe2.get('chi2_red')):7.4f}",
            'cyan'))

    if deltas:
        print(colored(f"  {'─'*82}", 'cyan'))
        print(colored(
            f"  Mean Δion = {np.mean(deltas):+.3f} dex   "
            f"Std = {np.std(deltas):.3f} dex   N = {len(deltas)}",
            'green'))


def _format_fe2_line_label(mode):
    wave_nm = _fe2_line_mode_wave(mode)
    return f"{wave_nm:.4f}" if wave_nm is not None else mode


def _load_fe2_diagnostic_line_metadata(half_width_nm=0.03):
    """Read static VALD metadata around the Fe II diagnostic windows."""
    import csv

    metadata = {
        label: {
            'wave_nm': wave_nm,
            'loggf': np.nan,
            'chi_eV': np.nan,
            'depth': np.nan,
            'blend_count': 0,
            'top_blends': '',
        }
        for label, wave_nm in FE2_DIAGNOSTIC_LINES
    }
    if not os.path.exists(ATOMIC_LINELIST):
        return metadata

    nearby = {label: [] for label, _ in FE2_DIAGNOSTIC_LINES}
    with open(ATOMIC_LINELIST, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for row in reader:
            try:
                wave_nm = float(row.get('wave_nm', 'nan'))
            except ValueError:
                continue
            for label, center_nm in FE2_DIAGNOSTIC_LINES:
                if abs(wave_nm - center_nm) <= half_width_nm:
                    try:
                        depth = float(row.get('theoretical_depth', 'nan'))
                    except ValueError:
                        depth = np.nan
                    nearby[label].append({
                        'element': row.get('element', ''),
                        'wave_nm': wave_nm,
                        'loggf': _safe_float(row.get('loggf')),
                        'chi_eV': _safe_float(row.get('lower_state_eV')),
                        'depth': depth,
                    })

    for label, center_nm in FE2_DIAGNOSTIC_LINES:
        rows = nearby[label]
        central = [
            r for r in rows
            if (r['element'].strip().lower() == 'fe 2' and
                abs(r['wave_nm'] - center_nm) <= FE2_LINE_TOL_NM)
        ]
        if central:
            best = min(central, key=lambda r: abs(r['wave_nm'] - center_nm))
            metadata[label].update({
                'loggf': best['loggf'],
                'chi_eV': best['chi_eV'],
                'depth': best['depth'],
            })
        blends = [
            r for r in rows
            if not (r['element'].strip().lower() == 'fe 2' and
                    abs(r['wave_nm'] - center_nm) <= FE2_LINE_TOL_NM)
        ]
        blends = sorted(
            blends,
            key=lambda r: -r['depth'] if np.isfinite(r['depth']) else np.inf)
        metadata[label]['blend_count'] = len(blends)
        metadata[label]['top_blends'] = '; '.join(
            f"{r['element']}@{r['wave_nm']:.4f}"
            for r in blends[:3]
        )
    return metadata


def _print_fe2_line_metadata():
    metadata = _load_fe2_diagnostic_line_metadata()
    print(colored(f"\n{'='*92}", 'green'))
    print(colored("  FE II DIAGNOSTIC LINE METADATA  (VALD ±0.03 nm blend census)", 'green'))
    print(colored(f"{'='*92}", 'green'))
    print(colored(
        f"  {'Mode':<11} {'λ[nm]':>9} {'loggf':>8} {'χlow':>7} "
        f"{'depth':>8} {'Nblend':>7}  {'top blends':<30}",
        'cyan'))
    print(colored(f"  {'─'*88}", 'cyan'))
    for label, _ in FE2_DIAGNOSTIC_LINES:
        row = metadata[label]
        print(colored(
            f"  {label:<11} {row['wave_nm']:9.4f} "
            f"{row['loggf']:8.3f} {row['chi_eV']:7.3f} "
            f"{row['depth']:8.3f} {row['blend_count']:7d}  "
            f"{row['top_blends']:<30}",
            'cyan'))


def _summarize_fe2_line_results(targets, csv_path=LINE_SUBSET_RESULTS_CSV):
    """Print latest per-line Fe II rows and deltas against Fe I."""
    import csv
    from collections import OrderedDict

    if not os.path.exists(csv_path):
        print(colored(f"  No line-subset CSV found: {csv_path}", 'yellow'))
        return

    wanted_modes = ('fe1', 'fe2', *FE2_LINE_TEST_MODES)
    targets = set(targets)
    latest = OrderedDict()
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('target') not in targets:
                continue
            subset = row.get('LineSubset', '')
            if subset in wanted_modes:
                latest[(row['target'], subset)] = row

    _print_fe2_line_metadata()

    print(colored(f"\n{'='*104}", 'green'))
    print(colored(
        "  FE II LINE-BY-LINE SUMMARY  (latest rows; Δline = [Fe/H]line - [Fe/H]FeI)",
        'green'))
    print(colored(f"{'='*104}", 'green'))
    print(colored(
        f"  {'Target':<10} {'Line':<11} {'N':>3} "
        f"{'FeI':>8} {'FeII_all':>9} {'FeII_line':>10} {'Δline':>8} "
        f"{'Teff':>7} {'χ²/RMS':>8} {'flags':<12}",
        'cyan'))
    print(colored(f"  {'─'*100}", 'cyan'))

    line_deltas = {mode: [] for mode in FE2_LINE_TEST_MODES}
    for target in sorted(targets):
        fe1 = latest.get((target, 'fe1'))
        fe2 = latest.get((target, 'fe2'))
        feh1 = _safe_float(fe1.get('Fe_H')) if fe1 else np.nan
        feh2 = _safe_float(fe2.get('Fe_H')) if fe2 else np.nan

        for mode in FE2_LINE_TEST_MODES:
            row = latest.get((target, mode))
            if row is None:
                print(colored(
                    f"  {target:<10} {_format_fe2_line_label(mode):<11} "
                    f"{'missing':>45}",
                    'yellow'))
                continue
            feh_line = _safe_float(row.get('Fe_H'))
            delta = feh_line - feh1 if np.isfinite(feh1) else np.nan
            if np.isfinite(delta):
                line_deltas[mode].append(delta)
            print(colored(
                f"  {target:<10} {_format_fe2_line_label(mode):<11} "
                f"{row.get('NLines', ''):>3} "
                f"{feh1:8.3f} {feh2:9.3f} {feh_line:10.3f} {delta:8.3f} "
                f"{_safe_float(row.get('Teff_K')):7.0f} "
                f"{_safe_float(row.get('chi2_red')):8.4f} "
                f"{row.get('quality_flags', ''):<12}",
                'cyan'))

    print(colored(f"  {'─'*100}", 'cyan'))
    for mode in FE2_LINE_TEST_MODES:
        vals = line_deltas[mode]
        if vals:
            print(colored(
                f"  Mean Δline {mode} ({_format_fe2_line_label(mode)} nm) = "
                f"{np.mean(vals):+.3f} dex   Std = {np.std(vals):.3f}   N = {len(vals)}",
                'green'))


def _run_ionization_test(targets, n_workers, diagnostic_plots, output_dir):
    """Run only Fe I and Fe II subsets for ionization-balance diagnostics."""
    global LINE_SUBSET_MODE

    modes = ('fe1', 'fe2')
    print(colored(f"\n{'='*72}", 'magenta'))
    print(colored("  FE I / FE II IONIZATION DIAGNOSTIC", 'magenta'))
    print(colored(f"  Targets : {', '.join(targets)}", 'magenta'))
    print(colored(f"  Subsets : {', '.join(modes)}", 'magenta'))
    print(colored(f"  Output  : {LINE_SUBSET_RESULTS_CSV}", 'magenta'))
    print(colored(f"{'='*72}", 'magenta'))

    all_results = []
    previous_mode = LINE_SUBSET_MODE
    try:
        for subset in modes:
            LINE_SUBSET_MODE = subset
            print(colored(
                f"\n[ionization] mode={subset}  → {LINE_SUBSET_RESULTS_CSV}",
                'yellow'))
            results = _run_targets_batch(
                targets, n_workers, diagnostic_plots, output_dir,
                label=f"ionization:{subset}")
            all_results.extend((subset, *r) for r in results)
    finally:
        LINE_SUBSET_MODE = previous_mode

    _summarize_ionization_results(targets)
    n_err = sum(1 for _, _, status, _ in all_results if status != 'OK')
    return all_results, n_err


def _run_fe2_line_test(targets, n_workers, diagnostic_plots, output_dir):
    """Run Fe I, all Fe II, and each diagnostic Fe II line separately."""
    global LINE_SUBSET_MODE

    modes = ('fe1', 'fe2', *FE2_LINE_TEST_MODES)
    print(colored(f"\n{'='*72}", 'magenta'))
    print(colored("  FE II LINE-BY-LINE DIAGNOSTIC", 'magenta'))
    print(colored(f"  Targets : {', '.join(targets)}", 'magenta'))
    print(colored(f"  Subsets : {', '.join(modes)}", 'magenta'))
    print(colored(f"  Output  : {LINE_SUBSET_RESULTS_CSV}", 'magenta'))
    print(colored(f"{'='*72}", 'magenta'))

    all_results = []
    previous_mode = LINE_SUBSET_MODE
    try:
        for subset in modes:
            LINE_SUBSET_MODE = subset
            print(colored(
                f"\n[fe2-line] mode={subset}  → {LINE_SUBSET_RESULTS_CSV}",
                'yellow'))
            results = _run_targets_batch(
                targets, n_workers, diagnostic_plots, output_dir,
                label=f"fe2-line:{subset}")
            all_results.extend((subset, *r) for r in results)
    finally:
        LINE_SUBSET_MODE = previous_mode

    _summarize_fe2_line_results(targets)
    n_err = sum(1 for _, _, status, _ in all_results if status != 'OK')
    return all_results, n_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Spectral synthesis and stellar parameter estimation for CARMENES spectra."
    )
    parser.add_argument("target", nargs='?', default=None,
                        help="Target name, e.g. TOI-1137 "
                             "(not required with --gbs-test or --gbs-star)")
    parser.add_argument("--gbs-test", action="store_true",
                        help="Run pipeline on all GBS stars and print a validation table "
                             "(ΔTeff, Δ[M/H], Δvmic vs. reference). "
                             "Does NOT save any calibration file.")
    parser.add_argument("--gbs-star", nargs='+', default=None,
                        metavar='NAME',
                        help="Restrict --gbs-test to specific stars. "
                             "Names must match GBS_CATALOG keys "
                             "(61CygA, HD36003).")
    parser.add_argument("--snr", action="store_true", help="Print S/N diagnostics and exit")
    parser.add_argument("--plot-spectrum", action="store_true",
                        help="Show the preprocessed spectrum interactively (telluric-cleaned, "
                             "windowed, normalised, RV-corrected) and exit without fitting.")
    parser.add_argument("--diagnostic-plots", action="store_true",
                        help="Generate all diagnostic PDFs: "
                             "{target}_diagnostic_rv_check.png (RV line positions), "
                             "{target}_check_mask.pdf (CCF mask overlay), "
                             "{target}_check_spectrum.pdf (observed vs synthetic), "
                             "{target}_hrd_parsec.pdf (PARSEC HRD, requires luminosity in catalog).")
    parser.add_argument("--line-subset", choices=LINE_SUBSET_CHOICES, default='all',
                        help="Optional Fe-line diagnostic subset. Default 'all' is "
                             "the production setup with weak Fe I lines and selected "
                             "problematic medium Fe I lines excluded. "
                             "Use 'all_diag' for the unfiltered baseline written to "
                             "kobe_line_subset_results.csv; all other non-'all' modes "
                             "also write there.")
    parser.add_argument("--line-subset-test", action="store_true",
                        help="Run all Fe-line diagnostic subsets "
                             f"({', '.join(LINE_SUBSET_TEST_MODES)}) for the "
                             "target or --batch targets. Results are appended to "
                             "output/kobe_line_subset_results.csv and summarized "
                             "against all_diag.")
    parser.add_argument("--ionization-test", action="store_true",
                        help="Run only Fe I and Fe II subsets and summarize "
                             "[Fe/H]FeI - [Fe/H]FeII. Results are appended to "
                             "output/kobe_line_subset_results.csv.")
    parser.add_argument("--fe2-line-test", action="store_true",
                        help="Run Fe I, all Fe II, and each diagnostic Fe II "
                             "line separately. With science targets, results "
                             "are appended to output/kobe_line_subset_results.csv. "
                             "With --gbs-test/--gbs-star, prints the same "
                             "line-by-line validation against literature [M/H].")
    parser.add_argument("--continuum-mode", choices=CONTINUUM_MODE_CHOICES,
                        default='baseline',
                        help="Continuum normalization stress-test mode. Default "
                             "'baseline' reproduces the production spline. "
                             "Non-baseline science results are written to "
                             "output/kobe_continuum_stress_results.csv by default.")
    parser.add_argument("--batch", nargs='+', metavar='NAME', default=None,
                        help="Run the pipeline on multiple targets in parallel. "
                             "E.g.: --batch TOI-1137 TOI-700 TOI-776  "
                             "Results are appended to kobe_results.csv as each "
                             "target finishes.")
    parser.add_argument("--workers", type=int, default=1, metavar='N',
                        help="Number of parallel workers for --batch (default: 1, max recommended: 6). "
                             "Workers inherit the already-loaded iSpec/Turbospectrum state "
                             "via os.fork() — no re-import overhead.")
    parser.add_argument("--vmic-fixed", type=float, default=None, metavar='KM_S',
                        help="Override the fixed cool-K vmic value for sensitivity tests. "
                             "When used, results are written to "
                             "output/kobe_vmic_sensitivity_results.csv by default.")
    parser.add_argument("--results-csv", type=str, default=None, metavar='PATH',
                        help="Override the cumulative results CSV path.")
    args = parser.parse_args()

    target = args.target

    diagnostic_plots_mode = args.diagnostic_plots
    snr_mode              = args.snr
    gbs_test_mode         = args.gbs_test
    plot_spectrum_mode    = args.plot_spectrum
    line_subset_test_mode = args.line_subset_test
    ionization_test_mode  = args.ionization_test
    fe2_line_test_mode    = args.fe2_line_test
    LINE_SUBSET_MODE      = args.line_subset
    CONTINUUM_MODE        = args.continuum_mode

    if args.results_csv is not None:
        RESULTS_CSV = args.results_csv

    if args.vmic_fixed is not None:
        if args.vmic_fixed <= 0:
            parser.error("--vmic-fixed must be positive.")
        VMIC_COOL_KDWARF = float(args.vmic_fixed)
        if args.results_csv is None:
            RESULTS_CSV = os.path.join(OUTPUT_DIR, 'kobe_vmic_sensitivity_results.csv')
        print(colored(
            f"  vmic sensitivity mode: fixed vmic={VMIC_COOL_KDWARF:.2f} km/s  "
            f"→ results will be written to {RESULTS_CSV}",
            'yellow'))

    if LINE_SUBSET_MODE != 'all':
        print(colored(
            f"  Line-subset diagnostic mode: {LINE_SUBSET_MODE}  "
            f"→ results will be written to {LINE_SUBSET_RESULTS_CSV}",
            'yellow'))

    if CONTINUUM_MODE != 'baseline':
        if args.results_csv is None and LINE_SUBSET_MODE == 'all' and args.vmic_fixed is None:
            RESULTS_CSV = CONTINUUM_STRESS_RESULTS_CSV
        print(colored(
            f"  Continuum stress mode: {CONTINUUM_MODE}  "
            f"→ non-GBS science results will be written to {RESULTS_CSV}",
            'yellow'))

    if line_subset_test_mode:
        if args.gbs_test or args.gbs_star:
            parser.error("--line-subset-test is currently for science KOBE targets, not --gbs-test.")
        subset_targets = args.batch if args.batch else ([target] if target else [])
        if not subset_targets:
            parser.error("--line-subset-test requires a target or --batch targets.")
        if LINE_SUBSET_MODE != 'all':
            parser.error("--line-subset-test runs its own subset list; do not combine it with --line-subset.")
        _, n_err = _run_line_subset_test(
            subset_targets,
            args.workers,
            diagnostic_plots_mode,
            OUTPUT_DIR)
        sys.exit(0 if n_err == 0 else 1)

    if ionization_test_mode:
        if args.gbs_test or args.gbs_star:
            parser.error("--ionization-test is currently for science KOBE targets, not --gbs-test.")
        ion_targets = args.batch if args.batch else ([target] if target else [])
        if not ion_targets:
            parser.error("--ionization-test requires a target or --batch targets.")
        if LINE_SUBSET_MODE != 'all':
            parser.error("--ionization-test runs its own subset list; do not combine it with --line-subset.")
        _, n_err = _run_ionization_test(
            ion_targets,
            args.workers,
            diagnostic_plots_mode,
            OUTPUT_DIR)
        sys.exit(0 if n_err == 0 else 1)

    if fe2_line_test_mode:
        if LINE_SUBSET_MODE != 'all':
            parser.error("--fe2-line-test runs its own subset list; do not combine it with --line-subset.")
        _gbs_target_match = None
        if target is not None and not args.batch and not args.gbs_star:
            _gbs_target_match = next(
                (k for k in GBS_CATALOG if k.lower() == target.lower()), None)
        if args.gbs_test or args.gbs_star or _gbs_target_match is not None:
            _gbs_filter = args.gbs_star
            if _gbs_filter is None and _gbs_target_match is not None:
                _gbs_filter = [_gbs_target_match]
            n_err = run_gbs_fe2_line_test(star_filter=_gbs_filter)
            sys.exit(0 if n_err == 0 else 1)
        fe2_targets = args.batch if args.batch else ([target] if target else [])
        if not fe2_targets:
            parser.error("--fe2-line-test requires a target, --batch targets, or --gbs-star/--gbs-test.")
        _, n_err = _run_fe2_line_test(
            fe2_targets,
            args.workers,
            diagnostic_plots_mode,
            OUTPUT_DIR)
        sys.exit(0 if n_err == 0 else 1)

    # ── Batch mode ────────────────────────────────────────────────────────────
    # --batch NAME1 NAME2 ...  [--workers N]
    # Loads iSpec once in the parent, then forks N workers.  Each worker sets
    # its own per-target globals and calls main().  Results land in the shared
    # kobe_results.csv via fcntl-locked append_results_csv().
    if args.batch:
        results = _run_targets_batch(
            args.batch, args.workers, diagnostic_plots_mode, OUTPUT_DIR)
        n_err = sum(1 for _, s, _ in results if s != 'OK')
        sys.exit(0 if n_err == 0 else 1)

    # ── Shortcut: bare GBS name as positional target ─────────────────────────
    # Allows `python kobe.spectral_synthesis.py 61CygA` (with or without
    # --gbs-test) to run the GBS validation pipeline on that star.
    # Case-insensitive; resolves to the canonical GBS_CATALOG key.
    if target is not None and not args.gbs_star:
        _gbs_match = next(
            (k for k in GBS_CATALOG if k.lower() == target.lower()), None)
        if _gbs_match is not None:
            print(colored(
                f"  → Target '{target}' matches GBS calibrator '{_gbs_match}'; "
                f"routing to --gbs-test pipeline.", 'cyan'))
            run_gbs_test(star_filter=[_gbs_match])
            sys.exit(0)

    # ── Initial parameter seeds (always from KOBE catalog) ────────────────────
    # All values come from the catalog. logg is computed physically:
    #   R = SB(L_cat, Teff_cat),  logg_init = logg☉ + log10(M_cat) − 2·log10(R)
    # M_cat, L_cat are catalog constants used throughout the pipeline.
    if target is not None:
        seeds = get_catalog_seeds(target)
        if seeds is None:
            print(colored(
                f"ERROR: {target} not found in KOBE catalog — "
                f"cannot determine Teff or [Fe/H].", 'red'))
            sys.exit(1)
        INITIAL_TEFF = seeds['teff']
        INITIAL_MH   = seeds['mh']
        M_CAT        = seeds['M_cat']
        eM_CAT       = seeds['eM_cat']

        if not (np.isfinite(M_CAT) and M_CAT > 0):
            print(colored(
                f"ERROR: catalog mass (M_cat) missing or invalid for {target} — "
                f"cannot compute physical logg.", 'red'))
            sys.exit(1)

        LUMINOSITY = get_vosa_luminosity(target)
        if LUMINOSITY is None:
            print(colored(
                f"ERROR: luminosity (L_cat) missing for {target} in KOBE catalog — "
                f"cannot compute physical logg.", 'red'))
            sys.exit(1)
        L_CAT, eL_CAT = LUMINOSITY

        INITIAL_LOGG, _e_logg_init, _R_init = logg_from_catalog(
            M_CAT, L_CAT, INITIAL_TEFF, e_teff=0.0, e_L=eL_CAT)

        print(colored(
            f"  Seeds from KOBE catalog: "
            f"Teff={INITIAL_TEFF:.0f} K,  [Fe/H]={INITIAL_MH:+.2f},  "
            f"M={M_CAT:.3f} M☉,  L={L_CAT:.4f} ± {eL_CAT:.4f} L☉",
            'cyan'))
        print(colored(
            f"  logg_init  = {INITIAL_LOGG:.3f} ± {_e_logg_init:.3f}  "
            f"(R_init={_R_init:.3f} R☉,  Teff_cat={INITIAL_TEFF:.0f} K,  "
            f"M_cat={M_CAT:.3f} M☉,  SB)",
            'cyan'))
    else:
        LUMINOSITY = None
        M_CAT      = np.nan
        eM_CAT     = np.nan
        L_CAT      = np.nan
        eL_CAT     = np.nan

    INITIAL_PARAMETERS = build_initial_parameters(INITIAL_TEFF, INITIAL_LOGG, INITIAL_MH)

    # ── FREE_PARAMS ────────────────────────────────────────────────────────────
    # logg is NEVER free — always physically fixed during the fit.
    # vmic is fixed at VMIC_COOL_KDWARF for all KOBE targets.
    # vmac is always hard-fixed at 0.0 km/s.
    # vsini is hard-fixed inside _run_parameter_fit.
    FREE_PARAMS = ["teff", "MH", "vsini"]

    _vmic_c, _vmic_lo, _vmic_hi, _vmic_fixed = vmic_zone_strategy(INITIAL_TEFF, INITIAL_LOGG)
    _vmac_c = vmac_zone_anchor(INITIAL_TEFF)
    _vmic_status = f"FIXED ({VMIC_COOL_KDWARF:.2f} km/s)"
    print(colored(
        f"  vmic seed: {_vmic_c:.2f} km/s  ({_vmic_status})  "
        f"(Teff={INITIAL_TEFF:.0f} K, logg={INITIAL_LOGG:.2f})", 'cyan'))
    print(colored(
        f"  vmac anchor (FIXED): {_vmac_c:.2f} km/s  "
        f"(Teff={INITIAL_TEFF:.0f} K)", 'cyan'))

    # ── GBS validation test (standalone — no science target needed) ───────────
    # `--gbs-star` alone also triggers the GBS test, filtered to that star.
    if gbs_test_mode or args.gbs_star:
        run_gbs_test(star_filter=args.gbs_star)
        sys.exit(0)

    # All other modes require a science target
    if target is None:
        parser.error(
            "A target name is required unless --gbs-test or --gbs-star is specified.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(colored(
        f"Initial guess: Teff={INITIAL_TEFF:.4f} K, logg={INITIAL_LOGG:.4f}, [M/H]={INITIAL_MH:.4f}",
        'cyan'
    ))
    print(colored(f"Output directory: {os.path.abspath(OUTPUT_DIR)}", 'cyan'))

    if snr_mode:
        fits_file = os.path.join(SPECTRA_DIR, f"{target}_merged.fits")

        if not os.path.exists(fits_file):
            print(colored(f"ERROR: File not found: {fits_file}", 'red'))
            sys.exit(1)

        print(colored(f"\nSNR MODE: {target}", 'blue'))
        spectrum = read_cafe_spectrum(fits_file)
        print_snr_summary(spectrum, title=f"SNR summary for {target}")
        sys.exit(0)

    if plot_spectrum_mode:
        import matplotlib as _mpl
        # Force an interactive backend (try macOS-native first, then Tk, then Qt5)
        for _backend in ('MacOSX', 'TkAgg', 'Qt5Agg', 'WXAgg'):
            try:
                _mpl.use(_backend)
                break
            except Exception:
                continue
        import matplotlib.pyplot as _plt
        import matplotlib.ticker as _ticker

        fits_file = os.path.join(SPECTRA_DIR, f"{target}_merged.fits")
        if not os.path.exists(fits_file):
            print(colored(f"ERROR: File not found: {fits_file}", 'red'))
            sys.exit(1)

        print(colored(f"\nPLOT-SPECTRUM MODE: {target}", 'blue'))

        # ── Preprocessing (steps 1–5) ────────────────────────────────────────
        _spec = read_cafe_spectrum(fits_file)
        print(f"  [1] Read {len(_spec)} pixels  "
              f"({_spec['waveobs'].min()*10:.1f}–{_spec['waveobs'].max()*10:.1f} Å)")

        if CLEAN_TELLURIC:
            _spec, _bv, _bv_err = clean_telluric_regions(_spec)
            print(f"  [2] Telluric cleaned  (BV={_bv:.2f} km/s)  → {len(_spec)} px")

        _spec, _n_win, _n_pix = apply_synth_windows(_spec)
        print(f"  [3] Synthesis windows applied  → {_n_pix} px ({_n_win} windows)")

        if NORMALIZE_CONTINUUM and CONTINUUM_MODE != 'none':
            _spec, _ = normalize_spectrum_spline(_spec, mode=CONTINUUM_MODE)
            print(f"  [4] Continuum normalised  (median flux = "
                  f"{np.nanmedian(_spec['flux']):.4f})")
        else:
            print(f"  [4] Continuum normalisation skipped ({CONTINUUM_MODE})")

        try:
            _rv, _e_rv = estimate_rv_ccf(apply_rv_window(_spec))
            if ESTIMATE_RV:
                _spec = ispec.correct_velocity(_spec, _rv)
            print(f"  [5] RV corrected  ({_rv:+.3f} ± {_e_rv:.3f} km/s)")
        except Exception as _exc:
            print(colored(f"  [5] RV correction failed: {_exc}", 'yellow'))

        # ── Interactive plot ─────────────────────────────────────────────────
        _wave_aa = _spec['waveobs'] * 10.0   # nm → Å
        _flux    = _spec['flux']
        _err     = _spec['err']

        _fig, _ax = _plt.subplots(figsize=(16, 4))
        _ax.plot(_wave_aa, _flux, lw=0.5, color='steelblue', label='flux')
        _ax.fill_between(_wave_aa, _flux - _err, _flux + _err,
                         alpha=0.25, color='steelblue', label='±σ')
        _ax.axhline(1.0, color='gray', lw=0.6, ls='--')

        _ax.set_xlabel('Wavelength (Å)', fontsize=11)
        _ax.set_ylabel('Normalised flux', fontsize=11)
        _ax.set_title(
            f"{target}  —  preprocessed spectrum  "
            f"(telluric-cleaned · windowed · normalised · RV-corrected)",
            fontsize=11)
        _ax.xaxis.set_minor_locator(_ticker.AutoMinorLocator(5))
        _ax.yaxis.set_minor_locator(_ticker.AutoMinorLocator(4))
        _ax.legend(fontsize=9, loc='upper right')
        _ax.set_ylim(
            max(0.0, float(np.nanpercentile(_flux, 1)) - 0.05),
            float(np.nanpercentile(_flux, 99)) + 0.10)
        _plt.tight_layout()
        _plt.show()
        sys.exit(0)

    # ------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------
    main(
        target,
        diagnostic_plots=diagnostic_plots_mode,
        output_dir=OUTPUT_DIR
    )
