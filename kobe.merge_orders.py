#!/usr/bin/env python3
#
# Part of the KOBE spectral synthesis pipeline.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# LICENSE file or <https://www.gnu.org/licenses/> for the full license text.
#
"""
kobe.merge_orders.py

Read 2D echelle CARMENES/KOBE template spectra (61 orders × 14797 pixels),
merge overlapping orders into a single 1D spectrum, and save the result
as a FITS file compatible with the spectral synthesis pipeline.

The CARMENES template FITS files have:
  - Extension 'SPEC': 2D flux array (61 orders × 14797 pixels)
  - Extension 'WAVE': 2D wavelength array in ln(λ/Å)

Orders 10–51 contain valid data; orders 0–9 and 52–60 are empty.
Consecutive orders overlap by ~28–41% in wavelength.

Merging strategy:
  1. Convert wavelengths from ln(Å) to Å
  2. Trim a small fraction of each order's edges (configurable)
  3. Resample all orders onto a common uniform-in-Å wavelength grid
  4. In overlap zones, blend with a smooth S/N-weighted ramp
  5. Apply stellar RV correction (SERVAL TARG RV) to shift from barycentric
     to stellar rest frame
  6. Convert vacuum → air wavelengths (Edlén 1953) to match the iSpec/HARPS
     mask convention

Output:
  {TARGET}_merged.fits  with extensions WAVELENGTH (Å, air, stellar rest frame)
  and FLUX.

Usage:
    python kobe.merge_orders.py                   # process all KOBE-* files
    python kobe.merge_orders.py KOBE-001          # process one target
    python kobe.merge_orders.py --plot KOBE-001   # process and plot
"""

import os
import sys
import argparse
import glob
import numpy as np
from astropy.io import fits

# ── Paths ────────────────────────────────────────────────────────────────────
# Directory holding the 2D CARMENES template spectra ({TARGET}_template.fits);
# the merged 1D outputs ({TARGET}_merged.fits) are written here too. Configurable
# via the KOBE_TEMPLATE_DIR environment variable (default below is the original
# author setup).
SPECTRA_DIR = os.environ.get('KOBE_TEMPLATE_DIR', '/Users/ccifuentesr/Library/CloudStorage/Dropbox/DATA/data_kobe_master_spectra')

# ── Merging parameters ───────────────────────────────────────────────────────
ORDER_MIN    = 10     # First valid order
ORDER_MAX    = 51     # Last valid order
EDGE_TRIM    = 0.02   # Fraction of pixels to trim from each edge (per order)
PIXEL_STEP   = 0.01   # Output grid step in Å (CARMENES VIS pixel ~ 0.01 Å)

REFERENCE_LINES_NM = [
    ("Ca II K", 393.366),
    ("Ca II H", 396.847),
    ("Hbeta", 486.135),
    ("Na I D2", 588.995),
    ("Na I D1", 589.592),
    ("Halpha", 656.280),
]


C_KMS = 299792.458   # speed of light [km/s]


def vacuum_to_air(wave_aa):
    """
    Convert vacuum wavelengths to air wavelengths.

    Uses the formula of Edlén (1953), J. Opt. Soc. Am., 43, 339, which gives
    the refractive index of standard dry air (15 °C, 1013.25 mbar):

        n - 1 = 6.4328e-5 + 2.94981e-2 / (146 - σ²) + 2.5540e-4 / (41 - σ²)

    where σ = 1/λ(µm) is the wavenumber in µm⁻¹.  Valid across the visible
    and near-IR (λ > 2000 Å).  Accuracy sufficient for R ~ 100000 spectroscopy.

    Parameters
    ----------
    wave_aa : array — vacuum wavelengths in Å

    Returns
    -------
    wave_air_aa : array — air wavelengths in Å
    """
    sigma = 1e4 / wave_aa   # wavenumber in µm⁻¹  (1 µm = 1e4 Å)
    n = 1.0 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigma**2) + 2.5540e-4 / (41.0 - sigma**2)
    return wave_aa / n


def apply_rv_correction(wave_aa, rv_kms):
    """
    Shift wavelengths from the barycentric frame to the stellar rest frame.

    The SERVAL template is co-added in the barycentric frame; SERVAL TARG RV
    is stored as metadata but NOT applied to the wavelength axis.  This
    function applies the Doppler shift:

        λ_rest = λ_bary / (1 + v_star / c)

    Parameters
    ----------
    wave_aa : array — barycentric wavelengths in Å
    rv_kms  : float — stellar RV in km/s (SERVAL TARG RV; negative = blueshift)

    Returns
    -------
    wave_rest_aa : array — rest-frame wavelengths in Å
    """
    return wave_aa / (1.0 + rv_kms / C_KMS)


def read_2d_spectrum(fits_file):
    """
    Read a CARMENES 2D template FITS file.

    Returns
    -------
    wave_aa : list of 1-D arrays — wavelength in Å per order (trimmed, finite)
    flux    : list of 1-D arrays — flux per order (same trimming)
    header  : FITS primary header
    """
    with fits.open(fits_file) as hdul:
        spec_2d = hdul['SPEC'].data.copy()   # (61, 14797)
        wave_2d = hdul['WAVE'].data.copy()   # (61, 14797) in ln(λ/Å)
        header  = hdul[0].header

    wave_orders = []
    flux_orders = []

    for i in range(ORDER_MIN, ORDER_MAX + 1):
        w_ln = wave_2d[i]
        s    = spec_2d[i]

        # Mask invalid pixels
        good = (w_ln != 0) & np.isfinite(w_ln) & np.isfinite(s) & (s > 0)
        if good.sum() < 100:
            continue

        w_aa = np.exp(w_ln[good])
        s_good = s[good]

        # Trim edges
        n = len(s_good)
        trim = int(n * EDGE_TRIM)
        if trim > 0:
            w_aa   = w_aa[trim:-trim]
            s_good = s_good[trim:-trim]

        wave_orders.append(w_aa)
        flux_orders.append(s_good)

    return wave_orders, flux_orders, header


def estimate_order_snr(flux):
    """
    Quick S/N estimate for one order from the central 50% of pixels.
    Uses MAD of first-differences (robust to absorption lines).
    """
    n = len(flux)
    q1, q3 = n // 4, 3 * n // 4
    seg = flux[q1:q3]
    if len(seg) < 20:
        return 1.0
    diff = np.diff(seg)
    mad = np.median(np.abs(diff - np.median(diff)))
    sigma = 1.4826 * mad / np.sqrt(2)  # MAD → σ, diff adds √2
    median_flux = np.median(seg)
    if sigma <= 0 or not np.isfinite(sigma):
        return 1.0
    return max(median_flux / sigma, 1.0)


def merge_orders(wave_orders, flux_orders):
    """
    Merge a list of (wave_Å, flux) order arrays into a single 1D spectrum.

    In overlap regions between consecutive orders, a smooth linear ramp
    weighted by each order's S/N blends the two contributions.

    Returns
    -------
    wave_merged : 1-D array, wavelength in Å (uniform grid)
    flux_merged : 1-D array, merged flux
    """
    # Global wavelength range
    w_min = min(w.min() for w in wave_orders)
    w_max = max(w.max() for w in wave_orders)

    # Build uniform output grid
    wave_grid = np.arange(w_min, w_max + PIXEL_STEP, PIXEL_STEP)
    n_grid = len(wave_grid)

    # Accumulators for weighted average
    flux_sum   = np.zeros(n_grid)
    weight_sum = np.zeros(n_grid)

    for w_order, f_order in zip(wave_orders, flux_orders):
        snr = estimate_order_snr(f_order)
        weight = snr ** 2  # inverse-variance weighting

        # Build per-pixel ramp weight within this order:
        # 1.0 at center, tapering to 0.0 at edges over the overlap zone.
        n_ord = len(w_order)
        ramp_width = int(n_ord * 0.20)  # 20% ramp at each edge
        ramp = np.ones(n_ord)
        if ramp_width > 1:
            ramp[:ramp_width] = np.linspace(0.0, 1.0, ramp_width)
            ramp[-ramp_width:] = np.linspace(1.0, 0.0, ramp_width)

        # Interpolate this order onto the output grid
        # Only fill the region covered by this order
        idx_lo = np.searchsorted(wave_grid, w_order.min())
        idx_hi = np.searchsorted(wave_grid, w_order.max())
        idx_hi = min(idx_hi, n_grid - 1)

        sub_grid = wave_grid[idx_lo:idx_hi + 1]
        if len(sub_grid) == 0:
            continue

        f_interp = np.interp(sub_grid, w_order, f_order)
        r_interp = np.interp(sub_grid, w_order, ramp)

        w_pixel = weight * r_interp

        flux_sum[idx_lo:idx_hi + 1]   += f_interp * w_pixel
        weight_sum[idx_lo:idx_hi + 1] += w_pixel

    # Normalise
    valid = weight_sum > 0
    flux_merged = np.zeros(n_grid)
    flux_merged[valid] = flux_sum[valid] / weight_sum[valid]

    # Remove uncovered regions
    wave_merged = wave_grid[valid]
    flux_merged = flux_merged[valid]

    return wave_merged, flux_merged


def save_merged_spectrum(wave_aa, flux, header, output_path, rv_kms=0.0):
    """
    Save merged 1D spectrum as FITS with WAVELENGTH and FLUX extensions.
    Preserves the original primary header (metadata, RV, etc.) and records
    the wavelength reference frame and calibration applied.
    """
    primary = fits.PrimaryHDU(header=header)
    for key in ['NAXIS1', 'NAXIS2']:
        if key in primary.header:
            del primary.header[key]

    # Record processing applied to wavelength axis
    primary.header['WFRAME']  = ('stellar', 'Wavelength frame: stellar rest frame')
    primary.header['WVACAIR'] = ('air',     'Wavelength system: air (Edlen 1953)')
    primary.header['RVCORR']  = (rv_kms,    'RV correction applied [km/s] (SERVAL TARG RV)')

    wave_hdu = fits.ImageHDU(data=wave_aa.astype(np.float64), name='WAVELENGTH')
    flux_hdu = fits.ImageHDU(data=flux.astype(np.float64), name='FLUX')

    wave_hdu.header['BUNIT'] = 'Angstrom'
    wave_hdu.header.comments['BUNIT'] = 'Air wavelength, stellar rest frame'
    flux_hdu.header['BUNIT'] = 'adu'
    flux_hdu.header.comments['BUNIT'] = 'Flux (unnormalised)'

    hdul = fits.HDUList([primary, wave_hdu, flux_hdu])
    hdul.writeto(output_path, overwrite=True)


def plot_merged(wave_aa, flux, target_name, output_dir):
    """Optional diagnostic plot of the merged spectrum."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]})

    wave_nm = wave_aa / 10.0

    # Full spectrum
    axes[0].plot(wave_nm, flux, 'k-', lw=0.3)
    axes[0].set_ylabel('Flux (unnormalised)')
    axes[0].set_title(f'{target_name} — merged 1D spectrum')
    axes[0].set_xlim(wave_nm.min(), wave_nm.max())
    for _, line_nm in REFERENCE_LINES_NM:
        if wave_nm.min() <= line_nm <= wave_nm.max():
            axes[0].axvline(line_nm, color='red', linestyle='--', lw=0.8, alpha=0.8)

    # Zoom on overlap region (e.g. around 630 nm)
    zoom_center = 630.0
    zoom_half   = 5.0
    mask = (wave_nm > zoom_center - zoom_half) & (wave_nm < zoom_center + zoom_half)
    if mask.sum() > 10:
        axes[1].plot(wave_nm[mask], flux[mask], 'k-', lw=0.5)
        axes[1].set_ylabel('Flux')
        axes[1].set_title(f'Zoom: {zoom_center - zoom_half:.0f}–{zoom_center + zoom_half:.0f} nm')
        for _, line_nm in REFERENCE_LINES_NM:
            if (zoom_center - zoom_half) <= line_nm <= (zoom_center + zoom_half):
                axes[1].axvline(line_nm, color='red', linestyle='--', lw=0.8, alpha=0.8)
    axes[1].set_xlabel('Wavelength [nm]')

    plt.tight_layout()
    plt.show()
    out_png = os.path.join(output_dir, f'{target_name}_merged.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'    Plot saved: {out_png}')


def process_target(target_name, do_plot=False):
    """Process one KOBE target: read 2D, merge, save 1D."""
    fits_file = os.path.join(SPECTRA_DIR, f'{target_name}_template.fits')
    if not os.path.exists(fits_file):
        print(f'  ERROR: {fits_file} not found')
        return False

    print(f'  {target_name}:')

    # Read
    wave_orders, flux_orders, header = read_2d_spectrum(fits_file)
    n_orders = len(wave_orders)
    print(f'    Orders read: {n_orders} (valid data)')

    # Merge
    wave_aa, flux = merge_orders(wave_orders, flux_orders)
    print(f'    Merged: {len(wave_aa)} pixels, '
          f'{wave_aa.min()/10:.1f}–{wave_aa.max()/10:.1f} nm (vacuum, barycentric)')

    # RV correction: barycentric → stellar rest frame
    rv_kms = float(header.get('SERVAL TARG RV', 0.0))
    wave_aa = apply_rv_correction(wave_aa, rv_kms)
    print(f'    RV correction: {rv_kms:+.3f} km/s (SERVAL TARG RV)')

    # Vacuum → air (Edlén 1953), to match iSpec/HARPS mask convention
    wave_aa = vacuum_to_air(wave_aa)
    print(f'    Vacuum→air: range now {wave_aa.min()/10:.4f}–{wave_aa.max()/10:.4f} nm (air, rest frame)')

    # Save
    output_path = os.path.join(SPECTRA_DIR, f'{target_name}_merged.fits')
    save_merged_spectrum(wave_aa, flux, header, output_path, rv_kms=rv_kms)
    print(f'    Saved: {output_path}')

    # Plot
    if do_plot:
        plot_merged(wave_aa, flux, target_name, SPECTRA_DIR)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge CARMENES 2D echelle orders into 1D spectra.')
    parser.add_argument('targets', nargs='*', default=None,
                        help='Target name(s), e.g. KOBE-001. '
                             'If omitted, all KOBE-*_template.fits are processed.')
    parser.add_argument('--plot', action='store_true',
                        help='Save diagnostic PNG for each target.')
    args = parser.parse_args()

    if args.targets:
        targets = args.targets
    else:
        # Auto-discover all template files
        pattern = os.path.join(SPECTRA_DIR, 'KOBE-*_template.fits')
        files = sorted(glob.glob(pattern))
        targets = [os.path.basename(f).replace('_template.fits', '') for f in files]

    print(f'Processing {len(targets)} target(s)...\n')

    ok, fail = 0, 0
    for t in targets:
        if process_target(t, do_plot=args.plot):
            ok += 1
        else:
            fail += 1

    print(f'\nDone: {ok} merged, {fail} failed.')
