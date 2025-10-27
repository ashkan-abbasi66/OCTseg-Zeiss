# rnfl_contrast_robust.py
# Robust global contrast normalization for a single RNFL thickness map
# Methods: MAD/IQR-based limits + gamma/log/asinh tone curves
# Set PATH_TO_MAP to your .npy (or replace the loader to suit your format)

"""

How this works (and why it’s safer than CLAHE)

Limits (MAD/IQR): pick global vmin/vmax from the data distribution while ignoring outliers, so the mapping is stable and interpretable.

Tone curve (gamma/log/asinh): a global, monotonic curve expands useful contrast (e.g., pushes more pixels into the yellow band) without inventing local features.

NaNs/Infs: preserved (NaNs shown as light gray by default).

Comparability: since the mapping is global and monotonic, figures remain meaningful across subjects/visits.
"""


import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # GUI backend that's simple on Windows
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.colors import LinearSegmentedColormap
from compute_rnfl_thickness_map_batch import compute_rnfl_thickness_map


# ---------------------- user settings ---------------------- #
# PATH_TO_MAP = r"your_rnfl_thickness_map.npy"  # <-- set this to your file

USE_LIMITS = "MAD"          # "MAD" or "IQR"
# USE_LIMITS = "IQR"          # "MAD" or "IQR"
MAD_K = 4.0                 # median ± K*MAD (good range: 3–5)
IQR_K = 1.5                 # Tukey fences Q1 - K*IQR, Q3 + K*IQR

TONE = "gamma"              # "gamma", "log", "asinh", or "none"
# TONE = "log"              # "gamma", "log", "asinh", or "none"
GAMMA_CLAMP = (0.7, 1.3)    # gamma clamp if computed automatically
LOG_ALPHA = 5.0             # strength for log(1 + alpha*x)   (try 3–10)
ASINH_ALPHA = 5.0           # strength for asinh(alpha*x)     (try 3–10)

CMAP_CHOICE = "softjet_yellow"  # "softjet_yellow", "turbo", "gray", etc.

SHOW_HIST = True
SAVE_FIG = False
OUT_PATH = "rnfl_contrast_robust.png"
# ----------------------------------------------------------- #


# ---------------------- colormap helpers ---------------------- #
from contrast_rnfl_thickness_map import make_softjet_colormap as make_softjet_yellow

# def make_softjet_yellow():
#     """
#     Jet-like colormap with a widened yellow/orange band and softened blues/greens.
#     """
#     stops = [
#         (0.00, "#00103f"),  # deep navy
#         (0.15, "#0047ff"),  # royal blue
#         (0.33, "#00a5ff"),  # cyan-blue (muted)
#         (0.50, "#00e080"),  # green (slightly desaturated)
#         (0.60, "#ffff66"),  # start yellow (lighter)
#         (0.72, "#ffd24d"),  # warm yellow-orange
#         (0.85, "#ff9a33"),  # orange (stretched)
#         (1.00, "#e41e1e"),  # soft red
#     ]
#     return LinearSegmentedColormap.from_list("softjet_yellow", stops, N=256)


def get_cmap(name="softjet_yellow"):
    if name == "softjet_yellow":
        cm = make_softjet_yellow()
        cm.set_bad("#c9c9c9")  # NaNs appear light gray
        return cm
    try:
        cm = plt.get_cmap(name)
        cm.set_bad("#c9c9c9")
        return cm
    except Exception:
        cm = plt.get_cmap("turbo")
        cm.set_bad("#c9c9c9")
        return cm


# ---------------------- robust limits ---------------------- #
def robust_limits_mad(x, k=4.0):
    """
    vmin/vmax via median ± k*MAD. Ignores NaN/Inf.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite values for MAD limits.")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    vmin = med - k * mad
    vmax = med + k * mad
    # clip to data range to be safe
    return max(vmin, x.min()), min(vmax, x.max())


def robust_limits_iqr(x, k=1.5):
    """
    vmin/vmax via Tukey fences: Q1 - k*IQR, Q3 + k*IQR. Ignores NaN/Inf.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite values for IQR limits.")
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    vmin = q1 - k * iqr
    vmax = q3 + k * iqr
    return max(vmin, x.min()), min(vmax, x.max())


# ---------------------- tone curves ---------------------- #
def normalize_to_unit(x, vmin, vmax):
    """
    Clip to [vmin, vmax] and scale to [0,1], preserving NaNs.
    """
    y = x.astype(float).copy()
    # preserve NaNs
    mask_nan = ~np.isfinite(y)
    y = np.clip(y, vmin, vmax)
    if vmax == vmin:
        vmax = vmin + 1e-12
    y = (y - vmin) / (vmax - vmin)
    y[mask_nan] = np.nan
    return y


def auto_gamma_from_median(x01, clamp=(0.7, 1.3)):
    """
    Choose gamma so the median maps to ~0.5 (global, monotonic).
    x01: values in [0,1] (NaNs ignored)
    """
    finite = x01[np.isfinite(x01)]
    if finite.size == 0:
        return 1.0
    med = np.median(finite)
    eps = 1e-6
    med = np.clip(med, eps, 1 - eps)
    gamma = np.log(0.5) / np.log(med)
    return float(np.clip(gamma, clamp[0], clamp[1]))


def apply_tone_curve(x01, mode="gamma", gamma=None, log_alpha=5.0, asinh_alpha=5.0):
    """
    Apply global monotonic tone curve to [0,1] data (NaNs preserved).
    """
    y = x01.copy()
    mask_nan = ~np.isfinite(y)
    y = np.clip(y, 0.0, 1.0)
    if mode == "none":
        pass
    elif mode == "gamma":
        if gamma is None:
            gamma = auto_gamma_from_median(y)
        y = y ** gamma
    elif mode == "log":
        # y_log = log(1 + a*y) / log(1 + a)
        a = float(log_alpha)
        y = np.log1p(a * y) / np.log1p(a)
    elif mode == "asinh":
        # y_asinh = asinh(a*y) / asinh(a)
        a = float(asinh_alpha)
        y = np.arcsinh(a * y) / np.arcsinh(a)
    else:
        raise ValueError(f"Unknown tone mode: {mode}")
    y[mask_nan] = np.nan
    return y


# ---------------------- main pipeline ---------------------- #
def main():
    # ---- load RNFL thickness map ----
    # rnfl = np.load(PATH_TO_MAP)  # expects a 2D array; adapt if needed
    segmented_volume_path = r"logs/onh-oct-volumes/predict/my-pretrained-model/example-Optic Disc Cube 200x200-OS-cube_z.img.npy"
    svol = np.load(segmented_volume_path)

    rnfl = compute_rnfl_thickness_map(svol)

    if rnfl.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape {rnfl.shape}")

    finite = rnfl[np.isfinite(rnfl)]
    if finite.size == 0:
        raise ValueError("Map contains no finite values.")

    # ---- choose robust vmin/vmax ----
    if USE_LIMITS.upper() == "MAD":
        vmin, vmax = robust_limits_mad(rnfl, k=MAD_K)
        limits_label = f"MAD (k={MAD_K:g})"
    elif USE_LIMITS.upper() == "IQR":
        vmin, vmax = robust_limits_iqr(rnfl, k=IQR_K)
        limits_label = f"IQR (k={IQR_K:g})"
    else:
        raise ValueError("USE_LIMITS must be 'MAD' or 'IQR'.")

    # ---- normalize to [0,1] within [vmin,vmax] ----
    x01 = normalize_to_unit(rnfl, vmin, vmax)

    # ---- apply tone curve ----
    if TONE.lower() == "gamma":
        # auto-gamma by default, clamped
        gamma = auto_gamma_from_median(x01, clamp=GAMMA_CLAMP)
        x_tone = apply_tone_curve(x01, mode="gamma", gamma=gamma)
        tone_label = f"gamma (auto={gamma:.2f})"
    elif TONE.lower() == "log":
        x_tone = apply_tone_curve(x01, mode="log", log_alpha=LOG_ALPHA)
        tone_label = f"log (α={LOG_ALPHA:g})"
    elif TONE.lower() == "asinh":
        x_tone = apply_tone_curve(x01, mode="asinh", asinh_alpha=ASINH_ALPHA)
        tone_label = f"asinh (α={ASINH_ALPHA:g})"
    elif TONE.lower() == "none":
        x_tone = x01
        tone_label = "none"
    else:
        raise ValueError("TONE must be 'gamma', 'log', 'asinh', or 'none'.")

    # ---- choose colormap ----
    cmap = get_cmap(CMAP_CHOICE)

    # ---- optional histogram ----
    if SHOW_HIST:
        fig_h, ax_h = plt.subplots(figsize=(9, 3.2))
        ax_h.hist(finite, bins=256, alpha=0.7)
        ax_h.axvline(vmin, ls="--")
        ax_h.axvline(vmax, ls="--")
        ax_h.set_title("Histogram (finite values only)")
        ax_h.set_xlabel("Intensity")
        ax_h.set_ylabel("Count")
        plt.tight_layout()

    # ---- show result ----
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    im = ax.imshow(x_tone, cmap=cmap, vmin=0.0, vmax=1.0)  # already normalized & toned
    ax.set_title(f"Robust global contrast\nLimits: {limits_label} | Tone: {tone_label}\n"
                 f"[vmin, vmax]=[{vmin:.3g}, {vmax:.3g}]")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Normalized intensity")

    plt.tight_layout()
    if SAVE_FIG:
        fig.savefig(OUT_PATH, dpi=200)
        print(f"Saved: {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
