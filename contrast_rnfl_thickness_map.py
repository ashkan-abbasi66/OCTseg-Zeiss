"""
This script enhances visualization of RNFL thickness maps.
Thickness maps generated from raw segmented values often have low contrast
because the minimum and maximum values (mapped to 0 and 255) correspond to
the background and optic disc regions rather than the nerve fiber layer of interest.

The provided functions allow users to adjust contrast and color to produce a high-contrast, visually enhanced RNFL thickness map.


There is also a way to push more pixels into yellow/red:
from matplotlib.colors import PowerNorm
norm = PowerNorm(gamma=0.85)  # 0.7–0.9 makes warm colors appear sooner
plt.imshow(img, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)

"""
from compute_rnfl_thickness_map_batch import compute_rnfl_thickness_map
import numpy as np
import matplotlib

HAVE_BUTTON = True

matplotlib.use("TkAgg")  # use Tk backend to avoid Qt issues
import matplotlib.pyplot as plt

# Widgets
from matplotlib.widgets import EllipseSelector, RangeSlider
try:
    from matplotlib.widgets import Button
    HAVE_BUTTON = True
except Exception:
    HAVE_BUTTON = False

from matplotlib.colors import LinearSegmentedColormap

def make_modified_turbo_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    turbo = plt.cm.get_cmap('turbo', 256)
    turbo_colors = turbo(np.linspace(0, 1, 256)) # Convert the colormap to an array of RGBA values
    # Change the lowest and the highest values
    turbo_colors[0] = [0, 0, 0, 1]   # RGBA for black
    turbo_colors[-1] = [1, 0, 0, 1]  # RGBA for red
    modified_turbo = LinearSegmentedColormap.from_list('modified_turbo', turbo_colors)
    return modified_turbo

def make_softjet_colormap():

    # colors_list = [
    #     (0.0, "#0010ff"),  # dark blue
    #     (0.3, "#00b0ff"),  # cyan-blue
    #     (0.5, "#00ff80"),  # greenish
    #     (0.7, "#ffff00"),  # yellow
    #     (0.9, "#ff8000"),  # orange
    #     (1.0, "#ff0000")  # red
    # ]
    # softjet = LinearSegmentedColormap.from_list("softjet", colors_list, N=256)
    # return softjet

    # Control points: (position, hex color). We widen 0.60–0.85 for yellow/orange.
    # stops = [
    #     (0.00, "#00103f"),  # deep navy
    #     (0.15, "#0047ff"),  # royal blue
    #     (0.33, "#00a5ff"),  # cyan-blue (muted)
    #     (0.50, "#00e080"),  # green (slightly desaturated)
    #     (0.60, "#ffff66"),  # start yellow (lighter)
    #     (0.72, "#ffd24d"),  # warm yellow-orange
    #     (0.85, "#ff9a33"),  # orange (stretched)
    #     (1.00, "#e41e1e"),  # soft red (not neon)
    # ]
    stops = [
        (0.00, "#00103f"),  # deep navy
        (0.10, "#003aa7"),  # dark blue
        (0.22, "#0070ff"),  # blue
        (0.34, "#00b5ff"),  # blue-cyan
        (0.46, "#00d880"),  # green (muted)
        (0.55, "#ffff66"),  # <-- early yellow start
        (0.63, "#ffd24d"),  # yellow-orange
        (0.72, "#ff9a33"),  # orange
        (0.82, "#ff5a2a"),  # orange-red
        (0.92, "#e41e1e"),  # red
        (1.00, "#c01010"),  # deep red (avoids neon)
    ]
    return LinearSegmentedColormap.from_list("softjet_yellow", stops, N=256)


# --------------------------- geometry & masking --------------------------- #
def _ellipse_to_mask(h, w, center, width, height, angle_deg=0.0):
    """
    Build a boolean mask for an ellipse on an HxW grid.
    True means 'inside the ellipse'.
    """
    cx, cy = float(center[0]), float(center[1])   # EllipseSelector gives (x, y)
    rx, ry = float(width) / 2.0, float(height) / 2.0
    theta = np.deg2rad(float(angle_deg))

    y, x = np.mgrid[0:h, 0:w]   # row-major grid

    # translate
    xt = x - cx
    yt = y - cy

    # rotate by -theta to align with ellipse axes
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xr =  cos_t * xt + sin_t * yt
    yr = -sin_t * xt + cos_t * yt

    with np.errstate(divide="ignore", invalid="ignore"):
        inside = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
        inside[np.isnan(inside)] = False
    return inside


def apply_mask(original_map, mask, fill=np.nan, invert=False):
    """
    Apply a boolean mask to original_map.
      - invert=False: mask INSIDE the ellipse (default).
      - invert=True:  mask OUTSIDE the ellipse.
      - fill can be np.nan (recommended) or a numeric value, e.g., 0.0.
    """
    m = ~mask if invert else mask
    out = original_map.astype(float).copy()
    out[m] = fill
    return out


# --------------------------- percentile slider viewer --------------------------- #
def show_with_percentile_slider(img, title="", cmap="gray", init_percentiles=(1.0, 99.0)):
    """
    Display image and histogram/slider in two separate figures.
    The slider window controls the image window in real time.
    Returns (pmin, pmax, vmin, vmax).
    """
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        raise ValueError("No finite values in the provided image.")

    def p2v(pmin, pmax):
        vmin, vmax = np.percentile(finite, (pmin, pmax))
        if vmin == vmax:
            vmax = vmin + 1e-9
        return float(vmin), float(vmax)

    pmin0, pmax0 = init_percentiles
    vmin0, vmax0 = p2v(pmin0, pmax0)
    state = {"pmin": pmin0, "pmax": pmax0, "vmin": vmin0, "vmax": vmax0}

    # --- Figure A: image display ---
    fig_img, ax_img = plt.subplots(figsize=(6, 6))
    im = ax_img.imshow(img, cmap=cmap, vmin=vmin0, vmax=vmax0)
    ax_img.set_title(f"{title}\nContrast: {pmin0:.1f}–{pmax0:.1f}% => [{vmin0:.3g}, {vmax0:.3g}]")
    ax_img.axis("off")
    cb = fig_img.colorbar(im, ax=ax_img, fraction=0.046, pad=0.03)
    cb.set_label("Intensity")

    # --- Figure B: histogram + slider + buttons ---
    fig_ctrl = plt.figure(figsize=(10, 6))
    gs = fig_ctrl.add_gridspec(nrows=3, ncols=1, height_ratios=[3, 1, 0.6], hspace=0.4)
    ax_hist = fig_ctrl.add_subplot(gs[0, 0])
    ax_sl   = fig_ctrl.add_subplot(gs[1, 0])
    ax_btns = fig_ctrl.add_subplot(gs[2, 0])
    ax_btns.axis("off")

    counts, edges, _ = ax_hist.hist(finite, bins=256, alpha=0.65)
    ax_hist.set_title("Histogram (finite values only)")
    ax_hist.set_xlabel("Intensity")
    ax_hist.set_ylabel("Count")
    vline_min = ax_hist.axvline(vmin0, linestyle='--', linewidth=1.5)
    vline_max = ax_hist.axvline(vmax0, linestyle='--', linewidth=1.5)

    # Slider
    rslider = RangeSlider(ax=ax_sl,
                          label="",
                          valmin=0.0, valmax=100.0,
                          valinit=(pmin0, pmax0),
                          valstep=0.1)

    # Update handler
    def apply_update(pmin, pmax):
        vmin, vmax = p2v(pmin, pmax)
        im.set_clim(vmin, vmax)
        vline_min.set_xdata([vmin, vmin])
        vline_max.set_xdata([vmax, vmax])
        ax_img.set_title(f"{title}\n{pmin:.1f}–{pmax:.1f}% => [{vmin:.3g}, {vmax:.3g}]")
        rslider.valtext.set_text(f"{pmin:.1f}–{pmax:.1f}%")
        state.update(pmin=pmin, pmax=pmax, vmin=vmin, vmax=vmax)
        fig_img.canvas.draw_idle()
        fig_ctrl.canvas.draw_idle()

    def on_slider_change(_):
        pmin, pmax = rslider.val
        apply_update(pmin, pmax)

    rslider.on_changed(on_slider_change)

    def on_key(event):
        if event.key == "enter":
            plt.close(fig_ctrl)
            plt.close(fig_img)

    fig_ctrl.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return state["pmin"], state["pmax"], state["vmin"], state["vmax"]



# --------------------------- interactive selection --------------------------- #
def _read_ellipse_geometry(sel, ax):
    """
    Return (center_xy, width, height, angle_deg) for the current EllipseSelector.
    Robust across Matplotlib versions by trying several APIs.
    """
    # 1) Preferred: read from the selection artist
    art = getattr(sel, "_selection_artist", None)
    if art is not None:
        try:
            center = art.get_center()   # (x, y)
            width  = art.get_width()
            height = art.get_height()
            angle  = art.get_angle()
            return center, float(width), float(height), float(angle)
        except Exception:
            pass

    # 2) Direct attributes
    if hasattr(sel, "center") and hasattr(sel, "width") and hasattr(sel, "height"):
        center = sel.center
        width  = float(sel.width)
        height = float(sel.height)
        angle  = float(getattr(sel, "angle", 0.0))
        return center, width, height, angle

    # 3) Fallback: extents (no rotation)
    if hasattr(sel, "extents"):
        x1, x2, y1, y2 = sel.extents
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        width  = abs(x2 - x1)
        height = abs(y2 - y1)
        return center, float(width), float(height), 0.0

    # 4) Last resort: look for an Ellipse patch on the axes
    for p in ax.patches[::-1]:
        if isinstance(p, matplotlib.patches.Ellipse):
            center = p.get_center()
            width  = p.get_width()
            height = p.get_height()
            angle  = p.get_angle()
            return center, float(width), float(height), float(angle)

    raise RuntimeError("Could not read ellipse geometry from EllipseSelector.")


def define_mask_on(original_map):
    """
    Show original_map, let the user draw an ellipse, and return a boolean mask
    that's True **inside** the ellipse.

    Controls:
      - Drag to draw the ellipse (handles are draggable).
      - Press ENTER to confirm (or click 'Confirm' if the button is shown).
      - Press 'r' to reset and redraw.
    """
    fig, ax = plt.subplots()
    title = "Draw ellipse (drag). Press ENTER to finish. Press 'r' to reset."
    if HAVE_BUTTON:
        title = "Draw ellipse (drag). Press ENTER or click 'Confirm'. Press 'r' to reset."
    ax.set_title(title)
    ax.imshow(original_map, cmap="gray")
    plt.tight_layout()

    state = {"center": None, "width": None, "height": None, "angle": 0.0}

    def onselect(eclick, erelease):
        # Geometry will be read on confirm
        pass

    sel = EllipseSelector(
        ax,
        onselect,
        useblit=True,
        interactive=True,
        button=[1],          # left click
        minspanx=5,
        minspany=5,
        props=dict(edgecolor="yellow", linewidth=1.5, fill=False),
    )

    def on_confirm(event=None):
        center, width, height, angle = _read_ellipse_geometry(sel, ax)
        state.update(center=center, width=width, height=height, angle=angle)
        plt.close(fig)

    def on_key(event):
        if event.key == "enter":
            on_confirm()
        elif event.key == "r":
            sel.set_visible(False)
            sel.set_visible(True)
            fig.canvas.draw_idle()

    if HAVE_BUTTON:
        confirm_ax = fig.add_axes([0.83, 0.02, 0.15, 0.06])
        confirm_btn = Button(confirm_ax, "Confirm")
        confirm_btn.on_clicked(on_confirm)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if state["center"] is None:
        raise RuntimeError("No ellipse confirmed. (Press ENTER to finish.)")

    h, w = original_map.shape[:2]
    return _ellipse_to_mask(h, w, state["center"], state["width"], state["height"], state["angle"])

def show_heatmap_with_two_colormaps(heatmap, pmin, pmax):

    cmap1 = "gray"
    # cmap2 = make_modified_turbo_colormap()
    cmap2 = make_softjet_colormap()
    cmap2.set_bad("#c9c9c9")

    # cmap2 = "jet"
    # cmap2 = "nipy_spectral"

    # to ignore np.nan and only consider valid numbers when computing
    finite = heatmap[np.isfinite(heatmap)]
    vmin, vmax = np.percentile(finite, (pmin, pmax))
    print(f"vmin: {vmin}, vmax: {vmax}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(heatmap, cmap=cmap1, vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Original ({pmin:.1f}–{pmax:.1f}%)")
    axs[0].axis("off")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.03)

    im1 = axs[1].imshow(heatmap, cmap=cmap2, vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Masked ({pmin:.1f}–{pmax:.1f}%)")
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.03)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # Load data
    # original numpy file (npy) => the output of the segmentation method (`main_nyupitt`)
    # segmented_volume_path = r"./logs/onh-oct-volumes/predict/my-pretrained-model/example-Optic Disc Cube 200x200-OS-cube_z.img.npy"
    # svol = np.load(segmented_volume_path)

    # compressed version (NPZ) => for uploading to Github
    segmented_volume_path = r"./logs/onh-oct-volumes/predict/my-pretrained-model/example-Optic Disc Cube 200x200-OS-cube_z.img.npz"
    svol = np.load(segmented_volume_path)["svol"]

    heatmap = compute_rnfl_thickness_map(svol)

    plt.figure()
    plt.imshow(heatmap, cmap='gray')
    plt.title("Original map - baseline display")
    plt.axis("off")
    plt.show()

    # Interactive percentile slider for the original map
    pmin, pmax, vmin, vmax = show_with_percentile_slider(
        heatmap,
        title="Original map — choose percentile stretch",
        cmap="gray",
        init_percentiles=(1.0, 99.0)
    )

    show_heatmap_with_two_colormaps(heatmap, pmin, pmax)

    mask = define_mask_on(heatmap)
    heatmap_masked = apply_mask(heatmap, mask, fill=np.nan, invert=False)

    show_heatmap_with_two_colormaps(heatmap_masked, pmin, pmax)

