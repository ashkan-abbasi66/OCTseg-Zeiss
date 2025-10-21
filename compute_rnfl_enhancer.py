import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.widgets import EllipseSelector, Button
import matplotlib  # add this near the top

# ellipse_masker_full.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")     # switch backend to Tk

# Required: EllipseSelector
from matplotlib.widgets import EllipseSelector
# Optional: a Confirm button (ENTER also works)
try:
    from matplotlib.widgets import Button
    HAVE_BUTTON = True
except Exception:
    HAVE_BUTTON = False


# --------------------------- geometry & masking --------------------------- #
def _ellipse_to_mask(h, w, center, width, height, angle_deg=0.0):
    """
    Build a boolean mask for an ellipse on an HxW grid.
    True means 'inside the ellipse'.
    """
    # EllipseSelector reports center as (x, y)
    cx, cy = float(center[0]), float(center[1])
    rx, ry = float(width) / 2.0, float(height) / 2.0
    theta = np.deg2rad(float(angle_deg))

    # row-major grid
    y, x = np.mgrid[0:h, 0:w]

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


def show_with_new_dynamic_range(img, title="", cmap="gray"):

    finite = img[np.isfinite(img)] # Ignore np.nan values (masked regions may be filled with np.nan)

    # Option 1: Stretching the range to the minimum and maximum values
    #   => poor contrast when there are extreme outliers
    # vmin, vmax = np.percentile(img.flatten(), (0, 100))

    # Option 2 - Ignoring a percentage (e.g., 1%) of the brightest and the darkest pixels
    vmin, vmax = np.percentile(finite, (1, 99))
    if vmin == vmax:  # degenerate
        vmin = np.min(finite)
        vmax = np.max(finite)

    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.show()

    return vmin, vmax


# --------------------------- interactive selection --------------------------- #
def _read_ellipse_geometry(sel, ax):
    """
    Return (center_xy, width, height, angle_deg) for the current EllipseSelector.
    Robust across Matplotlib versions by trying several APIs.
    """
    # 1) Preferred: read from the selection artist (mpl ~3.4–3.9)
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

    # 2) Direct attributes (some builds)
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
        # Geometry will be read on confirm; nothing needed here.
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

if __name__ == '__main__':

    segmented_volume_path = r"logs/onh-oct-volumes/predict/my-pretrained-model/example-Optic Disc Cube 200x200-OS-cube_z.img.npy"
    svol = np.load(segmented_volume_path)

    # Generate RNFL thickness map by counting RNFL pixels (class 0) in each A-scan
    heatmap = np.zeros((200, 200))
    for i in range(200):  # We have 200 slices (b-scans), each with size of 1024x200
        # Count pixels classified as 0 (RNFL) in the current slice
        bscan = svol[i, :, :]
        slice_counts = np.count_nonzero(bscan == 0, axis=0)
        heatmap[i, :] = slice_counts
    # Normalize thickness values to [0,1] range
    heatmap /= heatmap.reshape(-1).max()

    plt.imshow(heatmap, cmap='gray')
    plt.title("Original map - simple display")
    plt.show()

    # Show original
    show_with_new_dynamic_range(heatmap, title="Original map - excluding extremes")

    # 1) interactively define the ellipse
    mask = define_mask_on(heatmap)

    # 2) apply the mask (mask INSIDE by default). To keep inside and blank out outside, set invert=True.
    new_map = apply_mask(heatmap, mask, fill=np.nan, invert=False)

    # new_map = new_map[np.isfinite(new_map)]

    # 3) show results with recomputed dynamic range
    show_with_new_dynamic_range(new_map, title="Masked map - excluding extremes")
    show_with_new_dynamic_range(new_map, title="Masked map - excluding extremes", cmap="jet")

